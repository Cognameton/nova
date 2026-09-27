"""Strength ladder for the scripted opponent pool (Phase 23.1).

Every config plays every other config, both colours, on CALIBRATION seeds.
Ratings are a Bradley-Terry fit (Elo scale, draws count half) anchored at
greedy = 1000. Each config's win rate as X against greedy O is reported
beside it, because that is the number Nova's live record and the plan's
37.5 / 48 / 81 reference lines are stated in.

CPU only; games run in a process pool.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nova.arena.harness import ARENA_DIR, calibration_seeds, play_game, summarize
from nova.arena.opponents import STYLES, ScriptedPlayer, parse_spec, spec_of

DEFAULT_EPSILONS = (0.0, 0.25, 0.5)


def default_configs(epsilons: tuple[float, ...] = DEFAULT_EPSILONS) -> list[str]:
    return [
        spec_of(ScriptedPlayer(s, e))
        for s in STYLES
        for e in (epsilons if s != "random" else (0.0,))
    ]


def _play_block(args: tuple[str, str, int, int]) -> list[tuple[str, str, str]]:
    x_spec, o_spec, n, offset = args
    x, o = parse_spec(x_spec), parse_spec(o_spec)
    return [(x_spec, o_spec, play_game(x, o, seed)["result"]) for seed in calibration_seeds(n, offset)]


def fit_ratings(results: list[tuple[str, str, str]], anchor: str = "greedy", iters: int = 2000) -> dict[str, float]:
    """Bradley-Terry by minorise-maximise; returned on the Elo scale."""
    wins: dict[str, float] = defaultdict(float)
    pair_games: dict[tuple[str, str], int] = defaultdict(int)
    players = set()
    for x, o, result in results:
        players.update((x, o))
        pair_games[(x, o)] += 1
        pair_games[(o, x)] += 1
        if result == "win":
            wins[x] += 1
        elif result == "loss":
            wins[o] += 1
        else:
            wins[x] += 0.5
            wins[o] += 0.5
    strength = {p: 1.0 for p in players}
    for _ in range(iters):
        new = {}
        for i in players:
            denom = sum(pair_games[(i, j)] / 2 / (strength[i] + strength[j])
                        for j in players if j != i and pair_games[(i, j)])
            new[i] = max(wins[i], 0.5) / denom if denom else strength[i]
        norm = new.get(anchor, 1.0)
        strength = {p: v / norm for p, v in new.items()}
    return {p: 1000 + 400 * math.log10(v) for p, v in strength.items()}


def calibrate(
    configs: list[str] | None = None,
    games_per_pair: int = 100,
    vs_greedy_games: int = 1000,
    workers: int | None = None,
    out_dir: Path | None = ARENA_DIR / "calibration",
) -> dict[str, Any]:
    configs = configs or default_configs()
    tasks = []
    for i, x in enumerate(configs):
        for j, o in enumerate(configs):
            if i != j:
                tasks.append((x, o, games_per_pair, (i * len(configs) + j) * games_per_pair))
    vs_offset = len(configs) ** 2 * games_per_pair
    vs_tasks = [(x, "greedy", vs_greedy_games, vs_offset + k * vs_greedy_games) for k, x in enumerate(configs)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rr = [r for block in pool.map(_play_block, tasks) for r in block]
        vs = list(pool.map(_play_block, vs_tasks))
    ratings = fit_ratings(rr)
    vs_greedy = {}
    for block in vs:
        games = [{"result": r, "final_score": {"X": 0, "O": 0}, "forfeits": 0} for _, _, r in block]
        s = summarize(games)
        vs_greedy[block[0][0]] = {"win_rate": s["win_rate"], "ci95": s["win_rate_ci95"], "games": s["games"]}
    ladder = sorted(
        ({"spec": c, "elo": round(ratings[c], 1), "x_vs_greedy": vs_greedy[c]} for c in configs),
        key=lambda row: row["elo"],
    )
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "games_per_pair": games_per_pair,
        "vs_greedy_games": vs_greedy_games,
        "round_robin_games": len(rr),
        "ladder": ladder,
    }
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "ladder.json").write_text(json.dumps(report, indent=2))
    return report
