"""Match runner for the Phase 23 arena.

X moves first, as in live play. The subject under test (a model, or a
scripted stand-in) plays X; opponents play O. Side-to-move alternates, and
a side with no legal move passes while the other can still move.

Randomness is derived per move from (seed, move count, side). For O it is
exactly the live runtime's derivation, random.Random(f"{seed}:{len(moves)}"),
so a greedy O here replays a live game given the same X moves.

Seed ranges are disjoint by construction: EVAL seeds are the only ones any
reported score uses and are never used to generate training data.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

from nova.agent.reversi import (
    NOVA,
    OPPONENT,
    apply_move,
    board_to_rows,
    legal_moves,
    new_board,
    other,
    parse_square,
    score,
)
from nova.arena.opponents import Player

EVAL_SEEDS = range(1_000_000, 1_000_300)
TRAIN_SEED_RANGE = range(2_000_000, 5_000_000)
CALIBRATION_SEED_RANGE = range(5_000_000, 6_000_000)

ARENA_DIR = Path("data/arena")


def train_seeds(n: int, offset: int = 0) -> range:
    """n consecutive training seeds starting `offset` into the train range."""
    start = TRAIN_SEED_RANGE.start + offset
    if n < 0 or offset < 0 or start + n > TRAIN_SEED_RANGE.stop:
        raise ValueError("requested seeds fall outside the training range")
    return range(start, start + n)


def calibration_seeds(n: int, offset: int = 0) -> range:
    start = CALIBRATION_SEED_RANGE.start + offset
    if n < 0 or offset < 0 or start + n > CALIBRATION_SEED_RANGE.stop:
        raise ValueError("requested seeds fall outside the calibration range")
    return range(start, start + n)


def move_rng(seed: int, n_moves: int, side: str) -> random.Random:
    if side == OPPONENT:
        return random.Random(f"{seed}:{n_moves}")
    return random.Random(f"{seed}:{n_moves}:X")


def play_game(x: Player, o: Player, seed: int) -> dict[str, Any]:
    """One full game. Returns a record; `result` is from X's point of view."""
    board = new_board()
    moves: list[dict[str, Any]] = []
    side = NOVA
    started = time.perf_counter()
    while True:
        legal = legal_moves(board, side)
        if not legal:
            if not legal_moves(board, other(side)):
                break
            moves.append({"n": len(moves) + 1, "player": side, "square": "pass", "flipped": 0})
            side = other(side)
            continue
        player = x if side == NOVA else o
        board_before = board_to_rows(board)
        rng = move_rng(seed, len(moves), side)
        square = player.choose(board, side, legal, rng)
        if square not in legal:
            raise RuntimeError(f"{player.name} returned illegal move {square!r} (legal {legal})")
        flipped = apply_move(board, side, *parse_square(square))  # type: ignore[misc]
        record: dict[str, Any] = {"n": len(moves) + 1, "player": side, "square": square, "flipped": flipped}
        detail = getattr(player, "last_detail", None)
        if side == NOVA:
            record["board_before"] = board_before
            record["legal"] = legal
            if detail:
                record.update(detail)
        moves.append(record)
        side = other(side)
    final = score(board)
    diff = final[NOVA] - final[OPPONENT]
    return {
        "seed": seed,
        "x": x.name,
        "o": o.name,
        "moves": moves,
        "final_board": board_to_rows(board),
        "final_score": final,
        "result": "win" if diff > 0 else "loss" if diff < 0 else "draw",
        "forfeits": sum(1 for m in moves if m.get("forfeit")),
        "seconds": round(time.perf_counter() - started, 3),
    }


def wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def summarize(games: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(games)
    wins = sum(g["result"] == "win" for g in games)
    losses = sum(g["result"] == "loss" for g in games)
    draws = n - wins - losses
    lo, hi = wilson(wins, n)
    margin = [g["final_score"][NOVA] - g["final_score"][OPPONENT] for g in games]
    return {
        "games": n,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / n if n else 0.0,
        "win_rate_ci95": [lo, hi],
        "mean_margin": sum(margin) / n if n else 0.0,
        "forfeits": sum(g["forfeits"] for g in games),
    }


@dataclass(slots=True)
class MatchResult:
    run_id: str
    x: str
    o: str
    seeds: list[int]
    games: list[dict[str, Any]] = field(default_factory=list)

    @property
    def summary(self) -> dict[str, Any]:
        return {"run_id": self.run_id, "x": self.x, "o": self.o, **summarize(self.games)}


def run_match(
    x: Player,
    o: Player,
    seeds: Iterable[int],
    *,
    out_dir: Path | None = None,
    run_id: str | None = None,
    progress: bool = False,
) -> MatchResult:
    """Play one game per seed. With out_dir, games stream to
    <out_dir>/<run_id>/games.jsonl as they finish and summary.json is
    written at the end, so a long model run survives interruption."""
    seeds = list(seeds)
    run_id = run_id or f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:6]}"
    result = MatchResult(run_id=run_id, x=x.name, o=o.name, seeds=seeds)
    fh = None
    if out_dir is not None:
        run_dir = Path(out_dir) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        fh = (run_dir / "games.jsonl").open("a", encoding="utf-8")
    try:
        for i, seed in enumerate(seeds, 1):
            game = play_game(x, o, seed)
            result.games.append(game)
            if fh is not None:
                fh.write(json.dumps(game) + "\n")
                fh.flush()
            if progress:
                s = summarize(result.games)
                print(f"[{i}/{len(seeds)}] seed {seed} {game['result']} "
                      f"{game['final_score'][NOVA]}-{game['final_score'][OPPONENT]}  "
                      f"running {s['win_rate']:.1%}", flush=True)
    finally:
        if fh is not None:
            fh.close()
    if out_dir is not None:
        (Path(out_dir) / run_id / "summary.json").write_text(json.dumps(result.summary, indent=2))
    return result
