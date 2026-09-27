"""python -m nova.arena {calibrate,match} ...

  calibrate                         rank the scripted pool (CPU)
  match --x SPEC --o SPEC --seeds eval|train:N[:OFFSET]|calibration:N[:OFFSET]
        SPEC is a scripted spec ("corner:0.25", "lookahead:0:3") or
        model:/path/to.gguf[,lora=/path/adapter.gguf][,scale=1.0][,temp=0.0]
        (model players run on the GPUs: stop the Nova daemon first)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nova.arena.harness import ARENA_DIR, EVAL_SEEDS, calibration_seeds, run_match, train_seeds
from nova.arena.opponents import parse_spec


def _player(spec: str):
    if not spec.startswith("model:"):
        return parse_spec(spec)
    from nova.arena.model_player import ModelPlayer, llama_generate

    path, *opts = spec[len("model:"):].split(",")
    kv = dict(o.split("=", 1) for o in opts)
    name = f"model:{Path(path).name}" + (f"+lora:{Path(kv['lora']).name}" if kv.get("lora") else "")
    generate = llama_generate(path, lora_path=kv.get("lora", ""), lora_scale=float(kv.get("scale", 1.0)))
    return ModelPlayer(generate, name=name, temperature=float(kv.get("temp", 0.0)))


def _seeds(text: str):
    if text == "eval":
        return EVAL_SEEDS
    kind, n, *rest = text.split(":")
    offset = int(rest[0]) if rest else 0
    if kind == "train":
        return train_seeds(int(n), offset)
    if kind == "calibration":
        return calibration_seeds(int(n), offset)
    raise SystemExit(f"unknown seed set {text!r}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m nova.arena")
    sub = ap.add_subparsers(dest="cmd", required=True)
    cal = sub.add_parser("calibrate")
    cal.add_argument("--games-per-pair", type=int, default=100)
    cal.add_argument("--vs-greedy-games", type=int, default=1000)
    cal.add_argument("--workers", type=int, default=None)
    m = sub.add_parser("match")
    m.add_argument("--x", required=True)
    m.add_argument("--o", default="greedy")
    m.add_argument("--seeds", default="eval")
    m.add_argument("--run-id", default=None)
    m.add_argument("--out", default=str(ARENA_DIR / "runs"))
    args = ap.parse_args()

    if args.cmd == "calibrate":
        from nova.arena.calibrate import calibrate

        report = calibrate(games_per_pair=args.games_per_pair,
                           vs_greedy_games=args.vs_greedy_games, workers=args.workers)
        print(f"{'spec':16} {'elo':>7}  X vs greedy (95% CI)")
        for row in report["ladder"]:
            v = row["x_vs_greedy"]
            print(f"{row['spec']:16} {row['elo']:7.0f}  {v['win_rate']:6.1%} "
                  f"({v['ci95'][0]:.1%}-{v['ci95'][1]:.1%})")
        return

    result = run_match(_player(args.x), _player(args.o), _seeds(args.seeds),
                       out_dir=Path(args.out), run_id=args.run_id, progress=True)
    print(json.dumps(result.summary, indent=2))


if __name__ == "__main__":
    main()
