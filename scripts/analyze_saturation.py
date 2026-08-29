#!/usr/bin/env python3
"""Reproduce the 2026-08-29 saturation analysis (findings F12/F13).

READ-ONLY. Opens nothing but .jsonl record files; writes nothing; needs no
model, no GPU, and no dependencies outside the stdlib. Safe to run while the
daemon is live.

This exists because the original analysis was done in ad-hoc shell heredocs.
The conclusions were recorded but the derivation was not, which is a poor
position for a project whose current problem is a metric that averaged a
collapse away. Anyone should be able to re-derive the eras table, the lock's
first appearance, and the Part D window comparison from the raw records.

Usage:
    .venv/bin/python scripts/analyze_saturation.py
    .venv/bin/python scripts/analyze_saturation.py --data-dir <path> --since 2026-08-18
    .venv/bin/python scripts/analyze_saturation.py --json
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "live" / "qwen3-14b"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def day_of(value: Any) -> str:
    return str(value or "")[:10]


def eras_table(explorations: list[dict], since: str) -> list[dict[str, Any]]:
    """Per-day opened / distinct-topic counts — the table that shows the eras."""
    by_day: dict[str, list[str]] = collections.defaultdict(list)
    for record in explorations:
        day = day_of(record.get("opened_at"))
        topic = str(record.get("topic", "")).strip()
        if day and topic:
            by_day[day].append(topic)
    rows = []
    for day in sorted(by_day):
        if day < since:
            continue
        topics = by_day[day]
        counts = collections.Counter(topics)
        rows.append({
            "day": day,
            "opened": len(topics),
            "distinct": len(counts),
            "diversity": round(len(counts) / len(topics), 3),
            "top_topic": counts.most_common(1)[0][0],
            "top_share": round(counts.most_common(1)[0][1] / len(topics), 3),
        })
    return rows


def find_lock(explorations: list[dict]) -> dict[str, Any]:
    """Longest trailing run of one identical topic, in store order.

    Store order, not timestamp order: the exploration store is append/
    rewrite-in-place JSONL from which no path removes records, and the runtime
    treats file order as recency. Sorting here is what produced the Stage
    22.12 streak defect.
    """
    topics = [
        str(r.get("topic", "")).strip()
        for r in explorations
        if str(r.get("topic", "")).strip()
    ]
    if not topics:
        return {"streak": 0, "topic": "", "first_seen": "", "first_id": ""}
    newest = topics[-1]
    streak = 0
    for topic in reversed(topics):
        if topic != newest:
            break
        streak += 1
    first = next(
        (r for r in explorations if str(r.get("topic", "")).strip() == newest),
        {},
    )
    return {
        "streak": streak,
        "topic": newest,
        "topic_chars": len(newest),
        "first_seen": str(first.get("opened_at", ""))[:19],
        "first_id": str(first.get("exploration_id", ""))[:8],
        "total_occurrences": sum(1 for t in topics if t == newest),
        "total_explorations": len(topics),
    }


def window(records: list[dict], lo: str, hi: str, stamp_key: str) -> list[dict]:
    return [r for r in records if lo <= day_of(r.get(stamp_key)) <= hi]


def compare(
    heartbeats: list[dict],
    explorations: list[dict],
    ladder: list[dict],
    label: str,
    lo: str,
    hi: str,
) -> dict[str, Any]:
    hb = window(heartbeats, lo, hi, "timestamp")
    ex = window(explorations, lo, hi, "opened_at")
    cl = window(ladder, lo, hi, "created_at")
    days = len({day_of(h.get("timestamp")) for h in hb}) or 1
    gap = sum(1 for h in hb if str(h.get("gap_assessment", "")).strip())
    topics = [str(r.get("topic", "")).strip() for r in ex if str(r.get("topic", "")).strip()]
    return {
        "label": label,
        "range": f"{lo}..{hi}",
        "days": days,
        "heartbeats": len(hb),
        "heartbeats_per_day": round(len(hb) / days, 1),
        "gap_assessment_rate": round(gap / len(hb), 3) if hb else 0.0,
        "explorations": len(ex),
        "distinct_topics": len(set(topics)),
        "claim_ladder_records": len(cl),
        "ladder_rung0": sum(1 for r in cl if str(r.get("rung", "")) == "0"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--since", default="2026-08-18",
                    help="first day shown in the per-day table (default 2026-08-18)")
    ap.add_argument("--baseline", default="2026-08-20:2026-08-25",
                    help="pre-arm window as LO:HI")
    ap.add_argument("--arm", default="2026-08-26:2026-08-28",
                    help="arm window as LO:HI (complete days only)")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = ap.parse_args()

    root = Path(args.data_dir)
    explorations = load_jsonl(root / "exploration" / "explorations.jsonl")
    heartbeats = load_jsonl(root / "heartbeats" / "heartbeats.jsonl")
    ladder = load_jsonl(root / "self_state" / "claim_ladder.jsonl")
    proposals = load_jsonl(root / "self_state" / "self_model_proposals.jsonl")

    rows = eras_table(explorations, args.since)
    lock = find_lock(explorations)
    b_lo, b_hi = args.baseline.split(":")
    a_lo, a_hi = args.arm.split(":")
    baseline = compare(heartbeats, explorations, ladder, "baseline", b_lo, b_hi)
    arm = compare(heartbeats, explorations, ladder, "arm", a_lo, a_hi)
    last_write = max(
        (str(p.get("created_at") or p.get("timestamp") or "") for p in proposals),
        default="",
    )

    payload = {
        "data_dir": str(root),
        "per_day": rows,
        "lock": lock,
        "baseline": baseline,
        "arm": arm,
        "self_model_proposals_total": len(proposals),
        "self_model_last_write": last_write[:19],
    }

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(f"data dir: {root}\n")
    print(f"{'day':<12}{'opened':>7}{'distinct':>9}{'divers':>8}{'top%':>7}  top topic")
    print("-" * 96)
    for r in rows:
        print(f"{r['day']:<12}{r['opened']:>7}{r['distinct']:>9}"
              f"{r['diversity']:>8.3f}{r['top_share'] * 100:>6.0f}%  {r['top_topic'][:44]}")

    print(f"\nTRAILING LOCK (store order)")
    print(f"  streak            {lock['streak']} consecutive explorations on one topic")
    print(f"  topic             {lock['topic'][:80]!r}")
    print(f"  topic length      {lock.get('topic_chars')} chars "
          f"(rendered into the prompt truncated at 90 — see runtime.py:924)")
    print(f"  first seen        {lock['first_seen']}  (exploration {lock['first_id']})")
    print(f"  occurrences       {lock['total_occurrences']} of {lock['total_explorations']}")

    print(f"\nWINDOW COMPARISON")
    for w in (baseline, arm):
        print(f"  {w['label']:<9} {w['range']}  {w['days']}d  "
              f"hb={w['heartbeats']} ({w['heartbeats_per_day']}/day)  "
              f"gap={w['gap_assessment_rate']:.3f}  "
              f"expl={w['explorations']} distinct={w['distinct_topics']}  "
              f"ladder={w['claim_ladder_records']} (rung0={w['ladder_rung0']})")

    print(f"\nSELF-MODEL WRITES")
    print(f"  {len(proposals)} proposals total, last written {last_write[:19] or '(none)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
