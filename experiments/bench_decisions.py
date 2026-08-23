"""Benchmark where game wall-time goes, from archived decision timings.

Each archived decision carries _dt_ms — wall time since our previous
decision in the same game. That gap covers everything Forge did in
between: engine work, stack resolution, and (when the gap crosses the
opponent's turn) the built-in AI deciding its whole turn. _model_ms is
our own net's forward pass, for comparison.

Attribution buckets per gap, labeled by what happened between the two
decision points:
  same-phase      consecutive decisions in one phase (our priority loop)
  phase-change    later phase of the same turn (combat math lives here)
  turn-change     the gap spans end-of-turn -> opponent's whole turn ->
                  back to us (opponent AI thinking dominates)

Usage:
  python experiments/bench_decisions.py [--last N]   # default 400 games
"""
from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent / "output"


def creatures(state):
    n = 0
    for key in ("my_battlefield", "opp_battlefield"):
        for c in state.get(key, []):
            if isinstance(c, dict) and c.get("cr"):
                n += 1
    return n


def cbucket(n):
    return ("0-2" if n <= 2 else "3-5" if n <= 5 else
            "6-9" if n <= 9 else "10+")


def fmt_row(label, ms, cnt, total_ms):
    share = 100 * ms / max(1, total_ms)
    avg = ms / max(1, cnt)
    return (f"  {label:<28} {ms/1000:>8.1f}s  {share:>5.1f}%"
            f"  {cnt:>6} gaps  avg {avg:>6.0f}ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--last", type=int, default=400)
    args = ap.parse_args()

    idx = OUT / "games_index.jsonl"
    rows = [json.loads(ln) for ln in
            idx.read_text(encoding="utf-8").splitlines()]
    rows = [r for r in rows if r.get("dur_s")][-args.last:]
    if not rows:
        print("no timed games archived yet (dur_s missing) - "
              "wait for instrumented games")
        return

    by_kind = defaultdict(lambda: [0, 0])       # ms, count
    by_transition = defaultdict(lambda: [0, 0])
    by_creatures = defaultdict(lambda: [0, 0])
    model_ms = model_n = 0
    total_ms = 0
    slowest = []
    dur_list = []

    for r in rows:
        with gzip.open(OUT / "games" / r["file"], "rt",
                       encoding="utf-8") as f:
            g = json.load(f)
        dur_list.append(g.get("dur_s", 0))
        prev = None
        for s, _reply in g["decisions"]:
            if "_model_ms" in s:
                model_ms += s["_model_ms"]
                model_n += 1
            dt = s.get("_dt_ms")
            if dt is not None and prev is not None:
                total_ms += dt
                by_kind[s.get("kind", "?")][0] += dt
                by_kind[s.get("kind", "?")][1] += 1
                if s.get("turn") != prev.get("turn"):
                    tr = "turn-change (opp turn inside)"
                elif s.get("phase") != prev.get("phase"):
                    tr = "phase-change (same turn)"
                else:
                    tr = "same-phase (priority loop)"
                by_transition[tr][0] += dt
                by_transition[tr][1] += 1
                cb = cbucket(creatures(s))
                by_creatures[cb][0] += dt
                by_creatures[cb][1] += 1
                slowest.append((dt, r["file"], s.get("turn"),
                                s.get("phase"), s.get("kind"),
                                creatures(s)))
            prev = s

    dur_list.sort()
    n = len(dur_list)
    p = lambda q: dur_list[min(n - 1, int(q * n))]
    print(f"== {n} instrumented games, "
          f"{total_ms/1000:.0f}s of measured gap time ==")
    print(f"game duration: median {p(.5):.0f}s  p90 {p(.9):.0f}s  "
          f"p99 {p(.99):.0f}s  max {dur_list[-1]:.0f}s")
    print(f"our model: {model_n} decisions, avg "
          f"{model_ms/max(1,model_n):.1f}ms "
          f"({model_ms/1000:.0f}s total = "
          f"{100*model_ms/max(1,total_ms):.1f}% of gap time)")

    print("\n-- by what the gap spans --")
    for k, (ms, cnt) in sorted(by_transition.items(),
                               key=lambda x: -x[1][0]):
        print(fmt_row(k, ms, cnt, total_ms))

    print("\n-- by next decision kind --")
    for k, (ms, cnt) in sorted(by_kind.items(), key=lambda x: -x[1][0]):
        print(fmt_row(k, ms, cnt, total_ms))

    print("\n-- by creatures on battlefield (both sides) --")
    for k in ("0-2", "3-5", "6-9", "10+"):
        if k in by_creatures:
            ms, cnt = by_creatures[k]
            print(fmt_row(k, ms, cnt, total_ms))

    print("\n-- 10 slowest single gaps --")
    slowest.sort(reverse=True)
    for dt, f, turn, phase, kind, cr in slowest[:10]:
        print(f"  {dt/1000:>6.1f}s  {f.replace('.json.gz',''):<18} "
              f"turn {turn:<3} {phase or kind:<22} {cr} creatures")


if __name__ == "__main__":
    main()
