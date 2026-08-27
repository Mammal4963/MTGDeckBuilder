"""Freeze the benchmark assets: benchmark_v2.pt + meta_v1 deck set.

meta_v1 is the fixed opponent deck set every benchmark recipe uses.
Selection is for DIVERSITY (stratified across pool sources) and
RELIABILITY (each deck must complete a screening match cleanly when
piloted by the benchmark) - NOT for win rate: deck strength is
cancelled by seat-swapping (pilot-skill) or same-pilot-both-seats
(deck-eval), so balance does not matter, but games that stall or
crash would poison every future measurement.

Usage:
  FORGE_SIM_SERVER=1 PILOT_D=192 PILOT_LAYERS=6 \
      python experiments/build_meta_v1.py --keep 12
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pilot_bridge import ModelPolicy, start_server, run_bridged  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
POOL = HERE / "decks" / "pool_all"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", type=int, default=12)
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--champion", default=str(OUT / "pilot2_rl22.pt"))
    args = ap.parse_args()

    bench = OUT / "benchmark_v2.pt"
    if not bench.exists():
        shutil.copy(args.champion, bench)
        print(f"[freeze] {args.champion} -> {bench}", flush=True)

    # stratify candidates across deck sources for diversity
    groups = {}
    for p in POOL.glob("*.dck"):
        key = re.match(r"([a-z]+)", p.stem)
        groups.setdefault(key.group(1) if key else "misc",
                          []).append(p.stem)
    rng = np.random.default_rng(89)
    cand = []
    per = max(2, (args.keep * 2) // max(1, len(groups)))
    for g, names in sorted(groups.items()):
        take = min(per, len(names))
        cand += list(rng.choice(sorted(names), take, replace=False))
    print(f"screening {len(cand)} candidates from "
          f"{len(groups)} sources", flush=True)

    policy = ModelPolicy(ckpt=bench)
    servers = [start_server(0, policy=policy)
               for _ in range(args.parallel)]
    ports = [s.server_address[1] for s in servers]

    def screen(job):
        wk, deck = job
        out = run_bridged(deck, "fac_g0", args.games,
                          90 + 40 * args.games, ports[wk % len(ports)],
                          player_filter="", quiet=True, worker=wk)
        games = len(re.findall(r"Game Result", out))
        return deck, games

    keep = []
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        for deck, games in pool.map(screen, list(enumerate(cand))):
            ok = games >= args.games       # every game completed
            print(f"[screen] {'OK  ' if ok else 'DROP'} {deck}: "
                  f"{games}/{args.games} games completed", flush=True)
            if ok:
                keep.append(deck)
    for s in servers:
        s.shutdown()
        s.server_close()

    # round-robin across sources for the final diverse pick
    by_src = {}
    for d in keep:
        by_src.setdefault(re.match(r"([a-z]+)", d).group(1),
                          []).append(d)
    final = []
    while len(final) < args.keep and any(by_src.values()):
        for src in sorted(by_src):
            if by_src[src] and len(final) < args.keep:
                final.append(by_src[src].pop(0))
    spec = {"benchmark": bench.name, "arch": "192,6",
            "decks": final, "screened": len(cand),
            "version": "meta_v1"}
    (OUT / "meta_v1.json").write_text(json.dumps(spec, indent=1))
    print(f"[meta_v1] {len(final)} decks: {', '.join(final)}",
          flush=True)


if __name__ == "__main__":
    main()
