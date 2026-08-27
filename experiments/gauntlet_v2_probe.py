"""Build gauntlet-v2: benchmark-piloted opponent decks.

Freezes the current champion as benchmark_v2, then probes sampled pool
decks with the benchmark piloting BOTH seats of fac_roaming vs
candidate. Decks where fac_roaming wins 25-75% are competitive
matchups - a sensitive yardstick with no builtin-AI ceiling (the
baseline is 50% self-play by construction). The 8 keepers are written
to output/gauntlet_v2.json.

Usage:
  FORGE_SIM_SERVER=1 PILOT_D=192 PILOT_LAYERS=6 \
      python experiments/gauntlet_v2_probe.py --candidates 30
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
DECK = "fac_roaming"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=int, default=30)
    ap.add_argument("--games", type=int, default=16)
    ap.add_argument("--keep", type=int, default=8)
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--champion", default=str(OUT / "pilot2_rl22.pt"))
    args = ap.parse_args()

    bench = OUT / "benchmark_v2.pt"
    if not bench.exists():
        shutil.copy(args.champion, bench)
        print(f"[freeze] {args.champion} -> {bench}", flush=True)

    decks = sorted(p.stem for p in POOL.glob("*.dck")
                   if p.stem not in (DECK,))
    rng = np.random.default_rng(83)
    cand = list(rng.choice(decks, args.candidates, replace=False))

    policy = ModelPolicy(ckpt=bench)
    servers = [start_server(0, policy=policy)
               for _ in range(args.parallel)]
    ports = [s.server_address[1] for s in servers]

    def probe(job):
        wk, deck = job
        out = run_bridged(DECK, deck, args.games,
                          90 + 40 * args.games, ports[wk % len(ports)],
                          player_filter="", quiet=True, worker=wk)
        wins = len(re.findall(
            rf"Game Result.*Ai\(1\)-{DECK} has won", out))
        games = len(re.findall(r"Game Result", out))
        return deck, wins, games

    results = []
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        for deck, wins, games in pool.map(
                probe, list(enumerate(cand))):
            wr = wins / max(1, games)
            flag = "KEEP" if 0.25 <= wr <= 0.75 and games >= \
                args.games * 0.75 else "    "
            print(f"[probe] {flag} {deck}: {wins}/{games} "
                  f"= {wr:.0%}", flush=True)
            results.append((deck, wins, games, wr))
    for s in servers:
        s.shutdown()
        s.server_close()

    keep = [r for r in results
            if 0.25 <= r[3] <= 0.75 and r[2] >= args.games * 0.75]
    # closest to 50% first = most sensitive matchups
    keep.sort(key=lambda r: abs(r[3] - 0.5))
    keep = keep[:args.keep]
    spec = {"benchmark": bench.name, "arch": "192,6",
            "decks": [r[0] for r in keep],
            "probe": {r[0]: {"wins": r[1], "games": r[2]}
                      for r in results}}
    (OUT / "gauntlet_v2.json").write_text(json.dumps(spec, indent=1))
    print(f"[gauntlet-v2] {len(keep)} decks: "
          f"{', '.join(r[0] for r in keep)}", flush=True)


if __name__ == "__main__":
    main()
