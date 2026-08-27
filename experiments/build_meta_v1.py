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
    ap.add_argument("--keep", type=int, default=200)
    ap.add_argument("--games", type=int, default=4)
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--champion", default=str(OUT / "pilot2_rl22.pt"))
    args = ap.parse_args()

    bench = OUT / "benchmark_v2.pt"
    if not bench.exists():
        shutil.copy(args.champion, bench)
        print(f"[freeze] {args.champion} -> {bench}", flush=True)

    # stratify for SOURCE x COLOR x BEHAVIOR diversity
    basics = {"Plains": "W", "Island": "U", "Swamp": "B",
              "Mountain": "R", "Forest": "G"}

    def colors(stem):
        try:
            text = (POOL / f"{stem}.dck").read_text(
                encoding="utf-8", errors="replace")
        except OSError:
            return "?"
        return "".join(c for b, c in basics.items()
                       if re.search(rf"\d+ {b}\b", text)) or "C"

    # behavior proxy: archived avg decisions/game where we have data
    import collections
    ndec = collections.defaultdict(list)
    gidx = OUT / "games_index.jsonl"
    if gidx.exists():
        for ln in gidx.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(ln)
                if r.get("deck"):
                    ndec[r["deck"]].append(r.get("n_dec", 0))
            except json.JSONDecodeError:
                pass

    def behavior(stem):
        v = ndec.get(stem)
        if not v:
            return "unknown"
        avg = sum(v) / len(v)
        return "fast" if avg < 180 else \
            "mid" if avg < 300 else "grindy"

    groups = {}
    for p in POOL.glob("*.dck"):
        src = re.match(r"([a-z]+)", p.stem)
        key = (src.group(1) if src else "misc",
               colors(p.stem), behavior(p.stem))
        groups.setdefault(key, []).append(p.stem)
    rng = np.random.default_rng(89)
    cand = []
    per = max(1, (args.keep * 3 // 2) // max(1, len(groups)))
    for g in sorted(groups):
        names = sorted(groups[g])
        take = min(max(per, 1), len(names))
        cand += list(rng.choice(names, take, replace=False))
    rng.shuffle(cand)
    cand = cand[:args.keep * 3 // 2]
    print(f"screening {len(cand)} candidates from "
          f"{len(groups)} source/color/behavior buckets", flush=True)

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

    # round-robin across the full diversity buckets for the final pick
    by_key = {}
    for d in keep:
        src = re.match(r"([a-z]+)", d)
        key = (src.group(1) if src else "misc",
               colors(d), behavior(d))
        by_key.setdefault(key, []).append(d)
    final = []
    while len(final) < args.keep and any(by_key.values()):
        for key in sorted(by_key):
            if by_key[key] and len(final) < args.keep:
                final.append(by_key[key].pop(0))
    reserve = sorted(set(keep) - set(final))
    spec = {"benchmark": bench.name, "arch": "192,6",
            "decks": final, "reserve": reserve,
            "screened": len(cand), "buckets": len(by_key),
            "version": "meta_v1"}
    (OUT / "meta_v1.json").write_text(json.dumps(spec, indent=1))
    print(f"[meta_v1] {len(final)} decks frozen (order is canonical: "
          f"subsets are the first N), {len(reserve)} in reserve",
          flush=True)


if __name__ == "__main__":
    main()
