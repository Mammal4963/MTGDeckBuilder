"""Rate the intrinsic strength of every meta_v1 deck.

Benchmark pilot on BOTH seats (deck-eval logic), random repeated
pairings within the meta: each round shuffles the 200 decks into 100
pairs, each pair plays a 2-game match (Forge alternates play/draw).
ROUNDS=10 -> 20 games per deck, 2000 total.

Output: output/meta_strength.json
  {"t": ..., "games_per_deck": ~20,
   "ratings": [{"deck", "wins", "games", "wr"}, ...]  # sorted desc}

Used to build the strength-skewed evolver meta (top slice) and a
holdout meta for honest reporting.
"""
from __future__ import annotations

import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pilot_bridge import ModelPolicy, start_server, run_bridged  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
PARALLEL = 3     # genesis is running; stay light
ROUNDS = 10      # 2-game matches per deck per round


def main():
    import os
    meta = json.loads((OUT / "meta_v1.json").read_text())
    decks = meta["decks"]
    d, layers = meta["arch"].split(",")
    os.environ["PILOT_D"], os.environ["PILOT_LAYERS"] = d, layers
    pol = ModelPolicy(ckpt=str(OUT / meta["benchmark"]))
    servers = [start_server(0, policy=pol) for _ in range(PARALLEL)]
    ports = [s.server_address[1] for s in servers]

    rng = random.Random(714)
    jobs = []
    for _r in range(ROUNDS):
        order = decks[:]
        rng.shuffle(order)
        for i in range(0, len(order) - 1, 2):
            jobs.append((order[i], order[i + 1]))

    tally = {dk: [0, 0] for dk in decks}
    done = [0]
    t0 = time.time()

    def one(job):
        ji, (da, db) = job
        wk = ji % PARALLEL
        out = run_bridged(da, db, 2, 170, ports[wk],
                          player_filter="", quiet=True, worker=wk)
        wa = len(re.findall(
            rf"Game Result.*Ai\(1\)-{re.escape(da)} has won", out))
        wb = len(re.findall(
            rf"Game Result.*Ai\(2\)-{re.escape(db)} has won", out))
        g = len(re.findall(r"Game Result", out))
        return da, db, wa, wb, g

    with ThreadPoolExecutor(max_workers=PARALLEL) as pool:
        for da, db, wa, wb, g in pool.map(one, list(enumerate(jobs))):
            tally[da][0] += wa
            tally[da][1] += g
            tally[db][0] += wb
            tally[db][1] += g
            done[0] += 1
            if done[0] % 25 == 0:
                el = time.time() - t0
                gp = sum(v[1] for v in tally.values()) // 2
                print(f"[rate] {done[0]}/{len(jobs)} matches, "
                      f"{gp} games ({el:.0f}s)", flush=True)

    ratings = sorted(
        ({"deck": dk, "wins": w, "games": g,
          "wr": round(w / g, 4) if g else 0.0}
         for dk, (w, g) in tally.items()),
        key=lambda r: -r["wr"])
    (OUT / "meta_strength.json").write_text(json.dumps(
        {"t": int(time.time()), "rounds": ROUNDS,
         "benchmark": meta["benchmark"], "ratings": ratings}, indent=1))
    print("[rate] top 10:", flush=True)
    for r in ratings[:10]:
        print(f"  {r['wr']:.0%} ({r['games']}g) {r['deck']}", flush=True)
    print("[rate] bottom 5:", flush=True)
    for r in ratings[-5:]:
        print(f"  {r['wr']:.0%} ({r['games']}g) {r['deck']}", flush=True)
    print(f"[rate] DONE -> meta_strength.json "
          f"({round(time.time() - t0)}s)", flush=True)
    for s in servers:
        s.shutdown()
        s.server_close()


if __name__ == "__main__":
    main()
