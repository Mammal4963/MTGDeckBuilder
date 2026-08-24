"""Round-3 BC collection: mulligan keep/mull decisions (+ London tucks
and target events) from builtin-piloted games, both players observed.

Events append to output/round3_events.jsonl with the builtin's choice
in `proposed`. Resume-safe via sidecar.

Usage:
  FORGE_SIM_SERVER=1 python experiments/collect_round3.py --games 300
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ["FORGE_EXT_TARGETS"] = "1"
os.environ["FORGE_EXT_MULL"] = "1"

from pilot_bridge import start_server, run_bridged  # noqa: E402

OUT = Path(__file__).resolve().parent / "output"
DECKS = ["fac_roaming", "fac_g0", "fac_g1", "fac_g2"]
EVENTS = OUT / "round3_events.jsonl"
SIDECAR = OUT / "round3_events_state.json"

lock = threading.Lock()
counts = {"mulligan": 0, "mulligan_tuck": 0, "target": 0}


def observing(state):
    k = state.get("kind")
    if k in counts and state.get("proposed") is not None:
        with lock:
            counts[k] += 1
            with open(EVENTS, "a", encoding="utf-8") as f:
                f.write(json.dumps(state, separators=(",", ":")) + "\n")
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--parallel", type=int, default=8)
    args = ap.parse_args()

    state = (json.loads(SIDECAR.read_text()) if SIDECAR.exists()
             else {"done": 0})
    matchups = list(combinations(DECKS, 2))
    per_chunk = 4
    total_chunks = args.games // per_chunk
    srv = start_server(0, policy=observing)
    port = srv.server_address[1]

    def one(ci):
        a, b = matchups[ci % len(matchups)]
        return run_bridged(a, b, per_chunk, 90 + 40 * per_chunk, port,
                           player_filter="", quiet=True,
                           worker=ci % args.parallel)

    try:
        todo = list(range(state["done"], total_chunks))
        for w0 in range(0, len(todo), args.parallel):
            wave = todo[w0:w0 + args.parallel]
            with ThreadPoolExecutor(max_workers=args.parallel) as pool:
                list(pool.map(one, wave))
            state["done"] = wave[-1] + 1
            SIDECAR.write_text(json.dumps(state))
            print(f"[collect3] chunks {state['done']}/{total_chunks} "
                  f"{json.dumps(counts)}", flush=True)
    finally:
        srv.shutdown()
        srv.server_close()


if __name__ == "__main__":
    main()
