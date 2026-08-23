"""Collect protocol-v4 target-decision BC data.

Observe-only: both players bridged (no FORGE_EXT_PLAYER filter),
FORGE_EXT_TARGETS=1, every kind=="target" event appended to
output/target_events.jsonl with the builtin AI's choice in `proposed`
(the BC label). Mixed matchups so targets come from every deck's
abilities. Resume-safe: appends; game count tracked in the sidecar.

Usage:
  FORGE_SIM_SERVER=1 python experiments/collect_targets.py --games 400
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

from pilot_bridge import start_server, run_bridged  # noqa: E402

OUT = Path(__file__).resolve().parent / "output"
DECKS = ["fac_roaming", "fac_g0", "fac_g1", "fac_g2"]
EVENTS = OUT / "target_events.jsonl"
SIDECAR = OUT / "target_events_state.json"

lock = threading.Lock()
counts = {"events": 0}


def observing(state):
    if state.get("kind") == "target" and state.get("proposed"):
        with lock:
            counts["events"] += 1
            with open(EVENTS, "a", encoding="utf-8") as f:
                f.write(json.dumps(state, separators=(",", ":")) + "\n")
    return "ok"        # observe-only: builtin plays every seat normally


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=400)
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
            print(f"[collect] chunks {state['done']}/{total_chunks}, "
                  f"{counts['events']} target events this session",
                  flush=True)
    finally:
        srv.shutdown()
        srv.server_close()
    total = sum(1 for _ in open(EVENTS, "rb")) if EVENTS.exists() else 0
    print(f"[collect] done: {total} total target events in {EVENTS.name}",
          flush=True)


if __name__ == "__main__":
    main()
