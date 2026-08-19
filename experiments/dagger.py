"""DAgger training for the pilot: fix behavior cloning's compounding error.

The factory verdict measured the classic BC failure: a clone that
agrees with its teacher ~70% per decision plays FAR worse than the
teacher (6% vs 26% winrate), because each divergence drifts the game
into states the clone never trained on.

DAgger's fix: collect the TEACHER's answer on the states the CLONE
actually visits, then retrain on the aggregate. Our bridge makes the
labels free - every decision message already carries `proposed` (the
built-in AI's choice at the current state, computed before the policy
overrides) and combat messages carry the AI's chosen attack/block sets
(serialized after super() fills them). So: fly the clone with
collection on, and the logged states ARE teacher-labeled clone-visited
states. train_pilot2's label extraction uses exactly those fields.

Each round: fly the current clone for N games (slight temperature for
trajectory diversity) -> append to the dataset -> retrain -> the next
round flies the improved clone. Finish with a validation A/B vs the
built-in AI at real budgets.

Usage (container demo scale; scale rounds/games on a GPU box):
  FORGE_SIM_SERVER=1 python3 experiments/dagger.py \
      --rounds 2 --games 100 --val-games 96
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pilot_bridge  # noqa: E402
from pilot_bridge import (ModelPolicy, start_server,  # noqa: E402
                          run_bridged, COLLECT_FILE)
from improve_deck import ci95  # noqa: E402

OUT = Path(__file__).resolve().parent / "output"
RNG = np.random.default_rng(9)

# diverse matchups: the clone flies deck A of each pair
PAIRS = [("fac_roaming", "fac_g0"), ("fac_roaming", "fac_g1"),
         ("fac_roaming", "fac_g2"),
         ("evo_tainted2_base", "evo_tainted2_g1"),
         ("gauntlet_1", "gauntlet_2"), ("burn", "gauntlet_0"),
         ("evo_tainted2_g0", "evo_tainted2_g2"),
         ("gauntlet_2", "burn")]


def fly_and_collect(games: int, temperature: float, log) -> int:
    """Clone pilots deck A of rotating pairs; teacher labels logged."""
    policy = ModelPolicy()                    # current pilot2.pt
    policy.temperature = temperature
    pilot_bridge.tainted_policy = policy
    COLLECT_FILE["fh"] = open(OUT / "pilot_dataset.jsonl", "a")
    srv = start_server(0)
    port = srv.server_address[1]
    n0_lines = 0
    try:
        per = max(1, games // len(PAIRS))
        for a, b in PAIRS:
            out = run_bridged(a, b, per, 90 + 40 * per, port,
                              player_filter=a, quiet=True)
            done = len(re.findall(r"Game Result", out))
            n0_lines += done
            log(f"  {a} vs {b}: {done} games flown")
    finally:
        COLLECT_FILE["fh"].close()
        COLLECT_FILE["fh"] = None
        srv.shutdown()
        srv.server_close()
    return n0_lines


def retrain(log) -> str:
    env = dict(os.environ)
    env.update({"TP2_ACT_CAP": "12000", "TP2_PASS_RATIO": "3",
                "TP2_EPOCHS": "3"})
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parent
                             / "train_pilot2.py")],
        capture_output=True, text=True, env=env)
    tail = (proc.stdout + proc.stderr).strip().splitlines()[-3:]
    for line in tail:
        log(f"  {line}")
    return tail[-2] if len(tail) >= 2 else ""


def validate(val_games: int, log, existing: dict | None = None,
             on_arm=None) -> dict:
    gauntlet = ["fac_g0", "fac_g1", "fac_g2"]
    per = max(4, val_games // len(gauntlet))
    arms = dict(existing or {})
    for label, use_policy in (("builtin", False), ("clone", True)):
        if label in arms:
            log(f"[validate] {label}: already measured - skip")
            continue
        if use_policy:
            pilot_bridge.tainted_policy = ModelPolicy()   # fresh reload
            srv = start_server(0)
            port = srv.server_address[1]
        else:
            srv, port = None, None
        # sub-arm chunks: a whole arm can outlive a container recycle
        # window, so bank progress after every ~8-game job and resume
        pkey = f"_{label}_partial"
        part = arms.get(pkey, {"wins": 0, "games": 0, "jobs": 0})
        chunks_per_g = 4
        per_chunk = max(2, per // chunks_per_g)
        jobs = [g for g in gauntlet for _c in range(chunks_per_g)]
        try:
            for ji, g in enumerate(jobs):
                if ji < part["jobs"]:
                    continue
                out = run_bridged("fac_roaming", g, per_chunk,
                                  90 + 40 * per_chunk, port,
                                  player_filter="fac_roaming", quiet=True)
                part["wins"] += len(re.findall(
                    r"Game Result.*Ai\(1\)-fac_roaming has won", out))
                part["games"] += len(re.findall(r"Game Result", out))
                part["jobs"] = ji + 1
                arms[pkey] = part
                if on_arm:
                    on_arm(arms)
        finally:
            if srv:
                srv.shutdown()
                srv.server_close()
        w, n = part["wins"], part["games"]
        p, half = ci95(w, n)
        arms[label] = {"wins": w, "games": n,
                       "winrate": round(p, 3), "ci": round(half, 3)}
        arms.pop(pkey, None)
        log(f"[validate] {label}: {w}/{n} = {p:.0%} ±{half:.0%}")
        if on_arm:
            on_arm(arms)                 # journal each arm immediately
    return arms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--games", type=int, default=100,
                    help="clone-flown games per round")
    ap.add_argument("--val-games", type=int, default=96)
    ap.add_argument("--temperature", type=float, default=0.25)
    args = ap.parse_args()

    jpath = OUT / "dagger.json"
    journal = (json.loads(jpath.read_text()) if jpath.exists()
               else {"rounds": [], "validation": None})

    def log(msg):
        print(msg, flush=True)

    def save():
        jpath.write_text(json.dumps(journal, indent=1))

    for r in range(args.rounds):
        log(f"=== DAgger round {r + 1}/{args.rounds}: flying the clone ===")
        flown = fly_and_collect(args.games, args.temperature, log)
        log(f"=== retraining on aggregate dataset ===")
        final_line = retrain(log)
        journal["rounds"].append({"round": r + 1, "games_flown": flown,
                                  "final": final_line})
        save()

    log("=== validation: clone vs builtin, same deck, same gauntlet ===")

    def on_arm(arms):
        journal["validation"] = arms
        save()

    journal["validation"] = validate(args.val_games, log,
                                     existing=journal.get("validation"),
                                     on_arm=on_arm)
    save()
    b, c = journal["validation"]["builtin"], journal["validation"]["clone"]
    log(f"[verdict] clone {c['winrate']:.0%} vs builtin {b['winrate']:.0%}"
        f" (was 6% vs 26% pre-DAgger)")


if __name__ == "__main__":
    main()
