"""One resume-safe DAgger round: fly -> retrain -> re-validate.

dagger.py runs whole rounds in one process; on a container that
recycles ~hourly a round can die three times before finishing. This
driver banks progress after every pair flown, resumes retraining
mid-epoch via train_pilot2's sidecar, and reuses dagger.py's
sub-arm-banked validation. Re-run the same command after any
interruption and it continues where it stopped.

The builtin validation arm doesn't depend on the model, so it is
carried forward from the previous round; only the clone arm is
re-measured. The previous round's full verdict is archived under
journal["history"] before the clone entry is cleared.

Usage:
  FORGE_SIM_SERVER=1 python3 experiments/dagger_round.py --round 2 \
      --games 200 --val-games 96
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pilot_bridge  # noqa: E402
from pilot_bridge import (ModelPolicy, start_server,  # noqa: E402
                          run_bridged, COLLECT_FILE)
from dagger import PAIRS, validate  # noqa: E402

OUT = Path(__file__).resolve().parent / "output"


def log(msg):
    print(msg, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--games", type=int, default=200,
                    help="clone-flown games this round")
    ap.add_argument("--val-games", type=int, default=96)
    ap.add_argument("--temperature", type=float, default=0.25)
    args = ap.parse_args()

    spath = OUT / f"dagger_round{args.round}_state.json"
    state = (json.loads(spath.read_text()) if spath.exists()
             else {"pairs_done": 0, "games_flown": 0, "retrain": None})

    def save():
        spath.write_text(json.dumps(state, indent=1))

    jpath = OUT / "dagger.json"
    journal = json.loads(jpath.read_text())

    # ---- stage 1: fly the current clone, teacher labels logged ----------
    per = max(1, args.games // len(PAIRS))
    if state["pairs_done"] < len(PAIRS):
        policy = ModelPolicy()                    # current pilot2.pt
        policy.temperature = args.temperature
        pilot_bridge.tainted_policy = policy
        COLLECT_FILE["fh"] = open(OUT / "pilot_dataset.jsonl", "a")
        srv = start_server(0)
        port = srv.server_address[1]
        try:
            CH = 5   # sub-chunk: slow matchups can outlive a recycle window
            for pi, (a, b) in enumerate(PAIRS):
                if pi < state["pairs_done"]:
                    continue
                nch = (per + CH - 1) // CH
                for ci in range(state.get("chunks_done", 0), nch):
                    n = min(CH, per - ci * CH)
                    out = run_bridged(a, b, n, 90 + 40 * n, port,
                                      player_filter=a, quiet=True)
                    state["games_flown"] += len(
                        re.findall(r"Game Result", out))
                    state["chunks_done"] = ci + 1
                    save()
                state["pairs_done"] = pi + 1
                state["chunks_done"] = 0
                save()
                log(f"[fly] {a} vs {b}: done "
                    f"({state['games_flown']} total)")
        finally:
            COLLECT_FILE["fh"].close()
            COLLECT_FILE["fh"] = None
            srv.shutdown()
            srv.server_close()

    # ---- stage 2: retrain on the aggregate ------------------------------
    if state["retrain"] != "done":
        sidecar = OUT / "pilot2_train_state.json"
        if state["retrain"] is None:
            # the sidecar still says the PREVIOUS round's training is
            # complete; clear it exactly once so this retrain runs,
            # while an interrupted retrain still resumes mid-epoch
            sidecar.unlink(missing_ok=True)
            state["retrain"] = "started"
            save()
        env = dict(os.environ)
        env.update({"TP2_ACT_CAP": "12000", "TP2_PASS_RATIO": "3",
                    "TP2_EPOCHS": "3", "TP2_RESUME": "1"})
        log("[retrain] training on aggregate dataset...")
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve().parent
                                 / "train_pilot2.py")],
            capture_output=True, text=True, env=env)
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-3:]
        for line in tail:
            log(f"  {line}")
        if proc.returncode != 0:
            log("[retrain] FAILED - rerun this command to resume")
            sys.exit(1)
        state["retrain"] = "done"
        state["retrain_final"] = tail
        save()
        journal["rounds"].append(
            {"round": args.round, "games_flown": state["games_flown"],
             "final": tail[-2] if len(tail) >= 2 else ""})
        jpath.write_text(json.dumps(journal, indent=1))

    # ---- stage 3: re-validate the clone arm -----------------------------
    journal = json.loads(jpath.read_text())
    val = journal.get("validation") or {}
    already = any(h.get("round") == args.round - 1
                  for h in journal.get("history", []))
    if "clone" in val and not already:
        journal.setdefault("history", []).append(
            {"round": args.round - 1, **val})
        val.pop("clone", None)
        journal["validation"] = val
        jpath.write_text(json.dumps(journal, indent=1))

    def on_arm(arms):
        journal["validation"] = arms
        jpath.write_text(json.dumps(journal, indent=1))

    journal["validation"] = validate(args.val_games, log,
                                     existing=journal.get("validation"),
                                     on_arm=on_arm)
    jpath.write_text(json.dumps(journal, indent=1))
    b = journal["validation"]["builtin"]
    c = journal["validation"]["clone"]
    log(f"[verdict r{args.round}] clone {c['winrate']:.0%} "
        f"±{c['ci']:.0%} vs builtin {b['winrate']:.0%} ±{b['ci']:.0%}")


if __name__ == "__main__":
    main()
