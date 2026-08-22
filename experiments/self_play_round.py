"""Resume-safe self-play (REINFORCE) round from the parity warm start.

Imitation is saturated: two independent 96-game samples read parity
with the built-in AI (dagger.json history). Going past the teacher
needs reward-driven learning, and this driver runs one such round in
the dagger_round.py stage-banked style so ~hourly container recycles
cost minutes, not stages:

  1. TRAIN: REINFORCE fine-tune from pilot2.pt (never modified) into
     pilot2_rl.pt. One game per sim job for exact trajectory<->reward
     pairing; reward = win/loss + lock-bonus. Guided exploration
     eps-forces locked-card casts (the parity clone inherited the
     teacher's AI-blindness to them), annealing to 0. Banked per
     iteration: model + optimizer + EMA baseline in pilot2_rl_state.pt.
  2. VALIDATE: the tuned checkpoint flies 96 games vs the same
     gauntlet in banked sub-chunks; promotion from banked chunks is
     part of the resume path (both DAgger rounds lost their final
     promotion step to a recycle and needed hand-promotion).

Verdict compares against dagger.json's builtin and clone arms.

Usage:
  FORGE_SIM_SERVER=1 python3 experiments/self_play_round.py \
      --iters 14 --games 16 --val-games 96
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pilot_bridge  # noqa: E402
from pilot_bridge import (ModelPolicy, start_server,  # noqa: E402
                          run_bridged)
from improve_deck import ci95  # noqa: E402
from self_play import (RecordingPolicy, reinforce_update,  # noqa: E402
                       game_lock_frac)

OUT = Path(__file__).resolve().parent / "output"
RNG = np.random.default_rng(11)

DECK = "fac_roaming"
GAUNTLET = ["fac_g0", "fac_g1", "fac_g2"]


class GuidedPolicy(RecordingPolicy):
    """eps-forced lock casts so sampling explores what BC never saw."""

    def __init__(self, inner, locks, eps):
        super().__init__(inner)
        self.locks = set(locks)
        self.eps = eps

    def __call__(self, state):
        if state.get("kind") == "cast" and float(RNG.random()) < self.eps:
            for cand in state.get("candidates", []):
                if cand["card"] in self.locks \
                        and cand["zone"] in ("hand", "graveyard"):
                    reply = f"force\t{cand['i']}"
                    self.buffer.append((state, reply))
                    return reply
        return super().__call__(state)


def log(msg):
    print(msg, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=14)
    ap.add_argument("--games", type=int, default=16)
    ap.add_argument("--val-games", type=int, default=96)
    ap.add_argument("--lock", action="append",
                    default=None, help="default: Random Encounter")
    ap.add_argument("--lock-bonus", type=float, default=0.3)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--eps0", type=float, default=0.5)
    args = ap.parse_args()
    locks = args.lock if args.lock else ["Random Encounter"]

    import torch
    jpath = OUT / "selfplay_round.json"
    journal = (json.loads(jpath.read_text()) if jpath.exists()
               else {"train": [], "validation": {}})

    def save():
        jpath.write_text(json.dumps(journal, indent=1))

    rl_ckpt = OUT / "pilot2_rl.pt"            # plain state-dict (serving)
    rl_state = OUT / "pilot2_rl_state.pt"     # bundle (exact resume)

    # ---- stage 1: REINFORCE fine-tune, banked per iteration -------------
    policy = GuidedPolicy(ModelPolicy(), locks, eps=args.eps0)
    policy.inner.temperature = args.temperature
    opt = torch.optim.Adam(policy.inner.model.parameters(), lr=args.lr)
    it0, baseline = 0, 0.0
    if rl_state.exists():
        bundle = torch.load(rl_state, weights_only=False)
        policy.inner.model.load_state_dict(bundle["model"])
        opt.load_state_dict(bundle["opt"])
        it0, baseline = bundle["iter"], bundle["baseline"]
        log(f"[resume] iteration {it0}, baseline {baseline:.3f}")

    if it0 < args.iters:
        pilot_bridge.tainted_policy = policy
        srv = start_server(0)
        port = srv.server_address[1]
        try:
            for it in range(it0, args.iters):
                policy.eps = args.eps0 * (1 - it / args.iters)
                batch, wins, lock_sum = [], 0, 0.0
                for _g in range(args.games):
                    policy.buffer = []
                    opp = GAUNTLET[int(RNG.integers(len(GAUNTLET)))]
                    out = run_bridged(DECK, opp, 1, 240, port,
                                      player_filter=DECK, quiet=True)
                    won = bool(re.search(
                        rf"Game Result.*Ai\(1\)-{DECK} has won", out))
                    lf = game_lock_frac(policy.buffer, set(locks))
                    batch.append((list(policy.buffer),
                                  (1.0 if won else -1.0)
                                  + args.lock_bonus * lf))
                    wins += won
                    lock_sum += lf
                loss = reinforce_update(policy.inner, torch, batch,
                                        baseline, opt)
                rewards = [r for _d, r in batch]
                baseline = 0.7 * baseline + 0.3 * float(np.mean(rewards))
                entry = {"iter": it, "winrate": round(wins / args.games, 3),
                         "lock_frac": round(lock_sum / args.games, 3),
                         "eps": round(policy.eps, 2), "loss": round(loss, 4)}
                journal["train"].append(entry)
                log(f"[train] {json.dumps(entry)}")
                torch.save(policy.inner.model.state_dict(), rl_ckpt)
                torch.save({"model": policy.inner.model.state_dict(),
                            "opt": opt.state_dict(), "iter": it + 1,
                            "baseline": baseline}, rl_state)
                save()
        finally:
            srv.shutdown()
            srv.server_close()

    # ---- stage 2: validate the tuned checkpoint, banked + idempotent ----
    val = journal["validation"]
    if "rl" not in val:
        per = max(4, args.val_games // len(GAUNTLET))
        chunks_per_g = 4
        per_chunk = max(2, per // chunks_per_g)
        jobs = [g for g in GAUNTLET for _c in range(chunks_per_g)]
        part = val.get("_rl_partial", {"wins": 0, "games": 0, "jobs": 0})
        if part["jobs"] < len(jobs):
            pilot_bridge.tainted_policy = ModelPolicy(ckpt=rl_ckpt)
            srv = start_server(0)
            port = srv.server_address[1]
            try:
                for ji, g in enumerate(jobs):
                    if ji < part["jobs"]:
                        continue
                    out = run_bridged(DECK, g, per_chunk,
                                      90 + 40 * per_chunk, port,
                                      player_filter=DECK, quiet=True)
                    part["wins"] += len(re.findall(
                        rf"Game Result.*Ai\(1\)-{DECK} has won", out))
                    part["games"] += len(re.findall(r"Game Result", out))
                    part["jobs"] = ji + 1
                    val["_rl_partial"] = part
                    save()
            finally:
                srv.shutdown()
                srv.server_close()
        # promotion IS the resume path: runs even if the process died
        # right after the last chunk banked
        w, n = part["wins"], part["games"]
        p, half = ci95(w, n)
        val["rl"] = {"wins": w, "games": n,
                     "winrate": round(p, 3), "ci": round(half, 3)}
        val.pop("_rl_partial", None)
        save()

    r = val["rl"]
    d = json.loads((OUT / "dagger.json").read_text())
    b, c = d["validation"]["builtin"], d["validation"]["clone"]
    log(f"[verdict] rl {r['winrate']:.0%} ±{r['ci']:.0%}"
        f" vs clone {c['winrate']:.0%} vs builtin {b['winrate']:.0%}")


if __name__ == "__main__":
    main()
