"""Pilot factory: deck in -> validated deck-specific pilot out.

The conclusion the Roaming Encounters validation forced: the evolver's
fitness is only as good as the pilot flying the decks, and scripted
force rules don't generalize. This pipeline makes the pilot a
FUNCTION OF THE DECK:

  1. parse the deck, write it to Forge
  2. probe nearest corpus neighbors for a power-matched gauntlet
     (10-90% winrate band - the informative opponents)
  3. warm-start from the shared BC checkpoint (pilot2.pt) and
     self-play fine-tune on THIS deck vs that gauntlet
     (REINFORCE: win/loss + locked-card bonus; cast, attack and
     block decisions all in the update)
  4. VALIDATION GATE at real budgets: tuned pilot vs built-in AI vs
     BC-argmax, N games each against the same gauntlet. The tuned
     checkpoint is promoted only if it beats builtin - the race-vs-
     confirmation lesson applied to pilots.

Outputs: output/pilot-<name>.pt (checkpoint), output/factory-<name>.json
(journal: gauntlet, training curve, validation verdict).

Usage (demo scale here; scale --iters/--games on a GPU box):
  FORGE_SIM_SERVER=1 python3 experiments/pilot_factory.py deck.txt \
      --format standard --lock "Random Encounter" --name roaming \
      --iters 12 --games 16 --val-games 96
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pilot_bridge  # noqa: E402
from pilot_bridge import (ModelPolicy, start_server,  # noqa: E402
                          run_bridged)
from improve_deck import Improver, norm, write_dck, ci95  # noqa: E402
from self_play import (RecordingPolicy, reinforce_update,  # noqa: E402
                       game_lock_frac)
from mtg_deckbuilder.collection import parse_collection  # noqa: E402

OUT = Path(__file__).resolve().parent / "output"
RNG = np.random.default_rng(5)


def build_deck(imp, deck_text, name):
    deck = {}
    for n, q in parse_collection(deck_text).items():
        row = imp.name_to_row.get(norm(n))
        if row is None:
            raise ValueError(f"unknown card: {n}")
        if not imp.playable(row):
            raise ValueError(f"not Forge-playable: {n}")
        deck[row] = deck.get(row, 0) + q
    pairs = sorted((imp.meta[r]["name"], q) for r, q in deck.items())
    write_dck(name, pairs)
    return deck


def probe_gauntlet(imp, deck, me, size, probe_games, log):
    """Nearest playable corpus neighbors in the 10-90% band."""
    from improve_deck import run_match
    nl = [(r, q) for r, q in deck.items() if not imp.is_land[r]]
    v = sum(imp.vecs[r] * q for r, q in nl)
    v /= max(np.linalg.norm(v), 1e-9)
    order = np.argsort(-(imp.centroids @ v))
    chosen, probed = [], 0
    for idx in order:
        if len(chosen) >= size or probed >= 12:
            break
        d = imp.corpus[int(idx)]
        if not all(norm(n) in imp.supported for n, _q in d["main"]):
            continue
        probed += 1
        gname = f"fac_g{len(chosen)}"
        write_dck(gname, d["main"])
        w, n = run_match(me, gname, probe_games, 90 + 30 * probe_games)
        log(f"probe #{probed}: {w}/{n}")
        if n and 0.1 <= w / n <= 0.9:
            chosen.append(d["main"])
    for gi, main in enumerate(chosen):
        write_dck(f"fac_g{gi}", main)
    return [f"fac_g{gi}" for gi in range(len(chosen))]


def measure_arm(label, deck_name, gauntlet, games_per_g, port, log,
                parallel=1):
    """Winrate vs the gauntlet; games split across `parallel` warm JVMs.
    On a many-core box (--val-parallel 6) validation is minutes."""
    from concurrent.futures import ThreadPoolExecutor
    chunks = []
    for g in gauntlet:
        per = max(1, games_per_g // parallel)
        for wk in range(parallel):
            chunks.append((g, per, wk))

    def one(job):
        g, per, wk = job
        return run_bridged(deck_name, g, per, 90 + 40 * per, port,
                           player_filter=deck_name if port else None,
                           quiet=True, worker=wk)

    w = n = 0
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        for out in pool.map(one, chunks):
            w += len(re.findall(
                rf"Game Result.*Ai\(1\)-{re.escape(deck_name)} has won", out))
            n += len(re.findall(r"Game Result", out))
    p, half = ci95(w, n)
    log(f"[validate] {label}: {w}/{n} = {p:.0%} ±{half:.0%}")
    return {"label": label, "wins": w, "games": n,
            "winrate": round(p, 3), "ci": round(half, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("deck")
    ap.add_argument("--format", default="standard")
    ap.add_argument("--name", default="job")
    ap.add_argument("--lock", action="append", default=[])
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--games", type=int, default=16,
                    help="training games per iteration")
    ap.add_argument("--val-games", type=int, default=96,
                    help="validation games per arm")
    ap.add_argument("--val-parallel", type=int, default=1,
                    help="parallel warm JVMs for validation (use ~cores/2)")
    ap.add_argument("--gauntlet", type=int, default=3)
    ap.add_argument("--probe-games", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--lock-bonus", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=5e-5)
    args = ap.parse_args()

    import torch
    journal = {"name": args.name, "format": args.format,
               "locks": args.lock, "train": [], "validation": None}
    jpath = OUT / f"factory-{args.name}.json"

    def log(msg):
        print(msg, flush=True)

    def save():
        jpath.write_text(json.dumps(journal, indent=1))

    imp = Improver(args.format)
    me = f"fac_{args.name}"
    deck = build_deck(imp, Path(args.deck).read_text(encoding="utf-8"), me)
    log(f"deck ok: {sum(deck.values())} cards")

    gauntlet = probe_gauntlet(imp, deck, me, args.gauntlet,
                              args.probe_games, log)
    if not gauntlet:
        log("no informative gauntlet found - aborting")
        return
    journal["gauntlet_size"] = len(gauntlet)
    log(f"gauntlet: {len(gauntlet)} decks")

    # ---- self-play fine-tune from the BC warm start ---------------------
    # Guided exploration: the BC clone inherited the built-in AI's
    # blindness (it never casts an AI-invisible lock card, so sampling
    # never explores it and REINFORCE has nothing to reinforce). With
    # probability guide-eps a cast decision is answered by the lock
    # force rule instead of the model; the update then computes the
    # model's log-prob OF THAT FORCED ACTION, so rewarded lock casts
    # pull the model toward casting the lock itself. Anneals to 0.
    class GuidedPolicy(RecordingPolicy):
        def __init__(self, inner, locks, eps):
            super().__init__(inner)
            self.locks = set(locks)
            self.eps = eps

        def __call__(self, state):
            if (state.get("kind") == "cast"
                    and float(RNG.random()) < self.eps):
                for cand in state.get("candidates", []):
                    if cand["card"] in self.locks \
                            and cand["zone"] in ("hand", "graveyard"):
                        # graveyard = the flashback line, when payable
                        reply = f"force\t{cand['i']}"
                        self.buffer.append((state, reply))
                        return reply
            return super().__call__(state)

    policy = GuidedPolicy(ModelPolicy(), args.lock, eps=0.6)
    policy.inner.temperature = args.temperature
    pilot_bridge.tainted_policy = policy
    srv = start_server(0)
    port = srv.server_address[1]
    opt = torch.optim.Adam(policy.inner.model.parameters(), lr=args.lr)
    ckpt = OUT / f"pilot-{args.name}.pt"
    baseline = 0.0
    try:
        for it in range(args.iters):
            batch, wins, lock_sum = [], 0, 0.0
            for g in range(args.games):
                policy.buffer = []
                opp = gauntlet[int(RNG.integers(len(gauntlet)))]
                out = run_bridged(me, opp, 1, 240, port,
                                  player_filter=me, quiet=True)
                won = bool(re.search(
                    rf"Game Result.*Ai\(1\)-{re.escape(me)} has won", out))
                lf = game_lock_frac(policy.buffer, set(args.lock))
                batch.append((list(policy.buffer),
                              (1.0 if won else -1.0)
                              + args.lock_bonus * lf))
                wins += won
                lock_sum += lf
            loss = reinforce_update(policy.inner, torch, batch,
                                    baseline, opt)
            rewards = [r for _d, r in batch]
            baseline = 0.7 * baseline + 0.3 * float(np.mean(rewards))
            policy.eps = max(0.0, 0.6 * (1 - (it + 1) / args.iters))
            entry = {"iter": it, "winrate": round(wins / args.games, 3),
                     "lock_frac": round(lock_sum / args.games, 3),
                     "eps": round(policy.eps, 2), "loss": round(loss, 4)}
            journal["train"].append(entry)
            log(f"[train] {json.dumps(entry)}")
            torch.save(policy.inner.model.state_dict(), ckpt)
            save()

        # ---- validation gate at real budgets ----------------------------
        per_g = max(4, args.val_games // len(gauntlet))
        arms = []
        pilot_bridge.tainted_policy = lambda s: "ok"      # observer noop
        arms.append(measure_arm("builtin", me, gauntlet, per_g, None, log,
                                parallel=args.val_parallel))
        bc = ModelPolicy()                                # argmax, no temp
        pilot_bridge.tainted_policy = bc
        arms.append(measure_arm("bc", me, gauntlet, per_g, port, log,
                                parallel=args.val_parallel))
        tuned = ModelPolicy(ckpt=ckpt)
        pilot_bridge.tainted_policy = tuned
        arms.append(measure_arm("tuned", me, gauntlet, per_g, port, log,
                                parallel=args.val_parallel))
        by = {a["label"]: a for a in arms}
        promoted = by["tuned"]["winrate"] > by["builtin"]["winrate"]
        journal["validation"] = {"arms": arms, "promoted": bool(promoted)}
        save()
        log(f"[verdict] tuned {by['tuned']['winrate']:.0%} vs builtin "
            f"{by['builtin']['winrate']:.0%} vs bc {by['bc']['winrate']:.0%}"
            f" -> {'PROMOTED' if promoted else 'NOT promoted'}")
    finally:
        srv.shutdown()
        srv.server_close()
        save()


if __name__ == "__main__":
    main()
