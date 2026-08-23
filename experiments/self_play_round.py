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
    """eps-forced lock casts so sampling explores what BC never saw.

    Each instance carries its own RNG so parallel collection workers
    stay independent (np.random.Generator is not thread-safe)."""

    def __init__(self, inner, locks, eps, rng=None):
        super().__init__(inner)
        self.locks = set(locks)
        self.eps = eps
        self.rng = rng if rng is not None else RNG

    def __call__(self, state):
        self._stamp(state)
        if state.get("kind") == "cast" and float(self.rng.random()) < self.eps:
            for cand in state.get("candidates", []):
                if cand["card"] in self.locks \
                        and cand["zone"] in ("hand", "graveyard"):
                    state["_eps_forced"] = True   # inspector: exploration,
                    reply = f"force\t{cand['i']}"  # not the model's choice
                    self.buffer.append((state, reply))
                    return reply
        return super().__call__(state, _stamped=True)


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
    ap.add_argument("--eps-floor", type=float, default=0.1,
                    help="exploration never anneals below this (the "
                    "container run annealed to 0 and the lock habit died)")
    ap.add_argument("--parallel", type=int, default=1,
                    help="warm JVM workers for collection + validation")
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
    # One shared model; per-worker GuidedPolicy wrappers (own buffer,
    # own RNG, own policy server + warm JVM) so games fly in parallel.
    # reinforce_update runs single-threaded between iterations.
    from concurrent.futures import ThreadPoolExecutor
    inner = ModelPolicy()
    inner.temperature = args.temperature
    nwk = max(1, args.parallel)
    workers = [GuidedPolicy(inner, locks, eps=args.eps0,
                            rng=np.random.default_rng(1000 + wk))
               for wk in range(nwk)]
    opt = torch.optim.Adam(inner.model.parameters(), lr=args.lr)
    it0, baseline = 0, 0.0
    if rl_state.exists():
        bundle = torch.load(rl_state, weights_only=False)
        inner.model.load_state_dict(bundle["model"])
        opt.load_state_dict(bundle["opt"])
        it0, baseline = bundle["iter"], bundle["baseline"]
        log(f"[resume] iteration {it0}, baseline {baseline:.3f}")

    if it0 < args.iters:
        servers = [start_server(0, policy=w) for w in workers]
        ports = [s.server_address[1] for s in servers]

        import gzip
        import threading
        import time
        games_dir = OUT / "games"
        games_dir.mkdir(exist_ok=True)
        index_lock = threading.Lock()

        def archive_game(it, wk, seq, opp, won, lf, decisions):
            """Full decision trajectory per game, for the inspector."""
            name = f"it{it:03d}_w{wk}_{seq:02d}.json.gz"
            dur_s = round(sum(s.get("_dt_ms", 0)
                              for s, _r in decisions) / 1000, 1)
            rec = {"iter": it, "worker": wk, "seq": seq, "opp": opp,
                   "won": won, "lock_frac": lf, "deck": DECK,
                   "dur_s": dur_s, "decisions": decisions}
            with gzip.open(games_dir / name, "wt", encoding="utf-8") as f:
                json.dump(rec, f, separators=(",", ":"))
            line = json.dumps({"iter": it, "worker": wk, "opp": opp,
                               "won": won, "lock_frac": lf, "dur_s": dur_s,
                               "n_dec": len(decisions), "t": int(time.time()),
                               "file": name})
            with index_lock:
                with open(OUT / "games_index.jsonl", "a",
                          encoding="utf-8") as f:
                    f.write(line + "\n")

        def worker_games(it, wk, n):
            p, results = workers[wk], []
            for g in range(n):
                p.buffer = []
                opp = GAUNTLET[int(p.rng.integers(len(GAUNTLET)))]
                out = run_bridged(DECK, opp, 1, 240, ports[wk],
                                  player_filter=DECK, quiet=True, worker=wk)
                won = bool(re.search(
                    rf"Game Result.*Ai\(1\)-{DECK} has won", out))
                lf = game_lock_frac(p.buffer, set(locks))
                decisions = list(p.buffer)
                try:
                    archive_game(it, wk, g, opp, won, lf, decisions)
                except OSError:
                    pass          # archiving must never stall training
                results.append((decisions, won, lf))
            return results

        try:
            counts = [args.games // nwk + (1 if wk < args.games % nwk else 0)
                      for wk in range(nwk)]
            for it in range(it0, args.iters):
                eps = max(args.eps_floor,
                          args.eps0 * (1 - it / args.iters))
                for w in workers:
                    w.eps = eps
                with ThreadPoolExecutor(max_workers=nwk) as pool:
                    futs = [pool.submit(worker_games, it, wk, counts[wk])
                            for wk in range(nwk)]
                    flown = [r for f in futs for r in f.result()]
                batch = [(dec, (1.0 if won else -1.0)
                          + args.lock_bonus * lf)
                         for dec, won, lf in flown]
                wins = sum(won for _d, won, _lf in flown)
                lock_sum = sum(lf for _d, _w, lf in flown)
                loss = reinforce_update(inner, torch, batch,
                                        baseline, opt)
                rewards = [r for _d, r in batch]
                baseline = 0.7 * baseline + 0.3 * float(np.mean(rewards))
                n = len(flown)
                entry = {"iter": it, "winrate": round(wins / n, 3),
                         "lock_frac": round(lock_sum / n, 3),
                         "eps": round(eps, 2), "loss": round(loss, 4),
                         "games": n, "t": int(__import__("time").time())}
                journal["train"].append(entry)
                log(f"[train] {json.dumps(entry)}")
                torch.save(inner.model.state_dict(), rl_ckpt)
                torch.save({"model": inner.model.state_dict(),
                            "opt": opt.state_dict(), "iter": it + 1,
                            "baseline": baseline}, rl_state)
                save()
        finally:
            for s in servers:
                s.shutdown()
                s.server_close()
            # release the collection JVMs before validation spawns its own
            import sim_server
            with sim_server._SHARED_LOCK:
                for c in sim_server._SHARED.values():
                    c.close()
                sim_server._SHARED.clear()

    # ---- stage 2: validate the tuned checkpoint, banked + idempotent ----
    val = journal["validation"]
    if "rl" not in val:
        per = max(4, args.val_games // len(GAUNTLET))
        chunks_per_g = 4
        per_chunk = max(2, per // chunks_per_g)
        jobs = [g for g in GAUNTLET for _c in range(chunks_per_g)]
        part = val.get("_rl_partial", {"wins": 0, "games": 0, "jobs": 0})
        if part["jobs"] < len(jobs):
            from concurrent.futures import ThreadPoolExecutor
            nwk = max(1, args.parallel)
            vpol = ModelPolicy(ckpt=rl_ckpt)
            srv = start_server(0, policy=vpol)
            port = srv.server_address[1]

            def one_chunk(job):
                ji, g = job
                return run_bridged(DECK, g, per_chunk,
                                   90 + 40 * per_chunk, port,
                                   player_filter=DECK, quiet=True,
                                   worker=ji % nwk)
            try:
                todo = [(ji, g) for ji, g in enumerate(jobs)
                        if ji >= part["jobs"]]
                # waves of nwk chunks: parallel inside a wave, banked
                # after each wave so a kill loses at most one wave
                for w0 in range(0, len(todo), nwk):
                    wave = todo[w0:w0 + nwk]
                    with ThreadPoolExecutor(max_workers=nwk) as pool:
                        outs = list(pool.map(one_chunk, wave))
                    for out in outs:
                        part["wins"] += len(re.findall(
                            rf"Game Result.*Ai\(1\)-{DECK} has won", out))
                        part["games"] += len(re.findall(r"Game Result", out))
                    part["jobs"] = wave[-1][0] + 1
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
