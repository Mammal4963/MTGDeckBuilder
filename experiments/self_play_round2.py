"""Round 2 self-play: ONE pilot, BOTH seats, targeting head live.

Differences from self_play_round.py (round 1, parity-confirmed):
  - Both players route through the bridge (no FORGE_EXT_PLAYER filter):
    the same network plays our deck AND the gauntlet deck each game -
    playing against itself without mirror-deck degeneracy. Two
    trajectories per game, rewarded +1/-1 from each seat's own
    game_end record.
  - FORGE_EXT_TARGETS=1: single-target choices flow through the
    tgt head (warm-started by train_targets.py) and train via
    REINFORCE alongside cast/combat.
  - eps lock-forcing applies only to the fac_roaming seat.

Validation stays the sacred yardstick: fac_roaming seat only, builtin
opponent, greedy, vs the same gauntlet.

Usage:
  FORGE_SIM_SERVER=1 python experiments/self_play_round2.py \
      --iters 60 --games 64 --val-games 96 --parallel 8 \
      --ckpt output/pilot2_v4.pt
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import os
os.environ["FORGE_EXT_TARGETS"] = "1"

import pilot_bridge  # noqa: E402
from pilot_bridge import ModelPolicy, start_server, run_bridged  # noqa: E402
from improve_deck import ci95  # noqa: E402
from self_play import (RecordingPolicy, reinforce_update,  # noqa: E402
                       ppo_update, ppo_update_gpu, game_lock_frac)

OUT = Path(__file__).resolve().parent / "output"
DECK = "fac_roaming"
GAUNTLET = ["fac_g0", "fac_g1", "fac_g2"]


class SeatAwarePolicy(RecordingPolicy):
    """Records both seats; eps-forces locked casts only for our seat."""

    def __init__(self, inner, locks, eps, rng):
        super().__init__(inner)
        self.locks = set(locks)
        self.eps = eps
        self.rng = rng

    def __call__(self, state, _stamped=False):
        self._stamp(state)
        if state.get("kind") == "cast" \
                and DECK in state.get("player", "") \
                and float(self.rng.random()) < self.eps:
            for cand in state.get("candidates", []):
                if cand["card"] in self.locks \
                        and cand["zone"] in ("hand", "graveyard"):
                    state["_eps_forced"] = True
                    reply = f"force\t{cand['i']}"
                    self.buffer.append((state, reply))
                    return reply
        return super().__call__(state, _stamped=True)


def chosen_card(state, reply):
    if state.get("kind") != "cast":
        return None
    if reply == "ok":
        p = state.get("proposed", [])
        return p[0] if p else None
    if reply.startswith("force\t"):
        idx = int(reply.split("\t")[1])
        return next((c["card"] for c in state.get("candidates", [])
                     if c["i"] == idx), None)
    return None


def shaped_bonus(dec, locks, feat, ramp_w, lock_w):
    """Small shaped rewards, our seat only. Win/loss stays dominant:
    ramp = ramp_w * min(1, lands in play by turn 5 / 5)
    lock = lock_w * (16 - first lock cast turn)/12, clamped to [0, lock_w]
    """
    from train_pilot2 import norm
    lands5, first_lock = 0, None
    for s, r in dec:
        if s.get("kind") == "game_end":
            continue
        t = s.get("turn", 0)
        if t <= 5:
            n = sum(1 for c in s.get("my_battlefield", [])
                    if norm(c["n"] if isinstance(c, dict) else c)
                    in feat.lands)
            lands5 = max(lands5, n)
        if first_lock is None and chosen_card(s, r) in locks:
            first_lock = t
    ramp = ramp_w * min(1.0, lands5 / 5.0)
    lock = (lock_w * max(0.0, min(1.0, (16 - first_lock) / 12.0))
            if first_lock is not None else 0.0)
    return ramp + lock, first_lock


def split_seats(buffer):
    """-> {player_name: (decisions, won)} using each seat's game_end.

    Forge fires the game-finished event before the WINNER's view marks
    the opponent as lost (opp_lost reads False for the winner), but the
    LOSER's own i_lost is already set - so the winner is the seat that
    hasn't lost while some other seat has. Life <= 0 is the fallback
    when only one seat's record survived."""
    seats = {}
    for state, reply in buffer:
        seats.setdefault(state.get("player", "?"), []).append((state, reply))
    ends = {}
    for name, dec in seats.items():
        ge = next((s for s, _r in reversed(dec)
                   if s.get("kind") == "game_end"), None)
        if ge is not None:
            ends[name] = ge
    out = {}
    for name, dec in seats.items():
        ge = ends.get(name)
        if ge is None:
            continue        # no terminator: skip seat (timeout/crash)
        others_lost = any(bool(g.get("i_lost"))
                          for n, g in ends.items() if n != name)
        won = not ge.get("i_lost") and (
            bool(ge.get("opp_lost")) or others_lost
            or ge.get("opp_life", 1) <= 0)
        out[name] = (dec, won)
    return out


def log(msg):
    print(msg, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--games", type=int, default=64)
    ap.add_argument("--val-games", type=int, default=96)
    ap.add_argument("--lock", action="append", default=None)
    ap.add_argument("--lock-bonus", type=float, default=0.5)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--eps0", type=float, default=0.4)
    ap.add_argument("--eps-floor", type=float, default=0.1)
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--ckpt", default=str(OUT / "pilot2_v4.pt"))
    ap.add_argument("--tag", default="2",
                    help="round tag: journal/ckpt/archive names")
    ap.add_argument("--mull", action="store_true",
                    help="route mulligans through the bridge")
    ap.add_argument("--ramp-bonus", type=float, default=0.0,
                    help="reward for lands in play by turn 5 (our seat)")
    ap.add_argument("--lock-credit", type=float, default=0.0,
                    help="per-decision advantage boost for early lock "
                    "casts (surgical gradient, not trajectory-diluted)")
    ap.add_argument("--gpu", action="store_true",
                    help="batched PPO update on CUDA (casts batched; "
                    "other kinds per-decision)")
    ap.add_argument("--deck-pool", default=None,
                    help="dir of verified pool .dck files: mixed "
                    "curriculum (50%% our-deck games vs gauntlet+pool, "
                    "50%% pool-vs-pool for general skill)")
    ap.add_argument("--potential", type=float, default=0.0,
                    help="value-derived shaping weight: adv += "
                    "beta*(V(s') - V(s)) from the frozen value head")
    ap.add_argument("--ppo-epochs", type=int, default=0,
                    help=">0: PPO update (value-head baseline + clipped "
                    "K-epoch replay) instead of one-shot REINFORCE")
    ap.add_argument("--max-decisions", type=int, default=12000,
                    help="PPO: subsample cap per batch (update cost)")
    args = ap.parse_args()
    locks = args.lock if args.lock else ["Random Encounter"]
    global TAG
    TAG = args.tag

    import torch
    if args.mull:
        os.environ["FORGE_EXT_MULL"] = "1"
    jpath = OUT / f"selfplay_round{args.tag}.json"
    journal = (json.loads(jpath.read_text()) if jpath.exists()
               else {"train": [], "validation": {}})

    def save():
        jpath.write_text(json.dumps(journal, indent=1))

    rl_ckpt = OUT / f"pilot2_rl{args.tag}.pt"
    rl_state = OUT / f"pilot2_rl{args.tag}_state.pt"

    inner = ModelPolicy(ckpt=Path(args.ckpt))
    assert inner.has_tgt, "checkpoint lacks target head - run train_targets"
    inner.temperature = args.temperature
    nwk = max(1, args.parallel)
    workers = [SeatAwarePolicy(inner, locks, args.eps0,
                               np.random.default_rng(2000 + wk))
               for wk in range(nwk)]
    opt = torch.optim.Adam(inner.model.parameters(), lr=args.lr)
    it0, baseline = 0, 0.0
    if rl_state.exists():
        bundle = torch.load(rl_state, weights_only=False)
        inner.model.load_state_dict(bundle["model"])
        opt.load_state_dict(bundle["opt"])
        it0, baseline = bundle["iter"], bundle["baseline"]
        log(f"[resume] iteration {it0}, baseline {baseline:.3f}")

    games_dir = OUT / "games"
    games_dir.mkdir(exist_ok=True)
    index_lock = threading.Lock()

    def archive(it, wk, seq, opp, dec, won, deck_label=DECK, side=""):
        name = f"r{TAG}it{it:03d}_w{wk}_{seq:02d}{side}.json.gz"
        lf = game_lock_frac(dec, set(locks))
        dur = round(sum(s.get("_dt_ms", 0) for s, _r in dec) / 1000, 1)
        rec = {"iter": f"r{TAG}-{it}", "worker": wk, "seq": seq, "opp": opp,
               "won": won, "lock_frac": lf, "deck": deck_label,
               "dur_s": dur, "decisions": dec}
        with gzip.open(games_dir / name, "wt", encoding="utf-8") as f:
            json.dump(rec, f, separators=(",", ":"))
        with index_lock, open(OUT / "games_index.jsonl", "a",
                              encoding="utf-8") as f:
            f.write(json.dumps({"iter": f"r{TAG}-{it}", "worker": wk,
                                "opp": opp, "won": won, "lock_frac": lf,
                                "dur_s": dur, "n_dec": len(dec),
                                "t": int(time.time()), "file": name})
                    + "\n")

    if it0 < args.iters:
        servers = [start_server(0, policy=w) for w in workers]
        ports = [s.server_address[1] for s in servers]

        deck_pool = []
        if args.deck_pool:
            deck_pool = sorted(pp.stem for pp in
                               Path(args.deck_pool).glob("*.dck"))
            log(f"[pool] {len(deck_pool)} decks in mixed curriculum")

        def pick_matchup(rng):
            if deck_pool and float(rng.random()) < 0.5:
                i, j = rng.choice(len(deck_pool), 2, replace=False)
                return deck_pool[int(i)], deck_pool[int(j)]
            opps = GAUNTLET + deck_pool
            return DECK, opps[int(rng.integers(len(opps)))]

        prog = {"n": 0}

        def bump_progress(it, total):
            with index_lock:
                prog["n"] += 1
                try:
                    (OUT / "iter_progress.json").write_text(json.dumps(
                        {"iter": it, "done": prog["n"], "total": total}))
                except OSError:
                    pass

        def worker_games(it, wk, n):
            p, results = workers[wk], []
            for g in range(n):
                p.buffer = []
                deck_a, opp = pick_matchup(p.rng)
                run_bridged(deck_a, opp, 1, 240, ports[wk],
                            player_filter="", quiet=True, worker=wk)
                bump_progress(it, args.games)
                seats = split_seats(list(p.buffer))
                for name, (dec, won) in seats.items():
                    ours = DECK in name
                    lf = game_lock_frac(dec, set(locks)) if ours else 0.0
                    bonus = 0.0
                    if ours:
                        bonus, _flt = shaped_bonus(
                            dec, set(locks), inner.feat,
                            args.ramp_bonus, args.lock_bonus)
                    reward = (1.0 if won else -1.0) + bonus
                    results.append((dec, reward, won, lf, ours))
                    # archive EVERY seat of EVERY game: future models
                    # train on this corpus (the big net already did)
                    try:
                        own = re.sub(r"^Ai\(\d\)-", "", name)
                        other = opp if own == deck_a else deck_a
                        archive(it, wk, g, other, dec, won,
                                deck_label=own,
                                side="" if own == deck_a else "b")
                    except OSError:
                        pass
            return results

        try:
            counts = [args.games // nwk
                      + (1 if wk < args.games % nwk else 0)
                      for wk in range(nwk)]
            for it in range(it0, args.iters):
                eps = max(args.eps_floor,
                          args.eps0 * (1 - it / args.iters))
                for w in workers:
                    w.eps = eps
                prog["n"] = 0
                with ThreadPoolExecutor(max_workers=nwk) as pool:
                    futs = [pool.submit(worker_games, it, wk, counts[wk])
                            for wk in range(nwk)]
                    flown = [r for f in futs for r in f.result()]
                batch = [(dec, r) for dec, r, _w, _lf, _o in flown]
                lc = ({"locks": set(locks), "w": args.lock_credit}
                      if args.lock_credit else None)
                if args.ppo_epochs > 0 and args.gpu:
                    loss, vloss, meanv = ppo_update_gpu(
                        inner, torch, batch, opt,
                        epochs=args.ppo_epochs, lock_credit=lc,
                        max_decisions=args.max_decisions)
                elif args.ppo_epochs > 0:
                    loss, vloss, meanv = ppo_update(
                        inner, torch, batch, opt,
                        epochs=args.ppo_epochs, lock_credit=lc,
                        max_decisions=args.max_decisions,
                        potential=args.potential)
                else:
                    loss = reinforce_update(inner, torch, batch,
                                            baseline, opt, lock_credit=lc)
                    vloss = meanv = None
                rewards = [r for _d, r in batch]
                baseline = 0.7 * baseline + 0.3 * float(np.mean(rewards))
                ours = [x for x in flown if x[4]]
                n = max(1, len(ours))
                entry = {"iter": it,
                         "winrate": round(sum(x[2] for x in ours) / n, 3),
                         "lock_frac": round(sum(x[3] for x in ours) / n, 3),
                         "eps": round(eps, 2), "loss": round(loss, 4),
                         "games": len(ours), "trajs": len(flown),
                         "t": int(time.time())}
                if vloss is not None:
                    entry["vloss"] = round(vloss, 4)
                    entry["meanV"] = round(meanv, 3)
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
            import sim_server
            with sim_server._SHARED_LOCK:
                for c in sim_server._SHARED.values():
                    c.close()
                sim_server._SHARED.clear()

    # ---- validation: our seat only, builtin opponent, greedy ------------
    val = journal["validation"]
    vkey = f"rl{args.tag}"
    if vkey not in val:
        per = max(4, args.val_games // len(GAUNTLET))
        chunks_per_g = 4
        per_chunk = max(2, per // chunks_per_g)
        jobs = [g for g in GAUNTLET for _c in range(chunks_per_g)]
        part = val.get("_partial", {"wins": 0, "games": 0, "jobs": 0})
        if part["jobs"] < len(jobs):
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
                for w0 in range(0, len(todo), nwk):
                    wave = todo[w0:w0 + nwk]
                    with ThreadPoolExecutor(max_workers=nwk) as pool:
                        outs = list(pool.map(one_chunk, wave))
                    for out in outs:
                        part["wins"] += len(re.findall(
                            rf"Game Result.*Ai\(1\)-{DECK} has won", out))
                        part["games"] += len(re.findall(r"Game Result", out))
                    part["jobs"] = wave[-1][0] + 1
                    val["_partial"] = part
                    save()
            finally:
                srv.shutdown()
                srv.server_close()
        w, n = part["wins"], part["games"]
        p, half = ci95(w, n)
        val[vkey] = {"wins": w, "games": n,
                      "winrate": round(p, 3), "ci": round(half, 3)}
        val.pop("_partial", None)
        save()

    r = val[vkey]
    log(f"[verdict] {vkey} {r['winrate']:.0%} Â±{r['ci']:.0%}"
        f" vs round-1 rl 27% vs builtin 26% (288-game confirmation arms)")


if __name__ == "__main__":
    main()

