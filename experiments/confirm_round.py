"""Confirmation run: RL pilot vs builtin at 288 games/arm.

The 96-game gate read rl 30%+-9 vs builtin 22%+-8 (overlapping CIs,
same as the container run). Per the measurement culture, no accept
without a confirmation-scale rerun: 288 games/arm halves the CI to
~+-5.5%. Both arms fly the same gauntlet; banked per chunk, resume-safe.

Usage:
  FORGE_SIM_SERVER=1 python experiments/confirm_round.py --parallel 8
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pilot_bridge import ModelPolicy, start_server, run_bridged  # noqa: E402
from improve_deck import ci95  # noqa: E402
from self_play import RecordingPolicy, game_lock_frac  # noqa: E402

OUT = Path(__file__).resolve().parent / "output"
DECK = "fac_roaming"
GAUNTLET = ["fac_g0", "fac_g1", "fac_g2"]
PER_CHUNK = 8
LOCKS = {"Random Encounter"}


def log(msg):
    print(msg, flush=True)


def archive_chunk(arm, ji, opp, buffer):
    """Split a chunk's decision stream on game_end records and archive
    each game for the dashboard inspector (rl arm only)."""
    import gzip
    import time
    games_dir = OUT / "games"
    games_dir.mkdir(exist_ok=True)
    seg = []
    k = 0
    for state, reply in buffer:
        seg.append((state, reply))
        if state.get("kind") != "game_end":
            continue
        won = bool(state.get("opp_lost")) and not state.get("i_lost")
        name = f"cf{arm}_j{ji:02d}_{k:02d}.json.gz"
        lf = game_lock_frac(seg, LOCKS)
        dur = round(sum(s.get("_dt_ms", 0) for s, _r in seg) / 1000, 1)
        rec = {"iter": f"cf-{arm}", "worker": ji, "seq": k, "opp": opp,
               "won": won, "lock_frac": lf, "deck": DECK, "dur_s": dur,
               "decisions": seg}
        with gzip.open(games_dir / name, "wt", encoding="utf-8") as f:
            json.dump(rec, f, separators=(",", ":"))
        with open(OUT / "games_index.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"iter": f"cf-{arm}", "worker": ji,
                                "opp": opp, "won": won, "lock_frac": lf,
                                "dur_s": dur, "n_dec": len(seg),
                                "t": int(time.time()), "file": name})
                    + "\n")
        seg, k = [], k + 1


class CompositePolicy:
    """Two pilots in one game: our seat -> `mine` (recorded), opponent
    seats -> `opp` (the frozen benchmark). Dispatch by seat name."""

    def __init__(self, mine, opp):
        self.mine = mine          # RecordingPolicy
        self.opp = opp            # plain ModelPolicy

    @property
    def buffer(self):
        return self.mine.buffer

    @buffer.setter
    def buffer(self, v):
        self.mine.buffer = v

    def __call__(self, state):
        if DECK in state.get("player", DECK):
            return self.mine(state)
        return self.opp(state)


def make_policy(ckpt, arch=None):
    import os
    if not arch:
        return ModelPolicy(ckpt=ckpt)
    d, layers = arch.split(",")
    old = (os.environ.get("PILOT_D"), os.environ.get("PILOT_LAYERS"))
    os.environ["PILOT_D"], os.environ["PILOT_LAYERS"] = d, layers
    try:
        return ModelPolicy(ckpt=ckpt)
    finally:
        for k, v in zip(("PILOT_D", "PILOT_LAYERS"), old):
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def run_arm(name, journal, jpath, games_per_deck, nwk, ckpt=None,
            opp_ckpt=None, opp_arch=None):
    arm = journal.setdefault(name, {"wins": 0, "games": 0, "jobs": 0})
    chunks_per_deck = games_per_deck // PER_CHUNK
    jobs = [g for g in GAUNTLET for _ in range(chunks_per_deck)]
    if arm["jobs"] >= len(jobs):
        return arm

    bridged = ckpt is not None
    workers, servers, ports = [], [], []
    if bridged:
        inner = ModelPolicy(ckpt=ckpt)
        opp = (make_policy(opp_ckpt, opp_arch) if opp_ckpt else None)
        for _ in range(nwk):
            rec = RecordingPolicy(inner)
            workers.append(CompositePolicy(rec, opp) if opp else rec)
        servers = [start_server(0, policy=w) for w in workers]
        ports = [s.server_address[1] for s in servers]
    pfilter = "" if (bridged and opp_ckpt) else (DECK if bridged else None)

    def one(job):
        ji, g = job
        wk = ji % nwk           # distinct within a wave (consecutive ji)
        if bridged:
            workers[wk].buffer = []
        out = run_bridged(DECK, g, PER_CHUNK, 90 + 40 * PER_CHUNK,
                          ports[wk] if bridged else None,
                          player_filter=pfilter,
                          quiet=True, worker=wk)
        if bridged:
            try:
                archive_chunk(name, ji, g, list(workers[wk].buffer))
            except OSError:
                pass
        return out

    try:
        todo = [(ji, g) for ji, g in enumerate(jobs) if ji >= arm["jobs"]]
        for w0 in range(0, len(todo), nwk):
            wave = todo[w0:w0 + nwk]
            with ThreadPoolExecutor(max_workers=nwk) as pool:
                outs = list(pool.map(one, wave))
            for out in outs:
                arm["wins"] += len(re.findall(
                    rf"Game Result.*Ai\(1\)-{DECK} has won", out))
                arm["games"] += len(re.findall(r"Game Result", out))
            arm["jobs"] = wave[-1][0] + 1
            jpath.write_text(json.dumps(journal, indent=1))
            p, half = ci95(arm["wins"], arm["games"])
            log(f"[{name}] {arm['wins']}/{arm['games']} = {p:.0%} ±{half:.0%}")
    finally:
        for s in servers:
            s.shutdown()
            s.server_close()
    return arm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games-per-deck", type=int, default=96)  # x3 decks
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--ckpt", default=str(OUT / "pilot2_rl.pt"))
    ap.add_argument("--arm", default="rl",
                    help="journal key for the pilot arm (e.g. rl2)")
    ap.add_argument("--journal", default="confirm_round.json",
                    help="journal filename (new deck = new journal)")
    ap.add_argument("--opp-ckpt", default=None,
                    help="checkpoint piloting the OPPONENT seats (the "
                    "frozen benchmark); default: builtin AI opponents")
    ap.add_argument("--opp-arch", default=None,
                    help="D,layers for the opponent net if it differs "
                    "from PILOT_D/PILOT_LAYERS (e.g. '256,4')")
    ap.add_argument("--skip-pilot", action="store_true",
                    help="builtin arm only")
    args = ap.parse_args()

    jpath = OUT / args.journal
    journal = json.loads(jpath.read_text()) if jpath.exists() else {}

    # arm 1: RL pilot (greedy, no exploration), games archived
    rl = None
    if not args.skip_pilot:
        rl = run_arm(args.arm, journal, jpath, args.games_per_deck,
                     args.parallel, ckpt=Path(args.ckpt),
                     opp_ckpt=args.opp_ckpt, opp_arch=args.opp_arch)
    import sim_server
    with sim_server._SHARED_LOCK:
        for c in sim_server._SHARED.values():
            c.close()
        sim_server._SHARED.clear()

    # arm 2: builtin AI (no bridge, nothing to archive)
    bi = run_arm("builtin", journal, jpath, args.games_per_deck,
                 args.parallel, ckpt=None)

    if rl is None:
        pb, hb = ci95(bi["wins"], bi["games"])
        log(f"[confirm-verdict] builtin {pb:.0%} ±{hb:.0%} (baseline only)")
        return
    pr, hr = ci95(rl["wins"], rl["games"])
    pb, hb = ci95(bi["wins"], bi["games"])
    sep = (pr - hr) > (pb + hb)
    log(f"[confirm-verdict] {args.arm} {pr:.0%} ±{hr:.0%} vs builtin "
        f"{pb:.0%} ±{hb:.0%} -> "
        f"{'CONFIRMED GAIN' if sep else 'no detectable change'}")


if __name__ == "__main__":
    main()
