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


def run_arm(name, journal, jpath, games_per_deck, nwk, ckpt=None):
    arm = journal.setdefault(name, {"wins": 0, "games": 0, "jobs": 0})
    chunks_per_deck = games_per_deck // PER_CHUNK
    jobs = [g for g in GAUNTLET for _ in range(chunks_per_deck)]
    if arm["jobs"] >= len(jobs):
        return arm

    bridged = ckpt is not None
    workers, servers, ports = [], [], []
    if bridged:
        inner = ModelPolicy(ckpt=ckpt)
        workers = [RecordingPolicy(inner) for _ in range(nwk)]
        servers = [start_server(0, policy=w) for w in workers]
        ports = [s.server_address[1] for s in servers]

    def one(job):
        ji, g = job
        wk = ji % nwk           # distinct within a wave (consecutive ji)
        if bridged:
            workers[wk].buffer = []
        out = run_bridged(DECK, g, PER_CHUNK, 90 + 40 * PER_CHUNK,
                          ports[wk] if bridged else None,
                          player_filter=DECK if bridged else None,
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
    args = ap.parse_args()

    jpath = OUT / "confirm_round.json"
    journal = json.loads(jpath.read_text()) if jpath.exists() else {}

    # arm 1: RL pilot (greedy, no exploration), games archived
    rl = run_arm("rl", journal, jpath, args.games_per_deck,
                 args.parallel, ckpt=OUT / "pilot2_rl.pt")
    import sim_server
    with sim_server._SHARED_LOCK:
        for c in sim_server._SHARED.values():
            c.close()
        sim_server._SHARED.clear()

    # arm 2: builtin AI (no bridge, nothing to archive)
    bi = run_arm("builtin", journal, jpath, args.games_per_deck,
                 args.parallel, ckpt=None)

    pr, hr = ci95(rl["wins"], rl["games"])
    pb, hb = ci95(bi["wins"], bi["games"])
    sep = (pr - hr) > (pb + hb)
    log(f"[confirm-verdict] rl {pr:.0%} ±{hr:.0%} vs builtin {pb:.0%} "
        f"±{hb:.0%} -> {'CONFIRMED GAIN' if sep else 'no detectable change'}")


if __name__ == "__main__":
    main()
