"""The generic benchmark runner: [pilot A + deck] vs [pilot B + deck].

Every benchmark in the project is this primitive with different
things held fixed:

  pilot-skill:  gauntlet.py --label pilot-skill --a-ckpt CAND.pt
                  --decks meta_v1 --swap
                (B defaults to the frozen benchmark; --swap makes A
                 play BOTH decks of every pairing so deck strength
                 cancels; Forge alternates play/draw per game)
  deck-eval:    gauntlet.py --label deck-eval --a-deck MY_DECK
                  --a-ckpt benchmark_v2.pt --b-ckpt benchmark_v2.pt
                  --decks meta_v1
                (same pilot both seats -> measures the DECK)
  ft-progress:  gauntlet.py --label ft-progress --a-ckpt GEN_N.pt
                  --a-arch 192,6 --a-deck MY_DECK --decks meta_v1
                (fixed deck + fixed opposition -> the tuning curve;
                 run gen 0 = the base generalist as the baseline)
  league:       gauntlet.py --label league --a-ckpt P1.pt --a-deck D1
                  --b-ckpt P2.pt --b-deck D2 --games 96

Results append to output/benchmarks.jsonl (one summary record per
run) and show in the dashboard's Benchmarks panel, grouped by label.
Seat A is always Ai(1); dispatch is by seat, so mirrors and asymmetric
pairings both work.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pilot_bridge import ModelPolicy, start_server, run_bridged  # noqa: E402
from improve_deck import ci95  # noqa: E402
from self_play import RecordingPolicy  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
PER_CHUNK = 4      # small chunks so orientations interleave tightly
_io_lock = __import__("threading").Lock()


def make_policy(ckpt, arch):
    import os
    old = (os.environ.get("PILOT_D"), os.environ.get("PILOT_LAYERS"))
    d, layers = arch.split(",")
    os.environ["PILOT_D"], os.environ["PILOT_LAYERS"] = d, layers
    try:
        return ModelPolicy(ckpt=ckpt)
    finally:
        for k, v in zip(("PILOT_D", "PILOT_LAYERS"), old):
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class SeatPolicy:
    """A drives Ai(1) (recorded), B drives Ai(2)."""

    def __init__(self, a, b):
        self.a, self.b = a, b

    @property
    def buffer(self):
        return self.a.buffer

    @buffer.setter
    def buffer(self, v):
        self.a.buffer = v

    def __call__(self, state):
        p = state.get("player", "Ai(1)")
        return self.a(state) if p.startswith("Ai(1)") else self.b(state)


def archive_games(run_id, ji, deck_a, opp, buffer):
    """Split the seat-A decision stream on game_end and archive each
    game so the dashboard inspector can replay benchmark games."""
    import gzip
    games_dir = OUT / "games"
    games_dir.mkdir(exist_ok=True)
    seg, k = [], 0
    for state, reply in buffer:
        seg.append((state, reply))
        if state.get("kind") != "game_end":
            continue
        won = bool(state.get("opp_lost")) and not state.get("i_lost")
        name = f"cf{run_id}_j{ji:02d}_{k:02d}.json.gz"
        dur = round(sum(s.get("_dt_ms", 0) for s, _r in seg) / 1000, 1)
        rec = {"iter": f"cf-{run_id}", "worker": ji, "seq": k,
               "opp": opp, "won": won, "lock_frac": 0.0,
               "deck": deck_a, "dur_s": dur, "decisions": seg}
        with gzip.open(games_dir / name, "wt", encoding="utf-8") as f:
            json.dump(rec, f, separators=(",", ":"))
        with _io_lock, open(OUT / "games_index.jsonl", "a",
                            encoding="utf-8") as f:
            f.write(json.dumps({"iter": f"cf-{run_id}", "worker": ji,
                                "deck": deck_a, "opp": opp, "won": won,
                                "lock_frac": 0.0, "dur_s": dur,
                                "n_dec": len(seg),
                                "t": int(time.time()), "file": name})
                    + "\n")
        seg, k = [], k + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True,
                    help="what this run measures: pilot-skill | "
                    "deck-eval | ft-progress | league | ...")
    ap.add_argument("--a-ckpt", required=True)
    ap.add_argument("--a-arch", default="192,6")
    ap.add_argument("--a-deck", default=None,
                    help="fixed deck for seat A (deck-eval, "
                    "ft-progress, league); default: pairing decks")
    ap.add_argument("--b-ckpt", default=None,
                    help="seat-B pilot; default: the frozen benchmark")
    ap.add_argument("--b-arch", default=None)
    ap.add_argument("--b-deck", default=None,
                    help="single fixed deck for seat B (league mode)")
    ap.add_argument("--decks", default="meta_v1",
                    help="'meta_v1' or comma-separated opponent decks")
    ap.add_argument("--games", type=int, default=288,
                    help="total games for the run")
    ap.add_argument("--swap", action="store_true",
                    help="A also plays each pairing's other deck "
                    "(cancels deck strength; pilot-skill mode)")
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--note", default="")
    ap.add_argument("--meta-subset", type=int, default=0,
                    help="use only the first N meta decks (frozen "
                    "order) - cheap runs that stay comparable")
    args = ap.parse_args()

    if args.b_ckpt is None:
        bench = json.loads((OUT / "meta_v1.json").read_text())
        args.b_ckpt = str(OUT / bench["benchmark"])
        args.b_arch = args.b_arch or bench["arch"]
    args.b_arch = args.b_arch or "192,6"

    if args.decks == "meta_v1":
        meta = json.loads((OUT / "meta_v1.json").read_text())
        opp_decks = meta["decks"]
        if args.meta_subset:
            # deterministic prefix: cheap runs stay comparable
            opp_decks = opp_decks[:args.meta_subset]
    else:
        opp_decks = [d.strip() for d in args.decks.split(",")]
    if args.b_deck:
        opp_decks = [args.b_deck]

    pol_a = make_policy(args.a_ckpt, args.a_arch)
    same = (Path(args.a_ckpt).resolve() == Path(args.b_ckpt).resolve()
            and args.a_arch == args.b_arch)
    pol_b = pol_a if same else make_policy(args.b_ckpt, args.b_arch)
    workers = [SeatPolicy(RecordingPolicy(pol_a), pol_b)
               for _ in range(args.parallel)]
    servers = [start_server(0, policy=w) for w in workers]
    ports = [s.server_address[1] for s in servers]
    run_id = f"bench{int(time.time()) % 100000:05d}"

    # job list: (deck_for_A, deck_for_B), interleaving orientations
    pairings = []
    for opp in opp_decks:
        a_deck = args.a_deck or opp_decks[
            (opp_decks.index(opp) + 1) % len(opp_decks)]
        pairings.append((a_deck, opp))
        if args.swap:
            pairings.append((opp, a_deck))
    # spread the game budget over pairings; with a big meta (200
    # decks) each pairing gets 1-2 games - fine, the pooled CI comes
    # from total games and the warm JVMs make small chunks cheap
    chunk = max(1, min(PER_CHUNK, args.games // len(pairings)))
    reps = max(1, round(args.games / (chunk * len(pairings))))
    jobs = []
    for _k in range(reps):
        for pa, pb in pairings:
            jobs.append((pa, pb, chunk))

    results = {}

    total_planned = sum(j[2] for j in jobs)

    def one(job):
        ji, (da, db, chunk) = job
        wk = ji % args.parallel
        workers[wk].buffer = []
        out = run_bridged(da, db, chunk, 90 + 40 * chunk,
                          ports[wk], player_filter="",
                          quiet=True, worker=wk)
        try:
            archive_games(run_id, ji, da, db,
                          list(workers[wk].buffer))
        except OSError:
            pass
        wins = len(re.findall(
            rf"Game Result.*Ai\(1\)-{re.escape(da)} has won", out))
        games = len(re.findall(r"Game Result", out))
        return (da, db), wins, games

    t0 = time.time()
    done_w = done_g = 0
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        for pair, wins, games in pool.map(
                one, list(enumerate(jobs))):
            r = results.setdefault(f"{pair[0]}|{pair[1]}", [0, 0])
            r[0] += wins
            r[1] += games
            done_w += wins
            done_g += games
            p, half = ci95(done_w, done_g)
            print(f"[{args.label}] {done_w}/{done_g} = "
                  f"{p:.0%} ±{half:.0%}", flush=True)
            try:
                with _io_lock:
                    (OUT / "bench_progress.json").write_text(
                        json.dumps({"label": args.label,
                                    "a_ckpt": Path(args.a_ckpt).name,
                                    "wins": done_w, "games": done_g,
                                    "total": total_planned,
                                    "t": int(time.time())}))
            except OSError:
                pass
    for s in servers:
        s.shutdown()
        s.server_close()

    p, half = ci95(done_w, done_g)
    rec = {"t": int(time.time()), "label": args.label,
           "a_ckpt": Path(args.a_ckpt).name, "a_arch": args.a_arch,
           "a_deck": args.a_deck, "b_ckpt": Path(args.b_ckpt).name,
           "decks": opp_decks, "swap": args.swap, "note": args.note,
           "wins": done_w, "games": done_g, "wr": round(p, 4),
           "ci": round(half, 4), "dur_s": round(time.time() - t0),
           "pairs": {k: {"wins": v[0], "games": v[1]}
                     for k, v in results.items()}}
    with open(OUT / "benchmarks.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    (OUT / "bench_progress.json").unlink(missing_ok=True)
    print(f"[{args.label}] FINAL {done_w}/{done_g} = {p:.0%} "
          f"±{half:.0%} ({rec['dur_s']}s) -> benchmarks.jsonl",
          flush=True)


if __name__ == "__main__":
    main()
