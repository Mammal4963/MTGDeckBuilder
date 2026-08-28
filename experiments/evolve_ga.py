"""Deck evolution: raw genetic algorithm, zero structural bias.

Genome = a legal 60-card multiset over Forge's full implemented card
universe (33.5k cards). No skeleton, no manabase repair - lands are
just cards. The only hard constraints are game legality (4-copy limit,
basics exempt) and deck size 60.

Phase 0 (primordial): fitness = round-robin play WITHIN the population
(noise decks beat each other -> gradient exists at the floor).
Graduation: when a generation champion starts taking games off a weak
external reference, fitness switches to the frozen evolver meta -
comparable scores, racing, elites accumulating games across
generations. The benchmark pilot plays BOTH seats of every game, so
fitness measures the deck, not piloting asymmetry.

Operators: random-sample crossover (k from parent A, 60-k from B),
per-slot mutation from the universe with a SELF-ADAPTIVE rate (each
lineage carries its own rate gene, log-normal perturbed each
inheritance), and duplication drift (+1 copy of one card, -1 other).

State persists per campaign in output/evolve/<name>/ - rerun to
resume. Progress: evolve_progress.json (dashboard tile) + history.

Usage:
  FORGE_SIM_SERVER=1 PILOT_D=192 PILOT_LAYERS=6 \
      python experiments/evolve_ga.py --name genesis --pop 20 --gens 40
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pilot_bridge import ModelPolicy, start_server, run_bridged  # noqa: E402
from improve_deck import FORGE_DECKS  # noqa: E402
from self_play import RecordingPolicy  # noqa: E402
import threading  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
BASICS = {"Plains", "Island", "Swamp", "Mountain", "Forest", "Wastes",
          "Snow-Covered Plains", "Snow-Covered Island",
          "Snow-Covered Swamp", "Snow-Covered Mountain",
          "Snow-Covered Forest"}
DECK_SIZE = 60
MUT0, MUT_MIN, MUT_MAX = 0.08, 0.002, 0.3


def cap(card):
    return 99 if card in BASICS else 4


class Genome:
    __slots__ = ("cards", "mut", "wins", "games", "gid")

    def __init__(self, cards, mut, gid):
        self.cards = cards          # dict name -> count, sums to 60
        self.mut = mut
        self.wins = 0               # accumulated (meta phase)
        self.games = 0
        self.gid = gid

    def fitness(self):
        return self.wins / self.games if self.games else 0.0

    def to_json(self):
        return {"cards": self.cards, "mut": self.mut,
                "wins": self.wins, "games": self.games,
                "gid": self.gid}

    @staticmethod
    def from_json(d):
        g = Genome(d["cards"], d["mut"], d["gid"])
        g.wins, g.games = d["wins"], d["games"]
        return g


def rand_deck(universe, rng, gid):
    cards = {}
    while sum(cards.values()) < DECK_SIZE:
        c = universe[int(rng.integers(len(universe)))]
        if cards.get(c, 0) < cap(c):
            cards[c] = cards.get(c, 0) + 1
    return Genome(cards, MUT0, gid)


def expand(cards):
    return [c for c, n in cards.items() for _ in range(n)]


def collapse(lst):
    d = {}
    for c in lst:
        d[c] = d.get(c, 0) + 1
    return d


def crossover(a, b, universe, rng, gid):
    k = int(rng.integers(1, DECK_SIZE))
    ea, eb = expand(a.cards), expand(b.cards)
    rng.shuffle(ea)
    rng.shuffle(eb)
    child = []
    counts = {}
    for src, want in ((ea, k), (eb, DECK_SIZE - k)):
        got = 0
        for c in src:
            if got >= want:
                break
            if counts.get(c, 0) < cap(c):
                child.append(c)
                counts[c] = counts.get(c, 0) + 1
                got += 1
    while len(child) < DECK_SIZE:       # cap collisions: fill random
        c = universe[int(rng.integers(len(universe)))]
        if counts.get(c, 0) < cap(c):
            child.append(c)
            counts[c] = counts.get(c, 0) + 1
    mut = float(np.clip((a.mut + b.mut) / 2
                        * np.exp(rng.normal(0, 0.25)),
                        MUT_MIN, MUT_MAX))
    return Genome(collapse(child), mut, gid)


def mutate(g, universe, rng):
    lst = expand(g.cards)
    counts = dict(g.cards)
    for i in range(len(lst)):
        if rng.random() < g.mut:
            old = lst[i]
            for _try in range(8):
                c = universe[int(rng.integers(len(universe)))]
                if counts.get(c, 0) < cap(c):
                    counts[old] -= 1
                    counts[c] = counts.get(c, 0) + 1
                    lst[i] = c
                    break
    if rng.random() < 0.3:              # duplication drift
        dupable = [c for c in counts if 0 < counts[c] < cap(c)]
        if dupable:
            c = dupable[int(rng.integers(len(dupable)))]
            drop_pool = [x for x in counts
                         if counts[x] > 0 and x != c]
            if drop_pool:
                d = drop_pool[int(rng.integers(len(drop_pool)))]
                counts[c] += 1
                counts[d] -= 1
    g.cards = {c: n for c, n in counts.items() if n > 0}
    return g


def write_deck(name, cards):
    main = "\n".join(f"{n} {c}" for c, n in sorted(cards.items()))
    (FORGE_DECKS / f"{name}.dck").write_text(
        f"[metadata]\nName={name}\n[Main]\n{main}\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--pop", type=int, default=20)
    ap.add_argument("--gens", type=int, default=40)
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--p0-opps", type=int, default=3)
    ap.add_argument("--p0-games", type=int, default=4)
    ap.add_argument("--meta-games", type=int, default=16)
    ap.add_argument("--grad-ref", default="fac_g0",
                    help="weak external reference; champion taking "
                    "2/8 games off it triggers meta-phase graduation")
    args = ap.parse_args()

    universe = json.loads((OUT / "card_universe.json")
                          .read_text(encoding="utf-8"))
    camp = OUT / "evolve" / args.name
    camp.mkdir(parents=True, exist_ok=True)
    state_f = camp / "state.json"
    rng = np.random.default_rng(97)

    if state_f.exists():
        st = json.loads(state_f.read_text())
        pop = [Genome.from_json(d) for d in st["pop"]]
        gen0, phase, next_gid = st["gen"], st["phase"], st["next_gid"]
        rng = np.random.default_rng(st["seed"] + gen0)
        print(f"[resume] gen {gen0}, phase {phase}", flush=True)
    else:
        pop = [rand_deck(universe, rng, i) for i in range(args.pop)]
        gen0, phase, next_gid = 0, 0, args.pop

    meta = json.loads((OUT / "meta_v1.json").read_text())
    meta_decks = meta["decks"]
    bench = ModelPolicy(ckpt=OUT / meta["benchmark"])
    servers = [start_server(0, policy=bench)
               for _ in range(args.parallel)]
    ports = [s.server_address[1] for s in servers]
    # probe games are recorded and archived INSIDE the campaign dir -
    # strictly separate from the training corpus (never trained on)
    probe_rec = RecordingPolicy(bench)
    probe_srv = start_server(0, policy=probe_rec)
    probe_port = probe_srv.server_address[1]
    prog_lock = threading.Lock()
    live = {"done": 0, "total": 0, "games": 0}
    last_entry = {}
    hist_f = camp / "history.jsonl"
    if hist_f.exists():
        lines = hist_f.read_text(encoding="utf-8").splitlines()
        if lines:
            last_entry = json.loads(lines[-1])

    def bump_live():
        try:
            with prog_lock:
                (OUT / "evolve_progress.json").write_text(json.dumps(
                    {**last_entry, "name": args.name,
                     "gens": args.gens, "live": dict(live)}))
        except OSError:
            pass

    def archive_probe(gen, deck_name, buffer):
        import gzip
        gdir = camp / "games"
        gdir.mkdir(exist_ok=True)
        # champion's seat only: the server records BOTH seats, and
        # each game emits a game_end per seat - unfiltered, every
        # other "game" is a phantom 1-decision segment
        buffer = [x for x in buffer
                  if deck_name in x[0].get("player", "")]
        seg, k = [], 0
        for state, reply in buffer:
            seg.append((state, reply))
            if state.get("kind") != "game_end":
                continue
            won = bool(state.get("opp_lost")) \
                and not state.get("i_lost")
            fn = f"probe_gen{gen:03d}_{k:02d}.json.gz"
            with gzip.open(gdir / fn, "wt", encoding="utf-8") as f:
                json.dump({"gen": gen, "deck": deck_name, "won": won,
                           "decisions": seg}, f,
                          separators=(",", ":"))
            with open(camp / "probe_index.jsonl", "a",
                      encoding="utf-8") as f:
                f.write(json.dumps({"gen": gen, "file": fn,
                                    "won": won, "n_dec": len(seg),
                                    "t": int(time.time())}) + "\n")
            seg, k = [], k + 1

    def play(job):
        """(wk, deck_a, deck_b, n) -> (a_wins, b_wins, games)"""
        wk, da, db, n = job
        try:
            out = run_bridged(da, db, n, 120 + 60 * n,
                              ports[wk % len(ports)], player_filter="",
                              quiet=True, worker=wk % args.parallel)
            aw = len(re.findall(
                rf"Game Result.*Ai\(1\)-{re.escape(da)} has won", out))
            bw = len(re.findall(
                rf"Game Result.*Ai\(2\)-{re.escape(db)} has won", out))
            g = len(re.findall(r"Game Result", out))
            with prog_lock:
                live["done"] += 1
                live["games"] += g
            bump_live()
            return aw, bw, g
        except Exception:
            with prog_lock:
                live["done"] += 1
            bump_live()
            return 0, 0, 0

    for gen in range(gen0, args.gens):
        t0 = time.time()
        names = {}
        for g in pop:
            names[g.gid] = f"evo{args.name[:8]}_{g.gid:05d}"
            write_deck(names[g.gid], g.cards)

        gw = {g.gid: 0 for g in pop}    # this-gen wins
        gg = {g.gid: 0 for g in pop}    # this-gen games
        with prog_lock:
            live.update(done=0, games=0, gen=gen, phase=phase)
        if phase == 0:
            jobs = []
            for i, g in enumerate(pop):
                opps = rng.choice([x.gid for x in pop if x.gid != g.gid],
                                  args.p0_opps, replace=False)
                for o in opps:
                    jobs.append((len(jobs), names[g.gid],
                                 names[int(o)], args.p0_games,
                                 g.gid, int(o)))
            live["total"] = len(jobs)
            with ThreadPoolExecutor(max_workers=args.parallel) as tp:
                res = list(tp.map(
                    lambda j: (j[4], j[5], play(j[:4])), jobs))
            for ga, gb, (aw, bw, n) in res:
                gw[ga] += aw
                gg[ga] += n
                gw[gb] += bw
                gg[gb] += n
            for g in pop:                       # phase-0 fitness is
                g.wins, g.games = gw[g.gid], gg[g.gid]   # per-gen only
        else:
            jobs = []
            for i, g in enumerate(pop):
                picks = rng.choice(meta_decks, args.meta_games // 2,
                                   replace=False)
                for o in picks:
                    jobs.append((len(jobs), names[g.gid], str(o), 2,
                                 g.gid, None))
            live["total"] = len(jobs)
            with ThreadPoolExecutor(max_workers=args.parallel) as tp:
                res = list(tp.map(
                    lambda j: (j[4], play(j[:4])), jobs))
            for ga, (aw, _bw, n) in res:
                gw[ga] += aw
                gg[ga] += n
            for g in pop:                       # accumulate over gens
                g.wins += gw[g.gid]
                g.games += gg[g.gid]

        pop.sort(key=lambda g: -g.fitness())
        champ = pop[0]

        # graduation probe (phase 0 only) - recorded, archived in the
        # campaign dir only (never the training corpus)
        grad = ""
        if phase == 0:
            probe_rec.buffer = []
            try:
                out = run_bridged(names[champ.gid], args.grad_ref, 8,
                                  120 + 60 * 8, probe_port,
                                  player_filter="", quiet=True,
                                  worker=args.parallel)
                aw = len(re.findall(
                    rf"Game Result.*Ai\(1\)-"
                    rf"{re.escape(names[champ.gid])} has won", out))
                n = len(re.findall(r"Game Result", out))
            except Exception:
                aw, n = 0, 0
            try:
                archive_probe(gen, names[champ.gid],
                              list(probe_rec.buffer))
            except OSError:
                pass
            grad = f" · grad-probe {aw}/{n}"
            if aw >= 2:
                phase = 1
                for g in pop:
                    g.wins = g.games = 0        # fresh comparable slate
                grad += " -> GRADUATED to meta fitness"

        # next generation: elites + children
        n_elite = max(2, args.pop // 4)
        elites = pop[:n_elite]
        children = []
        parents = pop[:max(2, args.pop // 2)]
        while len(children) < args.pop - n_elite:
            a, b = rng.choice(len(parents), 2, replace=False)
            child = crossover(parents[int(a)], parents[int(b)],
                              universe, rng, next_gid)
            next_gid += 1
            children.append(mutate(child, universe, rng))
        pop = elites + children

        avg_mut = sum(g.mut for g in pop) / len(pop)
        lands_set = getattr(main, "_lands", None)
        if lands_set is None:
            try:
                main._lands = lands_set = set(json.loads(
                    (OUT / "card_lands.json").read_text(
                        encoding="utf-8")))
            except OSError:
                main._lands = lands_set = set()

        def nlands(g):
            return sum(n for c, n in g.cards.items()
                       if c in lands_set)
        entry = {"gen": gen, "phase": phase,
                 "best": round(champ.fitness(), 3),
                 "best_games": champ.games,
                 "mean": round(sum(gw[g] / max(1, gg[g])
                                   for g in gw) / len(gw), 3),
                 "avg_mut": round(avg_mut, 4),
                 "lands_mean": round(sum(nlands(g) for g in pop)
                                     / len(pop), 1),
                 "lands_champ": nlands(champ),
                 "dur_s": round(time.time() - t0), "t": int(time.time())}
        print(f"[gen {gen}] phase {phase} best "
              f"{entry['best']:.0%} ({champ.games}g) mean "
              f"{entry['mean']:.0%} mut {avg_mut:.3f} "
              f"({entry['dur_s']}s){grad}", flush=True)
        with open(camp / "history.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        (camp / f"champion_gen{gen:03d}.dck").write_text(
            (FORGE_DECKS / f"{names[champ.gid]}.dck").read_text(
                encoding="utf-8"), encoding="utf-8")
        state_f.write_text(json.dumps(
            {"gen": gen + 1, "phase": phase, "next_gid": next_gid,
             "seed": 97, "pop": [g.to_json() for g in pop]}))
        last_entry = entry
        try:
            (OUT / "evolve_progress.json").write_text(json.dumps(
                {**entry, "name": args.name, "gens": args.gens}))
        except OSError:
            pass

    for s in servers + [probe_srv]:
        s.shutdown()
        s.server_close()
    print(f"[evolve] campaign {args.name} done at gen {args.gens}",
          flush=True)


if __name__ == "__main__":
    main()
