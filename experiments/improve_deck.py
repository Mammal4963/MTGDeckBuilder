"""Deck improvement loop: geometry proposes swaps, Forge sims decide.

    python3 experiments/improve_deck.py mydeck.txt --format pauper

Pipeline
  1. Read the deck; embed it in the 60-card co-occurrence space.
  2. Propose swaps: cut candidates = the deck's lowest-synergy /
     least-played nonland cards; add candidates = format-legal,
     Forge-supported cards with the highest neighbor-deck presence and
     synergy to the deck. One playset-level swap per candidate.
  3. Build a diverse gauntlet from the MTGO corpus (same format).
  4. Race: every variant (baseline included) plays a short stage against
     the whole gauntlet; the top variants advance to a longer stage.
  5. Report win rates with normal-approximation CIs, JSON + table.

Forge specifics learned in the spike (see FORGE-NOTES.md): xvfb-run
required, decks live in ~/.forge/decks/constructed, ~2.6 s/game.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mtg_deckbuilder.carddata import CardDatabase  # noqa: E402
from mtg_deckbuilder.collection import parse_collection  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "output"
SCRATCH = Path("/tmp/claude-0/-home-user-MTGDeckBuilder/6e07b8e8-48d2-56ad-84cc-9478073b7720/scratchpad")
FORGE_DIR = SCRATCH / "forge"
FORGE_DECKS = Path.home() / ".forge" / "decks" / "constructed"
SUPPORTED_CACHE = SCRATCH / "forge-supported-names.json"

RNG = np.random.default_rng(23)


def norm(name: str) -> str:
    return re.sub(r"\s+", " ", name).strip().lower()


def forge_supported_names() -> set:
    if SUPPORTED_CACHE.exists():
        return set(json.loads(SUPPORTED_CACHE.read_text()))
    z = zipfile.ZipFile(FORGE_DIR / "res" / "cardsfolder" / "cardsfolder.zip")
    names = set()
    for entry in z.namelist():
        if not entry.endswith(".txt"):
            continue
        head = z.read(entry).split(b"\n", 1)[0].decode("utf-8", "replace")
        if head.startswith("Name:"):
            names.add(norm(head[5:]))
    SUPPORTED_CACHE.write_text(json.dumps(sorted(names)))
    return names


class Improver:
    def __init__(self, fmt: str):
        self.fmt = fmt
        self.vecs = np.load(OUT / "covectors-60.npy")
        self.meta = json.loads((OUT / "cards_meta.json").read_text())
        self.name_to_row = {}
        for i, m in enumerate(self.meta):
            self.name_to_row.setdefault(norm(m["name"]), i)
            if " // " in m["name"]:
                self.name_to_row.setdefault(norm(m["name"].split(" // ")[0]), i)
        self.is_land = np.array([("Land" in m["type_line"]) for m in self.meta])
        self.db = CardDatabase.load(
            ROOT / "data" / "scryfall-oracle-cards-2026-08-12.jsonl.gz")
        self.supported = forge_supported_names()

        import gzip
        self.corpus = []
        with gzip.open(ROOT / "data" / "decks" / "mtgo-decks.jsonl.gz", "rt") as fh:
            for line in fh:
                d = json.loads(line)
                if d["format"] != fmt:
                    continue
                rows = {}
                ok = True
                for n, q in d["main"]:
                    r = self.name_to_row.get(norm(n))
                    if r is None:
                        ok = False
                        break
                    rows[r] = rows.get(r, 0) + q
                if ok and rows:
                    self.corpus.append({"main": d["main"], "rows": rows,
                                        "wins": d.get("wins", 0),
                                        "event": d["event"]})
        print(f"{len(self.corpus)} corpus decks in {fmt}")
        self.centroids = np.zeros((len(self.corpus), self.vecs.shape[1]), np.float32)
        for i, d in enumerate(self.corpus):
            nl = [(r, q) for r, q in d["rows"].items() if not self.is_land[r]]
            v = sum(self.vecs[r] * q for r, q in nl)
            v /= max(np.linalg.norm(v), 1e-9)
            self.centroids[i] = v

    def legal(self, row: int) -> bool:
        card = self.db.get(self.meta[row]["name"])
        return card is not None and card.legalities.get(self.fmt) in ("legal", "restricted")

    def playable(self, row: int) -> bool:
        return norm(self.meta[row]["name"]) in self.supported

    # ---------------- proposer ----------------

    def propose(self, deck: dict, n_swaps: int, locked: set):
        """deck: {row: qty}. Returns list of (cut_row, add_row, qty)."""
        nonland = {r: q for r, q in deck.items() if not self.is_land[r]}
        centroid = sum(self.vecs[r] * q for r, q in nonland.items())
        centroid /= max(np.linalg.norm(centroid), 1e-9)

        sims = self.centroids @ centroid
        neighbors = np.argsort(-sims)[:40]
        presence = defaultdict(float)
        for d in neighbors:
            w = sims[d]
            for r in self.corpus[d]["rows"]:
                if not self.is_land[r]:
                    presence[r] += w
        max_p = max(presence.values()) if presence else 1.0

        def cut_score(r):
            return float(self.vecs[r] @ centroid) + presence.get(r, 0) / max_p

        cuts = sorted(
            (r for r in nonland if norm(self.meta[r]["name"]) not in locked),
            key=cut_score)[:max(3, n_swaps // 2)]

        adds = []
        for r, p in sorted(presence.items(), key=lambda kv: -kv[1]):
            if r in deck or not self.legal(r) or not self.playable(r):
                continue
            score = 2 * p / max_p + float(self.vecs[r] @ centroid)
            adds.append((r, score))
            if len(adds) >= n_swaps * 2:
                break

        swaps, used_adds = [], set()
        for cut in cuts:
            for add, _s in adds:
                if add in used_adds:
                    continue
                used_adds.add(add)
                swaps.append((cut, add, min(nonland[cut], 4)))
                break
            if len(swaps) >= n_swaps:
                break
        # a second add for the weakest cut, if budget remains
        for add, _s in adds:
            if len(swaps) >= n_swaps:
                break
            if add not in used_adds and cuts:
                used_adds.add(add)
                swaps.append((cuts[0], add, min(nonland[cuts[0]], 4)))
        return swaps, presence

    # ---------------- gauntlet ----------------

    def gauntlet(self, size: int):
        """Diverse, Forge-playable, winning corpus decks (farthest-point)."""
        eligible = [i for i, d in enumerate(self.corpus)
                    if d["wins"] >= 4
                    and all(norm(n) in self.supported for n, _q in d["main"])]
        if not eligible:
            eligible = [i for i, d in enumerate(self.corpus)
                        if all(norm(n) in self.supported for n, _q in d["main"])]
        chosen = [eligible[0]]
        while len(chosen) < min(size, len(eligible)):
            best, best_d = None, -1
            for i in eligible:
                if i in chosen:
                    continue
                d = min(1 - float(self.centroids[i] @ self.centroids[j])
                        for j in chosen)
                if d > best_d:
                    best, best_d = i, d
            chosen.append(best)
        return chosen


# ---------------- Forge runner ----------------

def write_dck(name: str, main_pairs):
    FORGE_DECKS.mkdir(parents=True, exist_ok=True)
    lines = ["[metadata]", f"Name={name}", "[Main]"]
    lines += [f"{q} {n}" for n, q in main_pairs]
    lines += ["[Sideboard]"]
    (FORGE_DECKS / f"{name}.dck").write_text("\n".join(lines))


def run_match(deck_a: str, deck_b: str, games: int, timeout_s: int):
    """Returns (wins_a, games_completed)."""
    cmd = ["xvfb-run", "-a", "java", "-Xmx3g",
           "-Dio.netty.tryReflectionSetAccessible=true",
           "-Dfile.encoding=UTF-8", "-jar",
           str(FORGE_DIR / "forge-gui-desktop-2.0.14-jar-with-dependencies.jar"),
           "sim", "-d", f"{deck_a}.dck", f"{deck_b}.dck", "-n", str(games), "-q"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout_s, cwd=FORGE_DIR)
        out = proc.stdout + proc.stderr
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or b"").decode("utf-8", "replace") if isinstance(exc.stdout, bytes) \
            else (exc.stdout or "")
    wins_a = len(re.findall(rf"Ai\(1\)-{re.escape(deck_a)} has won", out))
    done = len(re.findall(r"Game Result", out))
    return wins_a, done


def evaluate(variants, gauntlet_names, games_each, workers=2):
    """variants: list of deck names. Returns {name: (wins, games)}."""
    jobs = [(v, g) for v in variants for g in gauntlet_names]
    results = defaultdict(lambda: [0, 0])
    timeout_s = 90 + 25 * games_each

    def one(job):
        v, g = job
        w, n = run_match(v, g, games_each, timeout_s)
        return v, w, n

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for v, w, n in pool.map(one, jobs):
            results[v][0] += w
            results[v][1] += n
            done = sum(1 for _v in results.values())
            print(f"  {v}: +{w}/{n}", flush=True)
    return {v: tuple(x) for v, x in results.items()}


def ci95(wins, games):
    if games == 0:
        return 0.0, 0.0
    p = wins / games
    half = 1.96 * math.sqrt(p * (1 - p) / games)
    return p, half


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("deck", help="decklist file (.txt/.dck style lines)")
    ap.add_argument("--format", default="pauper")
    ap.add_argument("--candidates", type=int, default=6)
    ap.add_argument("--gauntlet", type=int, default=3)
    ap.add_argument("--stage1", type=int, default=6, help="games per gauntlet deck")
    ap.add_argument("--stage2", type=int, default=14)
    ap.add_argument("--finalists", type=int, default=3)
    ap.add_argument("--lock", action="append", default=[],
                    help="card that must not be cut (repeatable)")
    args = ap.parse_args()

    imp = Improver(args.format)
    text = Path(args.deck).read_text(encoding="utf-8")
    coll = parse_collection(text)
    deck, unmatched, unsupported = {}, [], []
    deck_pairs = []
    for name, qty in coll.items():
        row = imp.name_to_row.get(norm(name))
        if row is None:
            unmatched.append(name)
            continue
        deck[row] = deck.get(row, 0) + qty
        deck_pairs.append((imp.meta[row]["name"], qty))
        if not imp.playable(row):
            unsupported.append(name)
    total = sum(deck.values())
    print(f"deck: {total} cards, {len(deck)} distinct"
          + (f"; UNMATCHED: {unmatched}" if unmatched else "")
          + (f"; NOT FORGE-PLAYABLE: {unsupported}" if unsupported else ""))
    if unsupported:
        print("warning: unsupported cards will break simulation - aborting")
        return

    locked = {norm(n) for n in args.lock}
    swaps, presence = imp.propose(deck, args.candidates, locked)
    print("\nproposed swaps:")
    for cut, add, qty in swaps:
        print(f"  -{qty} {imp.meta[cut]['name']:<28} +{qty} {imp.meta[add]['name']}")

    gauntlet_idx = imp.gauntlet(args.gauntlet)
    gnames = []
    for k, gi in enumerate(gauntlet_idx):
        gname = f"gauntlet_{k}"
        write_dck(gname, imp.corpus[gi]["main"])
        top = sorted(imp.corpus[gi]["rows"].items(),
                     key=lambda rq: -rq[1])[:3]
        label = ", ".join(imp.meta[r]["name"] for r, _q in top
                          if not imp.is_land[r])
        print(f"gauntlet {k}: {label} ({imp.corpus[gi]['event'].rsplit('/',1)[-1]})")
        gnames.append(gname)

    write_dck("base", deck_pairs)
    variant_swaps = {"base": None}
    for i, (cut, add, qty) in enumerate(swaps):
        pairs = [(n, q) for n, q in deck_pairs]
        newpairs = []
        for n, q in pairs:
            if norm(n) == norm(imp.meta[cut]["name"]):
                if q > qty:
                    newpairs.append((n, q - qty))
            else:
                newpairs.append((n, q))
        newpairs.append((imp.meta[add]["name"], qty))
        vname = f"cand_{i}"
        write_dck(vname, newpairs)
        variant_swaps[vname] = (cut, add, qty)

    print(f"\nstage 1: {len(variant_swaps)} variants x {len(gnames)} gauntlet "
          f"x {args.stage1} games")
    s1 = evaluate(list(variant_swaps), gnames, args.stage1)
    ranked = sorted(s1.items(), key=lambda kv: -(kv[1][0] / max(kv[1][1], 1)))
    print("\nstage 1 results:")
    for v, (w, n) in ranked:
        print(f"  {v:<8} {w}/{n}  ({w/max(n,1):.0%})")

    finalists = [v for v, _ in ranked if v != "base"][:args.finalists]
    stage2_set = ["base"] + finalists
    print(f"\nstage 2: {len(stage2_set)} variants x {len(gnames)} x {args.stage2} games")
    s2 = evaluate(stage2_set, gnames, args.stage2)

    # combine stages for the final estimate
    report = []
    for v in stage2_set:
        w = s1[v][0] + s2[v][0]
        n = s1[v][1] + s2[v][1]
        p, half = ci95(w, n)
        entry = {"variant": v, "wins": w, "games": n,
                 "winrate": round(p, 3), "ci95": round(half, 3)}
        if variant_swaps[v]:
            cut, add, qty = variant_swaps[v]
            entry["swap"] = f"-{qty} {imp.meta[cut]['name']} +{qty} {imp.meta[add]['name']}"
        report.append(entry)

    base_p = next(e["winrate"] for e in report if e["variant"] == "base")
    print("\n=== FINAL ===")
    for e in sorted(report, key=lambda e: -e["winrate"]):
        delta = e["winrate"] - base_p
        swap = e.get("swap", "(baseline)")
        print(f"  {e['winrate']:.1%} ±{e['ci95']:.1%}  ({delta:+.1%})  {swap}"
              f"   [{e['wins']}/{e['games']}]")

    out = OUT / f"improve-report-{Path(args.deck).stem}.json"
    out.write_text(json.dumps({"format": args.format, "deck": args.deck,
                               "results": report}, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
