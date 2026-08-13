"""Phase 4: generate a complete 60-card deck from seed cards.

Method - "the neighbors are the recipe":

1. Embed the seed picks as a centroid in the 60-card co-occurrence space.
2. Retrieve the K most similar real decks (tournament + casual).
3. Learn the recipe from those neighbors: nonland count, land count,
   mana-curve histogram, per-card typical copy counts, and which cards
   they actually play (presence, similarity-weighted).
4. Greedy fill: score = neighbor presence + synergy to seed, minus a
   curve penalty against the learned histogram; take the typical number
   of copies. Lands: the neighbors' nonbasic lands by presence, then
   basics split by the deck's colored pips.

Evaluated by rebuilding held-out MTGO decks from 3 random seed cards:
recall of the actual deck's nonland cards, vs popularity and
no-recipe (pure synergy) baselines.
"""
from __future__ import annotations

import gzip
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mtg_deckbuilder.carddata import CardDatabase  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "output"
RNG = np.random.default_rng(11)
K_NEIGHBORS = 25
DECK_SIZE = 60

BASIC_FOR_COLOR = {"W": "Plains", "U": "Island", "B": "Swamp",
                   "R": "Mountain", "G": "Forest"}
_PIP_RE = re.compile(r"\{([WUBRG])(?:/[WUBRGP])?\}")


def norm(name: str) -> str:
    return re.sub(r"\s+", " ", name).strip().lower()


class Generator:
    def __init__(self):
        self.vecs = np.load(OUT / "covectors-60.npy")
        self.meta = json.loads((OUT / "cards_meta.json").read_text())
        self.name_to_row = {}
        for i, m in enumerate(self.meta):
            self.name_to_row.setdefault(norm(m["name"]), i)
            if " // " in m["name"]:
                self.name_to_row.setdefault(norm(m["name"].split(" // ")[0]), i)
        self.is_land = np.array([("Land" in m["type_line"]) for m in self.meta])
        self.is_basic = np.array([("Basic" in m["type_line"]) for m in self.meta])
        self.mv = np.array([m["mana_value"] for m in self.meta])
        self.db = CardDatabase.load(
            ROOT / "data" / "scryfall-oracle-cards-2026-08-12.jsonl.gz")

        # 60-card decks (tournament train split + casual), lands included.
        self.decks = []          # {fmt, cards: {row: qty}, lands: {row: qty}}
        self.test = []
        mtgo = []
        with gzip.open(ROOT / "data" / "decks" / "mtgo-decks.jsonl.gz", "rt") as fh:
            for line in fh:
                deck = json.loads(line)
                rec = self._to_record(deck["format"], deck["main"])
                if rec:
                    mtgo.append(rec)
        order = RNG.permutation(len(mtgo))
        test_idx = set(order[: int(len(mtgo) * 0.1)].tolist())
        for i, rec in enumerate(mtgo):
            (self.test if i in test_idx else self.decks).append(rec)
        with gzip.open(ROOT / "data" / "decks" / "archidekt-decks.jsonl.gz", "rt") as fh:
            for line in fh:
                deck = json.loads(line)
                if deck["format"] == "commander":
                    continue
                rec = self._to_record(deck["format"], deck["cards"])
                if rec:
                    self.decks.append(rec)

        self.centroids = np.zeros((len(self.decks), self.vecs.shape[1]), np.float32)
        for d, rec in enumerate(self.decks):
            rows = np.array(list(rec["cards"]))
            qtys = np.array([rec["cards"][r] for r in rows], float)
            v = (self.vecs[rows] * qtys[:, None]).sum(0) / qtys.sum()
            n = np.linalg.norm(v)
            self.centroids[d] = v / n if n > 0 else v
        print(f"{len(self.decks)} recipe decks, {len(self.test)} held-out test")

    def _to_record(self, fmt, pairs):
        cards, lands = {}, {}
        for name, qty in pairs:
            row = self.name_to_row.get(norm(name))
            if row is None:
                continue
            (lands if self.is_land[row] else cards)[row] = min(int(qty), 4) \
                if not self.is_basic[row] else int(qty)
        if len(cards) < 8:
            return None
        return {"fmt": fmt, "cards": cards, "lands": lands}

    def legal_mask(self, fmt):
        return np.array([
            (c := self.db.get(m["name"])) is not None
            and c.legalities.get(fmt) in ("legal", "restricted")
            for m in self.meta
        ])

    def generate(self, seed_rows, fmt=None, use_recipe=True):
        seed_vec = self.vecs[seed_rows].mean(0)
        seed_vec /= max(np.linalg.norm(seed_vec), 1e-9)
        sims = self.centroids @ seed_vec
        neighbors = np.argsort(-sims)[:K_NEIGHBORS]
        n_sims = sims[neighbors]

        # Recipe from neighbors.
        nonland_counts, land_counts = [], []
        curve = np.zeros(8)
        presence = defaultdict(float)
        qty_sum = defaultdict(float)
        qty_n = defaultdict(float)
        land_presence = defaultdict(float)
        for w, d in zip(n_sims, neighbors):
            rec = self.decks[d]
            nonland_counts.append(sum(rec["cards"].values()))
            land_counts.append(sum(rec["lands"].values()))
            for row, qty in rec["cards"].items():
                curve[min(7, int(self.mv[row]))] += w * qty
                presence[row] += w
                qty_sum[row] += w * qty
                qty_n[row] += w
            for row, qty in rec["lands"].items():
                if not self.is_basic[row]:
                    land_presence[row] += w
        nonland_target = int(np.median(nonland_counts))
        land_target = DECK_SIZE - nonland_target
        curve /= max(curve.sum(), 1e-9)

        legal = self.legal_mask(fmt) if fmt else np.ones(len(self.meta), bool)
        max_presence = max(presence.values()) if presence else 1.0

        picked = {}
        for r in seed_rows:
            picked[r] = 4 if not self.is_land[r] else 0
        cur_curve = np.zeros(8)
        total = sum(picked.values())
        for r, q in picked.items():
            cur_curve[min(7, int(self.mv[r]))] += q

        candidates = [r for r in presence if not self.is_land[r]
                      and legal[r] and r not in picked]
        syn = {r: float(self.vecs[r] @ seed_vec) for r in candidates}
        while total < nonland_target and candidates:
            best, best_score = None, -1e9
            for r in candidates:
                score = 2.0 * presence[r] / max_presence + syn[r]
                if use_recipe:
                    b = min(7, int(self.mv[r]))
                    over = (cur_curve[b] + 2) / max(total + 2, 1) - curve[b]
                    score -= 3.0 * max(0.0, over)
                if score > best_score:
                    best, best_score = r, score
            if best is None:
                break
            copies = int(round(qty_sum[best] / max(qty_n[best], 1e-9)))
            copies = max(1, min(4, copies, nonland_target - total))
            picked[best] = picked.get(best, 0) + copies
            cur_curve[min(7, int(self.mv[best]))] += copies
            total += copies
            candidates.remove(best)

        # Lands: neighbors' nonbasics by presence, then basics by pips.
        lands = {}
        land_total = 0
        for row, _w in sorted(land_presence.items(), key=lambda kv: -kv[1]):
            if land_total >= int(land_target * 0.55):
                break
            if fmt and not legal[row]:
                continue
            copies = min(4, land_target - land_total)
            lands[row] = copies
            land_total += copies
        pips = Counter()
        for r, q in picked.items():
            card = self.db.get(self.meta[r]["name"])
            if card:
                for pip in _PIP_RE.findall(card.mana_cost):
                    pips[pip] += q
        remaining = land_target - land_total
        if pips and remaining > 0:
            total_pips = sum(pips.values())
            for color, count in pips.most_common():
                n = round(remaining * count / total_pips)
                if n > 0:
                    row = self.name_to_row[norm(BASIC_FOR_COLOR[color])]
                    lands[row] = lands.get(row, 0) + n
        elif remaining > 0:
            lands[self.name_to_row["wastes"]] = remaining
        return picked, lands

    # ------------------------------------------------------------------

    def evaluate(self, n_decks=100):
        results = {"recipe": [], "no_recipe": [], "popularity": []}
        pop = Counter()
        for rec in self.decks:
            for row in rec["cards"]:
                pop[row] += 1
        pop_rows = [r for r, _ in pop.most_common()]

        sample = RNG.choice(len(self.test), size=min(n_decks, len(self.test)),
                            replace=False)
        for t in sample:
            rec = self.test[t]
            rows = list(rec["cards"])
            if len(rows) < 8:
                continue
            seeds = RNG.choice(rows, size=3, replace=False).tolist()
            actual = set(rows)
            target_n = len(actual)
            for label, kwargs in (("recipe", {"use_recipe": True}),
                                  ("no_recipe", {"use_recipe": False})):
                picked, _lands = self.generate(seeds, fmt=rec["fmt"], **kwargs)
                gen = set(picked)
                results[label].append(
                    len(gen & actual) / max(len(actual), 1))
            legal = self.legal_mask(rec["fmt"])
            popgen = set(seeds)
            for r in pop_rows:
                if len(popgen) >= len(set(picked)):
                    break
                if legal[r] and not self.is_land[r]:
                    popgen.add(r)
            results["popularity"].append(len(popgen & actual) / max(len(actual), 1))

        print(f"\nRebuilding held-out MTGO decks from 3 seed cards "
              f"(recall of the deck's distinct nonland cards):")
        for label, vals in results.items():
            vals = np.array(vals)
            print(f"  {label:<12} mean recall {vals.mean():.1%}   "
                  f"median {np.median(vals):.1%}")

    def show(self, seed_names, fmt=None):
        seeds = [self.name_to_row[norm(n)] for n in seed_names]
        picked, lands = self.generate(seeds, fmt=fmt)
        print(f"\n=== generated from {seed_names} ({fmt or 'any'}) ===")
        entries = sorted(picked.items(), key=lambda rq: (self.mv[rq[0]], -rq[1]))
        for row, qty in entries:
            m = self.meta[row]
            print(f"{qty} {m['name']:<34} {m['mana_cost']:<10} {m['type_line'][:34]}")
        print("-- lands --")
        for row, qty in sorted(lands.items(), key=lambda rq: -rq[1]):
            print(f"{qty} {self.meta[row]['name']}")
        total = sum(picked.values()) + sum(lands.values())
        print(f"total {total}")


if __name__ == "__main__":
    gen = Generator()
    gen.evaluate()
    gen.show(["Blood Artist", "Viscera Seer"], fmt=None)
    gen.show(["Guttersnipe"], fmt="pauper")
