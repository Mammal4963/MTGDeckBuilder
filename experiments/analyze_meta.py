"""Phase 2 analysis: overlay real tournament decks on the card geometry.

1. Match every MTGO deck's cards to embedding rows (front-face aware).
2. Tightness test: is a real deck geometrically compact in the full
   embedding space, compared to (a) uniform random cards and (b) random
   cards drawn from the same format-legal pool? Reported as z-scores.
3. Per-format card play frequencies (for the map's meta-heat overlay).
4. Deck vectors (qty-weighted mean of nonland card embeddings) -> UMAP
   -> a deck-level map colored by format.

Reads  data/decks/mtgo-decks.jsonl.gz, experiments/output/{embeddings.npy,cards_meta.json}
Writes experiments/output/meta.json      (map overlay data + stats)
       experiments/output/deck_coords.npy, decks_meta.json
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
DECKS = ROOT / "data" / "decks" / "mtgo-decks.jsonl.gz"
SNAPSHOT = ROOT / "data" / "scryfall-oracle-cards-2026-08-12.jsonl.gz"

RNG = np.random.default_rng(7)
N_BASELINE = 200          # random sets per deck-size/format cell


def norm(name: str) -> str:
    return re.sub(r"\s+", " ", name).strip().lower()


def load_decks():
    decks = []
    with gzip.open(DECKS, "rt", encoding="utf-8") as fh:
        for line in fh:
            decks.append(json.loads(line))
    return decks


def main() -> None:
    emb = np.load(OUT / "embeddings.npy")
    meta = json.loads((OUT / "cards_meta.json").read_text())
    name_to_row = {}
    for i, m in enumerate(meta):
        name_to_row.setdefault(norm(m["name"]), i)
        if " // " in m["name"]:
            name_to_row.setdefault(norm(m["name"].split(" // ")[0]), i)

    db = CardDatabase.load(SNAPSHOT)
    is_land = np.array([("Land" in m["type_line"]) for m in meta])

    decks = load_decks()
    print(f"{len(decks)} decks loaded")

    # ---- match cards to rows -------------------------------------------
    matched_decks = []
    missing = Counter()
    for deck in decks:
        rows, qtys = [], []
        for name, qty in deck["main"]:
            row = name_to_row.get(norm(name))
            if row is None:
                missing[name] += 1
                continue
            rows.append(row)
            qtys.append(qty)
        if len(rows) >= 8:
            matched_decks.append((deck, np.array(rows), np.array(qtys)))
    print(f"{len(matched_decks)} decks matched (>=8 distinct cards found)")
    if missing:
        print("top unmatched names:", missing.most_common(8))

    # ---- per-format legal pools & play frequency ------------------------
    formats = sorted({d["format"] for d, _r, _q in matched_decks})
    legal_pool = {}
    for fmt in formats:
        legal = np.array([
            db.get(m["name"]) is not None
            and db.get(m["name"]).legalities.get(fmt) == "legal"
            for m in meta
        ])
        legal_pool[fmt] = np.where(legal & ~is_land)[0]

    freq = {fmt: Counter() for fmt in formats}
    for deck, rows, qtys in matched_decks:
        for row in set(rows.tolist()):
            freq[deck["format"]][row] += 1

    # ---- tightness ------------------------------------------------------
    def mean_pairwise_cos_dist(rows: np.ndarray) -> float:
        vecs = emb[rows]
        sims = vecs @ vecs.T
        n = len(rows)
        upper = sims[np.triu_indices(n, k=1)]
        return float(1.0 - upper.mean())

    baseline_cache: dict = {}

    def baseline(pool: np.ndarray, size: int, key) -> tuple:
        if key not in baseline_cache:
            vals = [
                mean_pairwise_cos_dist(RNG.choice(pool, size=size, replace=False))
                for _ in range(N_BASELINE)
            ]
            baseline_cache[key] = (float(np.mean(vals)), float(np.std(vals)))
        return baseline_cache[key]

    all_rows = np.where(~is_land)[0]
    stats = defaultdict(list)
    for deck, rows, qtys in matched_decks:
        nonland = rows[~is_land[rows]]
        distinct = np.unique(nonland)
        if len(distinct) < 6:
            continue
        d_deck = mean_pairwise_cos_dist(distinct)
        size_bucket = min(30, max(6, int(round(len(distinct) / 3.0)) * 3))
        mu_u, sd_u = baseline(all_rows, size_bucket, ("uniform", size_bucket))
        mu_f, sd_f = baseline(
            legal_pool[deck["format"]], size_bucket, (deck["format"], size_bucket)
        )
        stats[deck["format"]].append(
            (
                d_deck,
                (d_deck - mu_u) / sd_u if sd_u else 0.0,
                (d_deck - mu_f) / sd_f if sd_f else 0.0,
            )
        )

    print("\nTightness (mean pairwise cosine distance of distinct nonland cards)")
    print(f"{'format':<10} {'decks':>6} {'deck dist':>10} {'z vs any':>9} {'z vs legal':>11}")
    summary = {}
    for fmt in formats:
        arr = np.array(stats[fmt])
        if not len(arr):
            continue
        summary[fmt] = {
            "decks": int(len(arr)),
            "mean_dist": round(float(arr[:, 0].mean()), 4),
            "z_uniform": round(float(arr[:, 1].mean()), 2),
            "z_legal": round(float(arr[:, 2].mean()), 2),
        }
        print(f"{fmt:<10} {len(arr):>6} {arr[:,0].mean():>10.4f} "
              f"{arr[:,1].mean():>9.2f} {arr[:,2].mean():>11.2f}")

    # ---- deck vectors + UMAP -------------------------------------------
    import umap

    deck_vecs, decks_meta = [], []
    for deck, rows, qtys in matched_decks:
        keep = ~is_land[rows]
        if keep.sum() < 6:
            continue
        weights = qtys[keep].astype(np.float64)
        vec = (emb[rows[keep]] * weights[:, None]).sum(0) / weights.sum()
        deck_vecs.append(vec.astype(np.float32))
        top = rows[keep][np.argsort(-qtys[keep])][:4]
        decks_meta.append({
            "format": deck["format"],
            "wins": deck.get("wins", 0),
            "player": deck["player"],
            "event": deck["event"].rsplit("/", 1)[-1],
            "date": deck["date"],
            "top_cards": [meta[r]["name"] for r in top],
            "cards": [[int(r), int(q)] for r, q in zip(rows.tolist(), qtys.tolist())],
        })
    deck_vecs = np.vstack(deck_vecs)
    deck_vecs /= np.linalg.norm(deck_vecs, axis=1, keepdims=True)
    print(f"\n{len(deck_vecs)} deck vectors; running UMAP…")
    coords = umap.UMAP(
        n_neighbors=20, min_dist=0.1, metric="cosine", random_state=42
    ).fit_transform(deck_vecs).astype(np.float32)

    np.save(OUT / "deck_coords.npy", coords)
    (OUT / "decks_meta.json").write_text(json.dumps(decks_meta), encoding="utf-8")

    # ---- overlay bundle for the card map -------------------------------
    heat = {
        fmt: sorted(
            ((int(row), int(count)) for row, count in freq[fmt].items()),
            key=lambda rc: -rc[1],
        )
        for fmt in formats
    }
    overlay = {"formats": formats, "heat": heat, "tightness": summary}
    (OUT / "meta.json").write_text(
        json.dumps(overlay, separators=(",", ":")), encoding="utf-8"
    )
    print("wrote meta.json, deck_coords.npy, decks_meta.json")


if __name__ == "__main__":
    main()
