"""Phase 3: the co-occurrence space - what cards are *played with*.

1. Build PPMI card-pair statistics over the MTGO deck corpus (train
   split), factor with truncated SVD -> co-occurrence vectors for every
   card seen in enough decks (the "vocab").
2. Generalize: ridge-regress text embeddings -> co-occurrence vectors,
   so every card in the Scryfall snapshot gets a predicted synergy
   vector even if no tournament deck ever played it. Vocab cards keep
   their grounded vectors; everything else uses the projection.
3. Mine the disagreement list: pairs with high co-occurrence similarity
   but low text similarity - true synergy pairs no text model sees.
4. Evaluate the user-facing task: on held-out decks, hide a card and
   rank it among all format-legal candidates given the rest of the deck.
   Compare against text-only and popularity baselines.

Reads  data/decks/mtgo-decks.jsonl.gz, output/{embeddings.npy,cards_meta.json}
Writes output/covectors.npy       blended [n_cards, 64] unit vectors
       output/covocab.json        vocab rows + per-format play counts
       output/synergy_pairs.json  top disagreement pairs
"""
from __future__ import annotations

import gzip
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mtg_deckbuilder.carddata import CardDatabase  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "output"
DIM = 64
MIN_DECKS = 3          # vocab threshold
TEST_FRAC = 0.1
RNG = np.random.default_rng(11)


def norm(name: str) -> str:
    return re.sub(r"\s+", " ", name).strip().lower()


def main() -> None:
    emb = np.load(OUT / "embeddings.npy")
    meta = json.loads((OUT / "cards_meta.json").read_text())
    name_to_row = {}
    for i, m in enumerate(meta):
        name_to_row.setdefault(norm(m["name"]), i)
        if " // " in m["name"]:
            name_to_row.setdefault(norm(m["name"].split(" // ")[0]), i)
    is_land = np.array([("Land" in m["type_line"]) for m in meta])

    def to_rows(pairs):
        rows = {name_to_row[norm(n)] for n, _q in pairs if norm(n) in name_to_row}
        return sorted(r for r in rows if not is_land[r])

    # MTGO tournament decks (train/test split kept identical to phase 3
    # so results stay comparable).
    deck_rows = []
    with gzip.open(ROOT / "data" / "decks" / "mtgo-decks.jsonl.gz", "rt") as fh:
        for line in fh:
            deck = json.loads(line)
            rows = to_rows(deck["main"])
            if len(rows) >= 8:
                deck_rows.append((deck["format"], rows))

    order = RNG.permutation(len(deck_rows))
    n_test = int(len(deck_rows) * TEST_FRAC)
    test_idx = set(order[:n_test].tolist())
    train = [deck_rows[i] for i in range(len(deck_rows)) if i not in test_idx]
    test = [deck_rows[i] for i in sorted(test_idx)]

    # Casual corpus (Archidekt), train-only: teaches the archetypes
    # tournament play never shows (aristocrats, tribal, lifegain...).
    casual_path = ROOT / "data" / "decks" / "archidekt-decks.jsonl.gz"
    n_casual = 0
    if casual_path.exists():
        with gzip.open(casual_path, "rt") as fh:
            for line in fh:
                deck = json.loads(line)
                rows = to_rows(deck["cards"])
                if len(rows) >= 15:
                    train.append((deck["format"], rows))
                    n_casual += 1
    print(f"{len(train)} train ({n_casual} casual) / {len(test)} test decks")

    # ---- vocab & pair counts (train only) ------------------------------
    deck_freq = Counter()
    fmt_freq: dict = defaultdict(Counter)
    for fmt, rows in train:
        for r in rows:
            deck_freq[r] += 1
            fmt_freq[fmt][r] += 1
    vocab = sorted(r for r, c in deck_freq.items() if c >= MIN_DECKS)
    v_index = {r: i for i, r in enumerate(vocab)}
    print(f"vocab: {len(vocab)} cards seen in >= {MIN_DECKS} train decks")

    # Weighted pairs: every deck contributes roughly equal total pair
    # mass, so a 100-card Commander list doesn't out-vote a 60-card deck
    # (a 60-card deck has ~300 pairs; Commander decks ~2000).
    NV = len(vocab)
    pair_w: dict = defaultdict(float)     # weighted mass (for PPMI)
    pair_decks = Counter()                # integer deck counts (for display)
    for _fmt, rows in train:
        vrows = [v_index[r] for r in rows if r in v_index]
        n = len(vrows)
        if n < 2:
            continue
        w = min(1.0, 350.0 / (n * (n - 1) / 2))
        for a in range(n):
            for b in range(a + 1, n):
                i, j = (vrows[a], vrows[b]) if vrows[a] < vrows[b] else (vrows[b], vrows[a])
                key = i * NV + j
                pair_w[key] += w
                pair_decks[key] += 1

    # ---- PPMI + SVD ----------------------------------------------------
    n_pairs = sum(pair_w.values())
    card_pair_totals = defaultdict(float)
    for key, c in pair_w.items():
        card_pair_totals[key // NV] += c
        card_pair_totals[key % NV] += c
    rows_ix, cols_ix, vals = [], [], []
    for key, c in pair_w.items():
        i, j = key // NV, key % NV
        pmi = np.log(c * n_pairs / (card_pair_totals[i] * card_pair_totals[j]))
        if pmi > 0:
            rows_ix += [i, j]
            cols_ix += [j, i]
            vals += [pmi, pmi]
    M = sparse.csr_matrix(
        (vals, (rows_ix, cols_ix)), shape=(len(vocab), len(vocab))
    )
    from sklearn.decomposition import TruncatedSVD

    svd = TruncatedSVD(n_components=DIM, random_state=0)
    V = svd.fit_transform(M)
    V /= np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)
    print(f"PPMI matrix nnz={M.nnz}, SVD explained variance "
          f"{svd.explained_variance_ratio_.sum():.2f}")

    # ---- projection: text space -> co-occurrence space -----------------
    from sklearn.linear_model import Ridge

    vocab_rows = np.array(vocab)
    ridge = Ridge(alpha=1.0)
    ridge.fit(emb[vocab_rows], V)
    ridge_pred = ridge.predict(emb)
    ridge_pred /= np.maximum(np.linalg.norm(ridge_pred, axis=1, keepdims=True), 1e-9)

    # kNN transfer: an unplayed card borrows the grounded co-vectors of its
    # closest *played* text-neighbors (sharpened weights). This keeps casual
    # staples like Blood Artist attached to real archetype geometry instead
    # of a smoothed regression that echoes text similarity.
    K = 8
    knn_pred = np.zeros_like(ridge_pred)
    vocab_emb = emb[vocab_rows]
    for start in range(0, len(emb), 4096):
        chunk = emb[start:start + 4096]
        sims = chunk @ vocab_emb.T
        top = np.argpartition(-sims, K, axis=1)[:, :K]
        for local, i in enumerate(range(start, start + len(chunk))):
            w = np.maximum(sims[local, top[local]], 0.0) ** 4
            if w.sum() <= 0:
                continue
            knn_pred[i] = (V[top[local]] * (w / w.sum())[:, None]).sum(0)
    knn_pred /= np.maximum(np.linalg.norm(knn_pred, axis=1, keepdims=True), 1e-9)

    # Projection sanity on vocab (leave-self-out isn't exact here, but the
    # ridge comparison is like-for-like).
    holdout = RNG.choice(len(vocab), size=min(300, len(vocab)), replace=False)
    cos_r = (ridge_pred[vocab_rows[holdout]] * V[holdout]).sum(1)
    print(f"projection: ridge mean cos = {cos_r.mean():.3f}")

    # Ridge alone evaluates best on held-out reconstruction (kNN transfer
    # was tried at 50/50 and scored slightly worse; both fail the same way
    # for archetypes the corpus simply never plays - that needs more data,
    # not a different projection).
    predicted = ridge_pred
    blended = predicted.copy()
    blended[vocab_rows] = V            # grounded vectors win where they exist
    blended = blended.astype(np.float32)
    np.save(OUT / "covectors.npy", blended)

    # ---- synergy disagreement list -------------------------------------
    pairs_list = []
    for key, c in pair_decks.items():
        if c < 8:
            continue
        i, j = key // NV, key % NV
        co_sim = float(V[i] @ V[j])
        text_sim = float(emb[vocab[i]] @ emb[vocab[j]])
        pairs_list.append({
            "a": meta[vocab[i]]["name"], "b": meta[vocab[j]]["name"],
            "decks": c, "co_sim": round(co_sim, 3),
            "text_sim": round(text_sim, 3),
            "synergy": round(co_sim - text_sim, 3),
        })
    pairs_list.sort(key=lambda p: -p["synergy"])
    (OUT / "synergy_pairs.json").write_text(
        json.dumps(pairs_list[:500], indent=1), encoding="utf-8")
    print("\nTop synergy pairs (played together, textually unalike):")
    for p in pairs_list[:15]:
        print(f"  {p['synergy']:+.3f}  {p['a']}  +  {p['b']}   "
              f"({p['decks']} decks, text {p['text_sim']:.2f})")

    # ---- evaluation: held-out deck reconstruction ----------------------
    db = CardDatabase.load(ROOT / "data" / "scryfall-oracle-cards-2026-08-12.jsonl.gz")
    legal_rows = {}
    for fmt in fmt_freq:
        mask = np.array([
            (card := db.get(m["name"])) is not None
            and card.legalities.get(fmt) in ("legal", "restricted")
            for m in meta
        ])
        legal_rows[fmt] = np.where(mask & ~is_land)[0]

    def evaluate(vectors, label):
        ranks = []
        for fmt, rows in test:
            candidates = legal_rows[fmt]
            pos = {r: k for k, r in enumerate(candidates)}
            usable = [r for r in rows if r in pos]
            if len(usable) < 10:
                continue
            held = RNG.choice(usable, size=min(5, len(usable) // 2), replace=False)
            for h in held:
                context = [r for r in usable if r != h]
                ctx = vectors[context].mean(0)
                scores = vectors[candidates] @ ctx
                rank = int((scores > scores[pos[h]]).sum()) + 1
                ranks.append(rank)
        ranks = np.array(ranks)
        print(f"  {label:<22} median rank {np.median(ranks):>6.0f}   "
              f"hit@10 {np.mean(ranks<=10):.1%}   hit@50 {np.mean(ranks<=50):.1%}"
              f"   (n={len(ranks)}, pool ~{np.mean([len(legal_rows[f]) for f,_ in test]):.0f})")
        return ranks

    print("\nHeld-out reconstruction (hide a card, rank it among all legal cards):")
    evaluate(blended, "co-occurrence (ours)")
    evaluate(emb, "text embeddings")
    # Popularity baseline: rank by train play count in format.
    ranks = []
    for fmt, rows in test:
        candidates = legal_rows[fmt]
        pop = np.array([fmt_freq[fmt].get(r, 0) for r in candidates], dtype=float)
        pos = {r: k for k, r in enumerate(candidates)}
        usable = [r for r in rows if r in pos]
        if len(usable) < 10:
            continue
        held = RNG.choice(usable, size=min(5, len(usable) // 2), replace=False)
        for h in held:
            rank = int((pop > pop[pos[h]]).sum()) + 1
            ranks.append(rank)
    ranks = np.array(ranks)
    print(f"  {'popularity':<22} median rank {np.median(ranks):>6.0f}   "
          f"hit@10 {np.mean(ranks<=10):.1%}   hit@50 {np.mean(ranks<=50):.1%}")

    # ---- bundle for the seeker page ------------------------------------
    covocab = {
        "vocab_rows": [int(r) for r in vocab],
        "deck_freq": {str(r): int(c) for r, c in deck_freq.items()},
        "fmt_freq": {fmt: {str(r): int(c) for r, c in counts.items()}
                     for fmt, counts in fmt_freq.items()},
    }
    (OUT / "covocab.json").write_text(json.dumps(covocab), encoding="utf-8")
    print("\nwrote covectors.npy, covocab.json, synergy_pairs.json")

    # Qualitative check for the casual-archetype gap: what does the model
    # now recommend next to Blood Artist?
    row = name_to_row.get("blood artist")
    if row is not None:
        sims = blended @ blended[row]
        print(f"\nBlood Artist neighbors (in {deck_freq.get(row, 0)} train decks):")
        for j in np.argsort(-sims)[1:11]:
            print(f"  {sims[j]:.3f}  {meta[j]['name']}")


if __name__ == "__main__":
    main()
