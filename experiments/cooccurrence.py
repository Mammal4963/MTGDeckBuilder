"""Phase 3: co-occurrence spaces - what cards are *played with*.

Two separate synergy spaces, per the project owner's design:

* **60**  - MTGO tournament decks + Archidekt casual 60-card decks
            (standard/modern/legacy/pauper/historic community lists).
* **cmd** - Archidekt Commander decks only.

For each space: PPMI over card pairs (equal pair-mass per deck), SVD to
64 dims, ridge projection from text-embedding space so every card gets a
vector, and a mined "synergy disagreement" list (high co-occurrence, low
text similarity). The 60 space is evaluated by held-out reconstruction
on the same MTGO test split as earlier phases.

Writes output/covectors-{60,cmd}.npy, covocab-{60,cmd}.json,
       synergy_pairs-{60,cmd}.json
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
MIN_DECKS = 3
TEST_FRAC = 0.1
RNG = np.random.default_rng(11)


def norm(name: str) -> str:
    return re.sub(r"\s+", " ", name).strip().lower()


def build_space(train, emb, meta, label):
    """PPMI -> SVD -> ridge-blended vectors for one corpus. Returns
    (blended vectors, vocab rows, deck_freq, fmt_freq, pairs_list)."""
    deck_freq = Counter()
    fmt_freq: dict = defaultdict(Counter)
    for fmt, rows in train:
        for r in rows:
            deck_freq[r] += 1
            fmt_freq[fmt][r] += 1
    vocab = sorted(r for r, c in deck_freq.items() if c >= MIN_DECKS)
    v_index = {r: i for i, r in enumerate(vocab)}
    NV = len(vocab)

    pair_w: dict = defaultdict(float)
    pair_decks = Counter()
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

    n_pairs = sum(pair_w.values())
    totals = defaultdict(float)
    for key, c in pair_w.items():
        totals[key // NV] += c
        totals[key % NV] += c
    rows_ix, cols_ix, vals = [], [], []
    for key, c in pair_w.items():
        i, j = key // NV, key % NV
        pmi = np.log(c * n_pairs / (totals[i] * totals[j]))
        if pmi > 0:
            rows_ix += [i, j]
            cols_ix += [j, i]
            vals += [pmi, pmi]
    M = sparse.csr_matrix((vals, (rows_ix, cols_ix)), shape=(NV, NV))

    from sklearn.decomposition import TruncatedSVD
    from sklearn.linear_model import Ridge

    svd = TruncatedSVD(n_components=min(DIM, NV - 1), random_state=0)
    V = svd.fit_transform(M)
    V /= np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)
    print(f"[{label}] {len(train)} decks, vocab {NV}, nnz {M.nnz}, "
          f"SVD var {svd.explained_variance_ratio_.sum():.2f}")

    vocab_rows = np.array(vocab)
    ridge = Ridge(alpha=1.0)
    ridge.fit(emb[vocab_rows], V)
    predicted = ridge.predict(emb)
    predicted /= np.maximum(np.linalg.norm(predicted, axis=1, keepdims=True), 1e-9)
    blended = predicted.copy()
    blended[vocab_rows] = V
    blended = blended.astype(np.float32)

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
    return blended, vocab, deck_freq, fmt_freq, pairs_list


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

    # MTGO tournament decks; split kept identical to earlier phases.
    deck_rows = []
    with gzip.open(ROOT / "data" / "decks" / "mtgo-decks.jsonl.gz", "rt") as fh:
        for line in fh:
            deck = json.loads(line)
            rows = to_rows(deck["main"])
            if len(rows) >= 8:
                deck_rows.append((deck["format"], rows))
    order = RNG.permutation(len(deck_rows))
    test_idx = set(order[: int(len(deck_rows) * TEST_FRAC)].tolist())
    train_60 = [deck_rows[i] for i in range(len(deck_rows)) if i not in test_idx]
    test = [deck_rows[i] for i in sorted(test_idx)]

    # Archidekt: casual 60-card formats join the 60 space; Commander gets
    # its own space entirely - the corpora never mix.
    train_cmd = []
    casual = ROOT / "data" / "decks" / "archidekt-decks.jsonl.gz"
    n_c60 = 0
    if casual.exists():
        with gzip.open(casual, "rt") as fh:
            for line in fh:
                deck = json.loads(line)
                rows = to_rows(deck["cards"])
                if deck["format"] == "commander":
                    if len(rows) >= 30:
                        train_cmd.append((deck["format"], rows))
                elif len(rows) >= 15:
                    train_60.append((deck["format"], rows))
                    n_c60 += 1
    print(f"60 space: {len(train_60)} train ({n_c60} casual) / {len(test)} test")
    print(f"cmd space: {len(train_cmd)} commander decks")

    results = {}
    for label, train in (("60", train_60), ("cmd", train_cmd)):
        if not train:
            print(f"[{label}] no decks - skipped")
            continue
        blended, vocab, deck_freq, fmt_freq, pairs = build_space(
            train, emb, meta, label)
        np.save(OUT / f"covectors-{label}.npy", blended)
        (OUT / f"synergy_pairs-{label}.json").write_text(
            json.dumps(pairs[:500], indent=1), encoding="utf-8")
        (OUT / f"covocab-{label}.json").write_text(json.dumps({
            "vocab_rows": [int(r) for r in vocab],
            "deck_freq": {str(r): int(c) for r, c in deck_freq.items()},
            "fmt_freq": {f: {str(r): int(c) for r, c in cs.items()}
                         for f, cs in fmt_freq.items()},
        }), encoding="utf-8")
        results[label] = (blended, deck_freq, fmt_freq)
        print(f"[{label}] top synergy pairs:")
        for p in pairs[:8]:
            print(f"   {p['synergy']:+.3f}  {p['a']}  +  {p['b']}  ({p['decks']} decks)")

    # ---- evaluation (60 space only; comparable to earlier phases) ------
    blended, _freq, fmt_freq = results["60"]
    db = CardDatabase.load(ROOT / "data" / "scryfall-oracle-cards-2026-08-12.jsonl.gz")
    mtgo_fmts = {f for f, _ in test}
    legal_rows = {}
    for fmt in mtgo_fmts:
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
              f"hit@10 {np.mean(ranks<=10):.1%}   hit@50 {np.mean(ranks<=50):.1%}")

    print("\nHeld-out MTGO reconstruction (60 space):")
    evaluate(blended, "co-occurrence (60)")
    evaluate(emb, "text embeddings")

    # Qualitative: the casual-archetype gap.
    row = name_to_row.get("blood artist")
    if row is not None:
        for label, (vectors, freq, _f) in results.items():
            sims = vectors @ vectors[row]
            names = ", ".join(meta[j]["name"] for j in np.argsort(-sims)[1:9])
            print(f"\nBlood Artist [{label}] (in {freq.get(row, 0)} decks): {names}")


if __name__ == "__main__":
    main()
