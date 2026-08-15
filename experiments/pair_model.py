"""Learn synergy(A, B): do these two cards go well together?

The mathematical object the project has been circling: a trained
function over two cards' *rules text* that predicts whether they belong
in the same deck. Co-occurrence cosine (v0) can only score pairs people
have already played; this model reads the text embeddings, so it can
score pairs nobody has tried - which is also the engine for novel combo
mining (high predicted synergy, zero observed play).

Model: symmetric low-rank bilinear over frozen MiniLM embeddings
    s(a,b) = <A e_a, B e_b> + <A e_b, B e_a> + bias
(A, B: rank x 384). ~100k params, trains on CPU in minutes.

Data: "played together in a real deck" = positive; random in-corpus
pair = negative. Deck-level train/test split; showcase pairs are
excluded from training so their scores demonstrate generalization.

Writes output/pair_model.npz and prints AUCs + showcase percentiles.
"""
from __future__ import annotations

import gzip
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "output"
RNG = np.random.default_rng(31)
RANK = 128
MAX_POS = 600_000

SHOWCASE = [
    ("Blood Artist", "Viscera Seer"),
    ("Tainted Aether", "Hunted Horror"),
    ("Tainted Aether", "Forbidden Orchard"),
    ("Guttersnipe", "Lightning Bolt"),
    ("Aether Vial", "Tainted Aether"),
    ("Llanowar Elves", "Craterhoof Behemoth"),
    ("Counterspell", "Savannah Lions"),        # plausible anti-pair
    ("Healing Salve", "Goblin Guide"),         # nonsense pair
]


def norm(name: str) -> str:
    return re.sub(r"\s+", " ", name).strip().lower()


def load_decks(name_to_row, is_land):
    decks = []
    with gzip.open(ROOT / "data" / "decks" / "mtgo-decks.jsonl.gz", "rt") as fh:
        for line in fh:
            d = json.loads(line)
            rows = sorted({name_to_row[norm(n)] for n, _q in d["main"]
                           if norm(n) in name_to_row})
            rows = [r for r in rows if not is_land[r]]
            if len(rows) >= 8:
                decks.append(rows)
    with gzip.open(ROOT / "data" / "decks" / "archidekt-decks.jsonl.gz", "rt") as fh:
        for line in fh:
            d = json.loads(line)
            rows = sorted({name_to_row[norm(n)] for n, _q in d["cards"]
                           if norm(n) in name_to_row})
            rows = [r for r in rows if not is_land[r]]
            if len(rows) >= 15:
                decks.append(rows)
    return decks


def sample_pairs(decks, per_deck=60):
    pairs = set()
    for rows in decks:
        n = len(rows)
        total = n * (n - 1) // 2
        want = min(per_deck, total)
        seen = set()
        while len(seen) < want:
            i, j = RNG.integers(0, n, 2)
            if i == j:
                continue
            a, b = (rows[i], rows[j]) if rows[i] < rows[j] else (rows[j], rows[i])
            seen.add((a, b))
        pairs.update(seen)
        if len(pairs) >= MAX_POS:
            break
    return pairs


def auc(pos_scores, neg_scores):
    labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    scores = np.concatenate([pos_scores, neg_scores])
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos, n_neg = len(pos_scores), len(neg_scores)
    return (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main():
    emb = np.load(OUT / "embeddings.npy").astype(np.float32)
    meta = json.loads((OUT / "cards_meta.json").read_text())
    name_to_row = {}
    for i, m in enumerate(meta):
        name_to_row.setdefault(norm(m["name"]), i)
    is_land = np.array([("Land" in m["type_line"]) for m in meta])

    decks = load_decks(name_to_row, is_land)
    print(f"{len(decks)} decks for pair training")
    order = RNG.permutation(len(decks))
    n_test = int(len(decks) * 0.1)
    test_decks = [decks[i] for i in order[:n_test]]
    train_decks = [decks[i] for i in order[n_test:]]

    showcase_rows = {tuple(sorted((name_to_row[norm(a)], name_to_row[norm(b)])))
                     for a, b in SHOWCASE
                     if norm(a) in name_to_row and norm(b) in name_to_row}

    train_pos = sample_pairs(train_decks) - showcase_rows
    card_pool = np.array(sorted({r for rows in decks for r in rows}))
    print(f"{len(train_pos)} train positives, pool {len(card_pool)} cards")

    def sample_negs(n, forbidden):
        negs = set()
        while len(negs) < n:
            batch = RNG.choice(card_pool, size=(n, 2))
            for a, b in batch:
                if a == b:
                    continue
                key = (int(min(a, b)), int(max(a, b)))
                if key not in forbidden and key not in negs:
                    negs.add(key)
                    if len(negs) >= n:
                        break
        return negs

    train_neg = sample_negs(len(train_pos), train_pos)

    test_pos_all = sample_pairs(test_decks, per_deck=30)
    test_pos = list((test_pos_all - train_pos) - showcase_rows)[:50_000]
    test_neg = list(sample_negs(len(test_pos), train_pos | set(test_pos)))
    print(f"test: {len(test_pos)} novel positives")

    # ---- baselines ------------------------------------------------------
    def cos_scores(pairs, vectors):
        a = np.array([p[0] for p in pairs])
        b = np.array([p[1] for p in pairs])
        return (vectors[a] * vectors[b]).sum(1)

    print(f"baseline text-cosine AUC: "
          f"{auc(cos_scores(test_pos, emb), cos_scores(test_neg, emb)):.3f}")
    co = np.load(OUT / "covectors-60.npy")
    print(f"baseline co-space cosine AUC: "
          f"{auc(cos_scores(test_pos, co), cos_scores(test_neg, co)):.3f}")

    # ---- train ----------------------------------------------------------
    import torch
    torch.manual_seed(0)
    torch.set_num_threads(4)
    E = torch.from_numpy(emb)

    A = torch.nn.Parameter(torch.randn(RANK, 384) * 0.05)
    B = torch.nn.Parameter(torch.randn(RANK, 384) * 0.05)
    bias = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.Adam([A, B, bias], lr=2e-3)

    pos_arr = np.array(sorted(train_pos), dtype=np.int64)
    neg_arr = np.array(sorted(train_neg), dtype=np.int64)
    X = np.vstack([pos_arr, neg_arr])
    y = torch.cat([torch.ones(len(pos_arr)), torch.zeros(len(neg_arr))])

    def score_batch(idx):
        ea, eb = E[X[idx, 0]], E[X[idx, 1]]
        pa, pb = ea @ A.T, eb @ B.T
        qa, qb = eb @ A.T, ea @ B.T
        return (pa * pb).sum(1) + (qa * qb).sum(1) + bias

    n = len(X)
    for epoch in range(4):
        perm = np.random.permutation(n)
        total = 0.0
        for start in range(0, n, 8192):
            idx = perm[start:start + 8192]
            opt.zero_grad()
            s = score_batch(idx)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                s, y[idx])
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
        print(f"epoch {epoch}: loss {total / n:.4f}")

    with torch.no_grad():
        def model_scores(pairs):
            arr = np.array(pairs, dtype=np.int64)
            ea, eb = E[arr[:, 0]], E[arr[:, 1]]
            s = ((ea @ A.T) * (eb @ B.T)).sum(1) + ((eb @ A.T) * (ea @ B.T)).sum(1)
            return (s + bias).numpy()

        model_auc = auc(model_scores(test_pos), model_scores(test_neg))
        print(f"\ntrained pair-synergy AUC: {model_auc:.3f}")

        # showcase: percentile vs a large random-pair reference
        ref_pairs = list(sample_negs(20_000, train_pos))
        ref = np.sort(model_scores(ref_pairs))
        print("\nshowcase pairs (excluded from training; percentile vs random pairs):")
        for a, b in SHOWCASE:
            ra, rb = name_to_row.get(norm(a)), name_to_row.get(norm(b))
            if ra is None or rb is None:
                continue
            s = model_scores([(ra, rb)])[0]
            pct = float(np.searchsorted(ref, s)) / len(ref) * 100
            print(f"  {pct:5.1f}%  {a} + {b}")

    np.savez(OUT / "pair_model.npz", A=A.detach().numpy(),
             B=B.detach().numpy(), bias=bias.detach().numpy())
    print(f"\nwrote {OUT / 'pair_model.npz'}")


if __name__ == "__main__":
    main()
