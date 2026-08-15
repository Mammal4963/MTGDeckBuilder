"""Train the combo head: does (A, B) form a MECHANICAL combo?

The pair-synergy model (pair_model.py) learns "belong in the same
deck" from co-occurrence - it cannot distinguish a combo (cards that
form a loop/lock/win together) from ordinary synergy. Commander
Spellbook's curated database is ground truth for the former, so we
fine-tune a second bilinear head on the same frozen MiniLM text
embeddings:

    combo(a,b) = <A e_a, B e_b> + <A e_b, B e_a> + bias

Positives: pairs of cards appearing together in a verified Spellbook
combo (combos of size 2-4; a pair inside a 4-card combo is a weaker
but real signal - they are combo partners needing extra pieces).

Negatives, half and half:
  - HARD: pairs co-played in real decks but in no Spellbook combo
    (synergy-without-combo - the discrimination we actually want)
  - random in-pool pairs (calibration)

Split: 10% of CARDS held out; any pair touching a held-out card is
test-only. This measures generalization to unseen cards, which is the
whole point - scoring pairs nobody has catalogued.

Writes output/combo_model.npz, prints AUCs vs both negative types and
the pair-model baseline on the identical test set.
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
RNG = np.random.default_rng(47)
RANK = 128
MAX_COMBO_SIZE = 4

SHOWCASE = [
    ("Thassa's Oracle", "Demonic Consultation"),
    ("Splinter Twin", "Deceiver Exarch"),
    ("Heliod, Sun-Crowned", "Walking Ballista"),
    ("Sanguine Bond", "Exquisite Blood"),
    ("Basalt Monolith", "Rings of Brighthearth"),
    ("Blood Artist", "Viscera Seer"),          # synergy, NOT a 2-card combo
    ("Guttersnipe", "Lightning Bolt"),          # synergy, NOT a combo
    ("Healing Salve", "Goblin Guide"),          # nonsense
]


def norm(name: str) -> str:
    return re.sub(r"\s+", " ", name).strip().lower()


def load_spellbook_pairs(name_to_row):
    """(pair -> min combo size) for every in-vocab Spellbook pair."""
    pairs = {}
    n_combos = n_matched = 0
    with gzip.open(ROOT / "data/combos/spellbook-combos.jsonl.gz", "rt") as fh:
        for line in fh:
            c = json.loads(line)
            n_combos += 1
            if len(c["cards"]) > MAX_COMBO_SIZE:
                continue
            rows = sorted({name_to_row[norm(n)] for n in c["cards"]
                           if norm(n) in name_to_row})
            if len(rows) < 2 or len(rows) < len(c["cards"]):
                continue                      # only fully-resolved combos
            n_matched += 1
            for i in range(len(rows)):
                for j in range(i + 1, len(rows)):
                    key = (rows[i], rows[j])
                    sz = len(rows)
                    if key not in pairs or sz < pairs[key]:
                        pairs[key] = sz
    print(f"{n_matched}/{n_combos} combos fully in vocab -> "
          f"{len(pairs)} unique combo pairs")
    return pairs


def load_deck_pairs(name_to_row, is_land, cap=2_000_000):
    """Pairs co-played in real decks (for hard negatives)."""
    pairs = set()
    for path, key, min_cards in (
            (ROOT / "data/decks/mtgo-decks.jsonl.gz", "main", 8),
            (ROOT / "data/decks/archidekt-decks.jsonl.gz", "cards", 15)):
        try:
            with gzip.open(path, "rt") as fh:
                for line in fh:
                    if not line.endswith("\n"):
                        continue              # crawler may be mid-append
                    d = json.loads(line)
                    rows = sorted({name_to_row[norm(n)] for n, _q in d[key]
                                   if norm(n) in name_to_row})
                    rows = [r for r in rows if not is_land[r]]
                    if len(rows) < min_cards:
                        continue
                    take = RNG.choice(len(rows), size=(min(60, len(rows)), 2))
                    for i, j in take:
                        if i != j:
                            a, b = rows[i], rows[j]
                            pairs.add((a, b) if a < b else (b, a))
                    if len(pairs) >= cap:
                        return pairs
        except EOFError:
            continue
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

    combo_pairs = load_spellbook_pairs(name_to_row)
    deck_pairs = load_deck_pairs(name_to_row, is_land)
    print(f"{len(deck_pairs)} deck co-play pairs sampled")

    showcase_rows = {tuple(sorted((name_to_row[norm(a)], name_to_row[norm(b)])))
                     for a, b in SHOWCASE
                     if norm(a) in name_to_row and norm(b) in name_to_row}

    combo_cards = sorted({r for p in combo_pairs for r in p})
    card_pool = np.array(sorted({r for p in deck_pairs for r in p}
                                | set(combo_cards)))
    holdout = set(RNG.choice(combo_cards,
                             size=max(1, len(combo_cards) // 10),
                             replace=False).tolist())
    print(f"{len(combo_cards)} combo cards, {len(holdout)} held out")

    all_pos = set(combo_pairs) - showcase_rows
    test_pos = [p for p in all_pos if p[0] in holdout or p[1] in holdout]
    train_pos = [p for p in all_pos if p[0] not in holdout and p[1] not in holdout]
    print(f"{len(train_pos)} train / {len(test_pos)} test combo pairs")

    hard_all = list(deck_pairs - set(combo_pairs) - showcase_rows)
    RNG.shuffle(hard_all)
    is_test_pair = lambda p: p[0] in holdout or p[1] in holdout
    hard_test = [p for p in hard_all if is_test_pair(p)][:len(test_pos)]
    hard_train = [p for p in hard_all if not is_test_pair(p)][:len(train_pos)]

    def sample_random(n, forbidden):
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
        return list(negs)

    rand_train = sample_random(len(train_pos), set(combo_pairs))
    rand_test = sample_random(len(test_pos), set(combo_pairs) | set(hard_test))

    # ---- train ----------------------------------------------------------
    import torch
    torch.manual_seed(0)
    torch.set_num_threads(4)
    E = torch.from_numpy(emb)

    A = torch.nn.Parameter(torch.randn(RANK, 384) * 0.05)
    B = torch.nn.Parameter(torch.randn(RANK, 384) * 0.05)
    bias = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.Adam([A, B, bias], lr=2e-3)

    X = np.array(train_pos + hard_train + rand_train, dtype=np.int64)
    y = torch.cat([torch.ones(len(train_pos)),
                   torch.zeros(len(hard_train) + len(rand_train))])

    n = len(X)
    for epoch in range(6):
        perm = np.random.permutation(n)
        total = 0.0
        for start in range(0, n, 8192):
            idx = perm[start:start + 8192]
            ea, eb = E[X[idx, 0]], E[X[idx, 1]]
            s = ((ea @ A.T) * (eb @ B.T)).sum(1) \
                + ((eb @ A.T) * (ea @ B.T)).sum(1) + bias
            loss = torch.nn.functional.binary_cross_entropy_with_logits(s, y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
        print(f"epoch {epoch}: loss {total / n:.4f}")

    with torch.no_grad():
        def scores_with(pairs, Am, Bm, bm):
            arr = np.array(pairs, dtype=np.int64)
            ea, eb = E[arr[:, 0]], E[arr[:, 1]]
            s = ((ea @ Am.T) * (eb @ Bm.T)).sum(1) \
                + ((eb @ Am.T) * (ea @ Bm.T)).sum(1)
            return (s + bm).numpy()

        model_scores = lambda p: scores_with(p, A, B, bias)
        print(f"\ncombo head AUC vs HARD negatives (deck synergy, non-combo): "
              f"{auc(model_scores(test_pos), model_scores(hard_test)):.3f}")
        print(f"combo head AUC vs random pairs: "
              f"{auc(model_scores(test_pos), model_scores(rand_test)):.3f}")

        pm = np.load(OUT / "pair_model.npz")
        pA, pB = torch.from_numpy(pm["A"]), torch.from_numpy(pm["B"])
        pbias = torch.from_numpy(pm["bias"])
        pair_scores = lambda p: scores_with(p, pA, pB, pbias)
        print(f"pair-model baseline AUC vs HARD negatives: "
              f"{auc(pair_scores(test_pos), pair_scores(hard_test)):.3f}")
        print(f"pair-model baseline AUC vs random: "
              f"{auc(pair_scores(test_pos), pair_scores(rand_test)):.3f}")

        ref = np.sort(model_scores(sample_random(20_000, set(combo_pairs))))
        print("\nshowcase (excluded from training; percentile vs random pairs):")
        for a, b in SHOWCASE:
            ra, rb = name_to_row.get(norm(a)), name_to_row.get(norm(b))
            if ra is None or rb is None:
                print(f"   ?     {a} + {b} (not in vocab)")
                continue
            s = model_scores([(ra, rb)])[0]
            pct = float(np.searchsorted(ref, s)) / len(ref) * 100
            known = "known combo" if tuple(sorted((ra, rb))) in combo_pairs \
                else "not catalogued"
            print(f"  {pct:5.1f}%  {a} + {b}  [{known}]")

    np.savez(OUT / "combo_model.npz", A=A.detach().numpy(),
             B=B.detach().numpy(), bias=bias.detach().numpy())
    print(f"\nwrote {OUT / 'combo_model.npz'}")


if __name__ == "__main__":
    main()
