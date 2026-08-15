"""Novel combo candidate mining (task: circle back to combo discovery).

v1: score with the COMBO HEAD (combo_model.py, fine-tuned on Commander
Spellbook ground truth) instead of the pair-synergy model. v0 used the
synergy model and - as documented in FORGE-NOTES - surfaced "unplayed
archetype fits", not mechanical combos; the combo head is trained to
make exactly that distinction (AUC 0.846 vs deck-synergy negatives on
held-out cards, where the synergy model scores 0.334).

Novelty = the pair is NOT catalogued in Spellbook. Observed deck
co-play no longer disqualifies a pair (a real undiscovered combo may
well be co-played coincidentally) - it's reported as a column instead.
Functional near-reprints (text cosine > 0.9) are still excluded.

Writes output/combo-candidates.json, prints the top 30.
"""
from __future__ import annotations

import gzip
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mtg_deckbuilder.carddata import CardDatabase  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "output"
TOPK = 15


def norm(name: str) -> str:
    return re.sub(r"\s+", " ", name).strip().lower()


def main():
    emb = np.load(OUT / "embeddings.npy").astype(np.float32)
    meta = json.loads((OUT / "cards_meta.json").read_text())
    pm = np.load(OUT / "combo_model.npz")
    P = emb @ pm["A"].T
    Q = emb @ pm["B"].T
    bias = float(pm["bias"][0])
    name_to_row = {}
    for i, m in enumerate(meta):
        name_to_row.setdefault(norm(m["name"]), i)
    is_land = np.array([("Land" in m["type_line"]) for m in meta])

    # known Spellbook pairs = already catalogued, not novel
    known = set()
    with gzip.open(ROOT / "data/combos/spellbook-combos.jsonl.gz", "rt") as fh:
        for line in fh:
            c = json.loads(line)
            rows = sorted({name_to_row[norm(n)] for n in c["cards"]
                           if norm(n) in name_to_row})
            for i_a in range(len(rows)):
                for i_b in range(i_a + 1, len(rows)):
                    known.add(rows[i_a] * len(meta) + rows[i_b])
    print(f"{len(known)} catalogued Spellbook pairs")

    # observed co-play + play counts
    observed = set()
    play_count = np.zeros(len(meta), int)
    def safe_lines(path):
        # The crawler may be appending to this file right now; read the
        # complete prefix and stop cleanly at any truncated tail.
        try:
            with gzip.open(path, "rt") as fh:
                for line in fh:
                    if line.endswith("\n"):
                        yield line
        except EOFError:
            return

    for path, key in ((ROOT / "data/decks/mtgo-decks.jsonl.gz", "main"),
                      (ROOT / "data/decks/archidekt-decks.jsonl.gz", "cards")):
        for line in safe_lines(path):
            d = json.loads(line)
            rows = sorted({name_to_row[norm(n)] for n, _q in d[key]
                           if norm(n) in name_to_row})
            rows = [r for r in rows if not is_land[r]]
            for r in rows:
                play_count[r] += 1
            for a_i in range(len(rows)):
                for b_i in range(a_i + 1, len(rows)):
                    observed.add(rows[a_i] * len(meta) + rows[b_i])
    print(f"{len(observed)} observed pairs")

    db = CardDatabase.load(ROOT / "data/scryfall-oracle-cards-2026-08-12.jsonl.gz")
    eligible = np.array([
        i for i, m in enumerate(meta)
        if not is_land[i]
        and (card := db.get(m["name"])) is not None
        and any(card.legalities.get(f) in ("legal", "restricted")
                for f in ("standard", "pioneer", "modern", "legacy",
                          "vintage", "pauper", "commander"))
    ])
    print(f"{len(eligible)} eligible cards")

    cands = []
    Pe, Qe, Ee = P[eligible], Q[eligible], emb[eligible]
    for start in range(0, len(eligible), 1024):
        chunk = slice(start, min(start + 1024, len(eligible)))
        s = Pe[chunk] @ Qe.T + Qe[chunk] @ Pe.T
        text_sim = Ee[chunk] @ Ee.T
        s[text_sim > 0.9] = -1e9                    # functional reprints
        top = np.argpartition(-s, TOPK, axis=1)[:, :TOPK]
        for li, gi in enumerate(range(chunk.start, chunk.stop)):
            a = int(eligible[gi])
            for lj in top[li]:
                b = int(eligible[lj])
                if a >= b:
                    continue
                key = a * len(meta) + b
                if key in known:
                    continue                        # already catalogued
                if play_count[a] == 0 and play_count[b] == 0:
                    continue                        # need one foot in reality
                cands.append((float(s[li, lj]) + bias, a, b,
                              key in observed))
    cands.sort(key=lambda x: -x[0])

    seen_cards = set()
    unique = []
    for s, a, b, coplayed in cands:
        if a in seen_cards and b in seen_cards:
            continue
        seen_cards.update((a, b))
        unique.append((s, a, b, coplayed))
        if len(unique) >= 100:
            break

    out = [{"score": round(s, 2),
            "a": meta[a]["name"], "a_type": meta[a]["type_line"],
            "b": meta[b]["name"], "b_type": meta[b]["type_line"],
            "a_played": int(play_count[a]), "b_played": int(play_count[b]),
            "coplayed": bool(coplayed)}
           for s, a, b, coplayed in unique]
    (OUT / "combo-candidates.json").write_text(json.dumps(out, indent=1))
    print("\nTop combo-head candidates not catalogued in Spellbook:")
    for e in out[:30]:
        tag = "co-played" if e["coplayed"] else "never together"
        print(f"  {e['score']:5.2f}  {e['a']} ({e['a_played']}x)  +  "
              f"{e['b']} ({e['b_played']}x)  [{tag}]")
    print(f"\nwrote {OUT / 'combo-candidates.json'}")


if __name__ == "__main__":
    main()
