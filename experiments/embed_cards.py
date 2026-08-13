"""Embed every oracle card's rules text into a vector space.

Phase 1 of the geometry-driven deck building experiments: turn each card
into a point so that cards with similar *function* land near each other.

The embedded string is composed, not raw oracle text - type line, mana
cost, and stats change what a card *is* (a 1-mana 2/2 and a 6-mana 2/2
are different objects), and self-referencing names are normalized to `~`
so "Lightning Bolt deals 3 damage" and "Shock deals 2 damage" read the
same way (reusing the tagger's normalizer).

Outputs (in experiments/output/):
  embeddings.npy   float32 [n_cards, dim], L2-normalized
  cards_meta.json  per-card metadata aligned with the rows
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mtg_deckbuilder.carddata import Card, CardDatabase  # noqa: E402
from mtg_deckbuilder.tags import normalize_text, tag_card  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data" / "scryfall-oracle-cards-2026-08-12.jsonl.gz"
OUT = Path(__file__).resolve().parent / "output"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Priority order for the single "primary role" label used in the map.
ROLE_PRIORITY = [
    "board_wipe", "removal", "counterspell", "ramp", "draw",
    "tutor", "recursion", "card_selection",
]


def playable(card: Card) -> bool:
    """Real cards only: legal in at least one constructed format."""
    return any(v == "legal" for v in card.legalities.values())


def compose(card: Card) -> str:
    parts = [card.type_line]
    if card.mana_cost:
        parts.append(f"cost {card.mana_cost}")
    if card.power is not None:
        parts.append(f"{card.power}/{card.toughness}")
    text = normalize_text(card).strip()
    if text:
        parts.append(text)
    return ". ".join(parts)


def primary_role(card: Card) -> str:
    tags = tag_card(card)
    for role in ROLE_PRIORITY:
        if role in tags:
            return role
    return "other"


def color_label(card: Card) -> str:
    if card.is_land:
        return "land"
    identity = [c for c in "WUBRG" if c in card.color_identity]
    if not identity:
        return "colorless"
    if len(identity) > 1:
        return "multicolor"
    return identity[0]


def main() -> None:
    from sentence_transformers import SentenceTransformer

    db = CardDatabase.load(DATA)
    cards = [c for c in db.cards if playable(c)]
    print(f"{len(db)} cards in snapshot, {len(cards)} playable (legal somewhere)")

    texts = [compose(c) for c in cards]
    meta = [
        {
            "name": c.name,
            "type_line": c.type_line,
            "mana_cost": c.mana_cost,
            "mana_value": c.mana_value,
            "color": color_label(c),
            "role": primary_role(c),
            "rarity": c.rarity,
        }
        for c in cards
    ]

    print(f"loading {MODEL} ...")
    model = SentenceTransformer(MODEL, device="cpu")
    embeddings = model.encode(
        texts,
        batch_size=256,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    print("embeddings:", embeddings.shape)

    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "embeddings.npy", embeddings)
    with open(OUT / "cards_meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh)
    print(f"wrote {OUT / 'embeddings.npy'} and cards_meta.json")


if __name__ == "__main__":
    main()
