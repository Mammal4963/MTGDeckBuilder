"""Build the interactive card map: a self-contained HTML file with every
playable Magic card as a point (UMAP of rules-text embeddings).

Reads  experiments/output/coords.npy, cards_meta.json
Writes experiments/output/card-map.html
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mtg_deckbuilder.carddata import CardDatabase  # noqa: E402
from mtg_deckbuilder.tags import tag_card  # noqa: E402

OUT = Path(__file__).resolve().parent / "output"
DATA_SNAPSHOT = (Path(__file__).resolve().parent.parent / "data"
                 / "scryfall-oracle-cards-2026-08-12.jsonl.gz")

# board_wipe folds into removal: same functional family, and keeping both
# would put red and orange on screen together (fails palette validation).
FUNCTIONS = ["board_wipe", "removal", "counterspell", "ramp", "draw",
             "tutor", "recursion", "card_selection"]
ROLE_FOLD = {"board_wipe": "removal"}

# Theme archetypes break out the former "other" bucket (65% of the map).
# Priority order: sharper archetype identities first, broad ones later,
# so a card tagged both sacrifice:payoff and tokens:enabler reads as
# aristocrats. Prefixes ending in "_" match tag families (tribal_*).
THEMES = [
    ("aristocrats", ("sacrifice",)),
    ("graveyard & mill", ("graveyard", "mill")),
    ("blink & flicker", ("blink",)),
    ("spellslinger", ("spells",)),
    ("auras & equipment", ("voltron",)),
    ("tokens", ("tokens",)),
    ("+1/+1 counters", ("counters",)),
    ("lifegain", ("lifegain",)),
    ("artifacts matter", ("artifacts",)),
    ("enchantments matter", ("enchantments",)),
    ("tribal", ("tribal_",)),
]

ROLES = (sorted(set(ROLE_FOLD.get(f, f) for f in FUNCTIONS),
                key=lambda f: FUNCTIONS.index(f))
         + [label for label, _p in THEMES] + ["other"])
COLORS = ["W", "U", "B", "R", "G", "multicolor", "colorless", "land"]
TYPES = ["Creature", "Land", "Instant", "Sorcery", "Planeswalker",
         "Artifact", "Enchantment", "Sticker"]


def type_label(type_line: str) -> str:
    # The catch-all is literally just Un-set sticker sheets (48 cards),
    # so it gets named what it is instead of "Other".
    for t in TYPES[:-1]:
        if t in type_line:
            return t
    return "Sticker"


def primary_role(card) -> str:
    tags = tag_card(card)
    for role in FUNCTIONS:
        if role in tags:
            return ROLE_FOLD.get(role, role)
    base = {t.split(":")[0] for t in tags}
    for label, prefixes in THEMES:
        for p in prefixes:
            if (any(b.startswith(p) for b in base) if p.endswith("_")
                    else p in base):
                return label
    return "other"


def main() -> None:
    coords = np.load(OUT / "coords.npy")
    meta = json.loads((OUT / "cards_meta.json").read_text(encoding="utf-8"))
    assert len(meta) == coords.shape[0]

    # Roles are recomputed from the tagger (not read from cards_meta.json)
    # so the map picks up taxonomy changes without a re-embedding run.
    db = CardDatabase.load(DATA_SNAPSHOT)
    roles = []
    for m in meta:
        card = db.get(m["name"])
        roles.append(primary_role(card) if card is not None else "other")

    data = {
        "x": [round(float(v), 3) for v in coords[:, 0]],
        "y": [round(float(v), 3) for v in coords[:, 1]],
        "name": [m["name"] for m in meta],
        "tl": [m["type_line"] for m in meta],
        "cost": [m["mana_cost"] for m in meta],
        "role": [ROLES.index(r) for r in roles],
        "color": [COLORS.index(m["color"]) for m in meta],
        "type": [TYPES.index(type_label(m["type_line"])) for m in meta],
    }

    # Optional meta overlay (present once phase 2's analyze_meta.py has run).
    overlay = {"formats": [], "heat": {}, "decks": []}
    meta_path = OUT / "meta.json"
    decks_path = OUT / "decks_meta.json"
    if meta_path.exists():
        blob = json.loads(meta_path.read_text(encoding="utf-8"))
        overlay["formats"] = blob["formats"]
        overlay["heat"] = blob["heat"]
    if decks_path.exists():
        by_fmt = {}
        for deck in json.loads(decks_path.read_text(encoding="utf-8")):
            by_fmt.setdefault(deck["format"], []).append(deck)
        for fmt in sorted(by_fmt):
            ranked = sorted(
                by_fmt[fmt], key=lambda d: (-d.get("wins", 0), d["date"]),
            )[:40]
            for deck in ranked:
                overlay["decks"].append({
                    "format": fmt,
                    "label": f'{deck["player"]} · '
                             f'{" + ".join(deck["top_cards"][:2])} · {deck["date"]}',
                    "rows": [r for r, _q in deck["cards"]],
                })

    template = (Path(__file__).resolve().parent / "map_template.html").read_text(
        encoding="utf-8"
    )
    html = template.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    html = html.replace("__META__", json.dumps(overlay, separators=(",", ":")))
    out = OUT / "card-map.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(meta)} cards)")


if __name__ == "__main__":
    main()
