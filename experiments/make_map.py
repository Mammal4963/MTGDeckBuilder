"""Build the interactive card map: a self-contained HTML file with every
playable Magic card as a point (UMAP of rules-text embeddings).

Reads  experiments/output/coords.npy, cards_meta.json
Writes experiments/output/card-map.html
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "output"

# board_wipe folds into removal: same functional family, and keeping both
# would put red and orange on screen together (fails palette validation).
ROLES = ["removal", "counterspell", "ramp", "draw",
         "tutor", "recursion", "card_selection", "other"]
ROLE_FOLD = {"board_wipe": "removal"}
COLORS = ["W", "U", "B", "R", "G", "multicolor", "colorless", "land"]
TYPES = ["Creature", "Land", "Instant", "Sorcery", "Planeswalker",
         "Artifact", "Enchantment", "Other"]


def type_label(type_line: str) -> str:
    for t in TYPES[:-1]:
        if t in type_line:
            return t
    return "Other"


def main() -> None:
    coords = np.load(OUT / "coords.npy")
    meta = json.loads((OUT / "cards_meta.json").read_text(encoding="utf-8"))
    assert len(meta) == coords.shape[0]

    data = {
        "x": [round(float(v), 3) for v in coords[:, 0]],
        "y": [round(float(v), 3) for v in coords[:, 1]],
        "name": [m["name"] for m in meta],
        "tl": [m["type_line"] for m in meta],
        "cost": [m["mana_cost"] for m in meta],
        "role": [ROLES.index(ROLE_FOLD.get(m["role"], m["role"]))
                 if ROLE_FOLD.get(m["role"], m["role"]) in ROLES
                 else len(ROLES) - 1
                 for m in meta],
        "color": [COLORS.index(m["color"]) for m in meta],
        "type": [TYPES.index(type_label(m["type_line"])) for m in meta],
    }

    template = (Path(__file__).resolve().parent / "map_template.html").read_text(
        encoding="utf-8"
    )
    html = template.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    out = OUT / "card-map.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(meta)} cards)")


if __name__ == "__main__":
    main()
