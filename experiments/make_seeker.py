"""Build the deck seeker: pick cards, watch the good options collapse.

Bundles every card's blended co-occurrence vector (int8-quantized) plus
name/type/colors/legality/popularity into a self-contained page that
re-ranks all candidates on every pick.

Reads  output/{covectors.npy,covocab.json,cards_meta.json}
       data/scryfall-oracle-cards-2026-08-12.jsonl.gz  (legalities)
Writes output/deck-seeker.html
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mtg_deckbuilder.carddata import CardDatabase  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "output"
FORMATS = ["standard", "pioneer", "modern", "legacy", "vintage", "pauper"]
WUBRG = "WUBRG"


def main() -> None:
    vecs = np.load(OUT / "covectors.npy")
    meta = json.loads((OUT / "cards_meta.json").read_text())
    covocab = json.loads((OUT / "covocab.json").read_text())
    deck_freq = {int(k): v for k, v in covocab["deck_freq"].items()}
    db = CardDatabase.load(ROOT / "data" / "scryfall-oracle-cards-2026-08-12.jsonl.gz")

    q = np.clip(np.round(vecs * 127), -127, 127).astype(np.int8)
    b64 = base64.b64encode(q.tobytes()).decode()

    colors, legal, freq, names, types = [], [], [], [], []
    for i, m in enumerate(meta):
        card = db.get(m["name"])
        mask = 0
        for b, c in enumerate(WUBRG):
            if c in (card.color_identity if card else ()):
                mask |= 1 << b
        if m["color"] == "land":
            mask |= 1 << 5
        colors.append(mask)
        lmask = 0
        for b, fmt in enumerate(FORMATS):
            if card and card.legalities.get(fmt) in ("legal", "restricted"):
                lmask |= 1 << b
        legal.append(lmask)
        freq.append(deck_freq.get(i, 0))
        names.append(m["name"])
        types.append(m["type_line"][:48])

    data = {
        "formats": FORMATS,
        "dim": int(vecs.shape[1]),
        "names": names,
        "types": types,
        "colors": colors,
        "legal": legal,
        "freq": freq,
    }
    template = (Path(__file__).resolve().parent / "seeker_template.html").read_text()
    html = (template
            .replace("__DATA__", json.dumps(data, separators=(",", ":")))
            .replace("__VECS__", b64))
    out = OUT / "deck-seeker.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(names)} cards)")


if __name__ == "__main__":
    main()
