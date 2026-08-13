"""Build the deck seeker: pick cards, watch the good options collapse.

Two separate learned synergy spaces (chosen with a toggle, never mixed):
  60  - MTGO tournament decks + casual 60-card community decks
  cmd - Commander community decks

Reads  output/covectors-{60,cmd}.npy, covocab-{60,cmd}.json, cards_meta.json
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
FORMATS = ["standard", "pioneer", "modern", "legacy", "vintage", "pauper", "commander"]
WUBRG = "WUBRG"


def quantize(path: Path) -> str:
    vecs = np.load(path)
    q = np.clip(np.round(vecs * 127), -127, 127).astype(np.int8)
    return base64.b64encode(q.tobytes()).decode()


def main() -> None:
    meta = json.loads((OUT / "cards_meta.json").read_text())
    db = CardDatabase.load(ROOT / "data" / "scryfall-oracle-cards-2026-08-12.jsonl.gz")

    freqs = {}
    for label in ("60", "cmd"):
        covocab = json.loads((OUT / f"covocab-{label}.json").read_text())
        deck_freq = {int(k): v for k, v in covocab["deck_freq"].items()}
        freqs[label] = [deck_freq.get(i, 0) for i in range(len(meta))]

    colors, legal, names, types = [], [], [], []
    for m in meta:
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
        names.append(m["name"])
        types.append(m["type_line"][:48])

    data = {
        "formats": FORMATS[:-1],
        "commanderBit": 1 << (len(FORMATS) - 1),
        "dim": 64,
        "names": names,
        "types": types,
        "colors": colors,
        "legal": legal,
        "freq60": freqs["60"],
        "freqcmd": freqs["cmd"],
    }
    template = (Path(__file__).resolve().parent / "seeker_template.html").read_text()
    html = (template
            .replace("__DATA__", json.dumps(data, separators=(",", ":")))
            .replace("__VECS60__", quantize(OUT / "covectors-60.npy"))
            .replace("__VECSCMD__", quantize(OUT / "covectors-cmd.npy")))
    out = OUT / "deck-seeker.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(names)} cards)")


if __name__ == "__main__":
    main()
