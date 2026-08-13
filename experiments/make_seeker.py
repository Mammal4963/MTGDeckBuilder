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
FORMATS = ["standard", "pioneer", "modern", "legacy", "vintage", "pauper", "commander"]
WUBRG = "WUBRG"
_PIP_RE = re.compile(r"\{([WUBRG])(?:/[WUBRGP])?\}")


def norm(name: str) -> str:
    return re.sub(r"\s+", " ", name).strip().lower()


def load_decks60(name_to_row, is_land, is_basic):
    """All 60-card decks (tournament + casual) as row/qty lists, for the
    in-browser recipe retrieval."""
    decks = []

    def record(fmt, pairs):
        cards, lands = {}, {}
        for name, qty in pairs:
            row = name_to_row.get(norm(name))
            if row is None:
                continue
            qty = int(qty)
            if is_land[row]:
                lands[row] = lands.get(row, 0) + qty
            else:
                cards[row] = min(cards.get(row, 0) + qty, 4)
        if len(cards) >= 8:
            decks.append({
                "f": fmt,
                "c": sorted([int(r), int(q)] for r, q in cards.items()),
                "l": sorted([int(r), int(q)] for r, q in lands.items()),
            })

    with gzip.open(ROOT / "data" / "decks" / "mtgo-decks.jsonl.gz", "rt") as fh:
        for line in fh:
            deck = json.loads(line)
            record(deck["format"], deck["main"])
    with gzip.open(ROOT / "data" / "decks" / "archidekt-decks.jsonl.gz", "rt") as fh:
        for line in fh:
            deck = json.loads(line)
            if deck["format"] != "commander":
                record(deck["format"], deck["cards"])
    return decks


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

    colors, legal, names, types, mvs, pips = [], [], [], [], [], []
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
        mvs.append(int(min(m["mana_value"], 15)))
        pips.append("".join(_PIP_RE.findall(card.mana_cost)) if card else "")

    name_to_row = {}
    for i, m in enumerate(meta):
        name_to_row.setdefault(norm(m["name"]), i)
        if " // " in m["name"]:
            name_to_row.setdefault(norm(m["name"].split(" // ")[0]), i)
    is_land = [("Land" in m["type_line"]) for m in meta]
    is_basic = [("Basic" in m["type_line"]) for m in meta]
    decks60 = load_decks60(name_to_row, is_land, is_basic)
    basics = {c: name_to_row[norm(n)] for c, n in
              {"W": "Plains", "U": "Island", "B": "Swamp",
               "R": "Mountain", "G": "Forest"}.items()}
    print(f"{len(decks60)} recipe decks bundled")

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
        "mv": mvs,
        "pips": pips,
        "decks60": decks60,
        "basics": basics,
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
