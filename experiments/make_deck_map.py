"""Build the meta deck map: every tournament deck as a point.

Reads  experiments/output/deck_coords.npy, decks_meta.json
Writes experiments/output/deck-map.html
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "output"


def main() -> None:
    coords = np.load(OUT / "deck_coords.npy")
    meta = json.loads((OUT / "decks_meta.json").read_text(encoding="utf-8"))
    assert len(meta) == coords.shape[0]
    formats = sorted({m["format"] for m in meta})

    data = {
        "formats": formats,
        "x": [round(float(v), 3) for v in coords[:, 0]],
        "y": [round(float(v), 3) for v in coords[:, 1]],
        "f": [formats.index(m["format"]) for m in meta],
        "p": [m["player"] for m in meta],
        "e": [m["event"] for m in meta],
        "d": [m["date"] for m in meta],
        "top": [m["top_cards"] for m in meta],
    }
    template = (Path(__file__).resolve().parent / "deck_map_template.html").read_text(
        encoding="utf-8"
    )
    out = OUT / "deck-map.html"
    out.write_text(
        template.replace("__DATA__", json.dumps(data, separators=(",", ":"))),
        encoding="utf-8",
    )
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(meta)} decks)")


if __name__ == "__main__":
    main()
