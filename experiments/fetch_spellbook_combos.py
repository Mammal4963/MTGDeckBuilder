"""Fetch the Commander Spellbook combo database and distill it.

Commander Spellbook (https://commanderspellbook.com) curates ~100k
verified combo "variants" - the ground truth for MECHANICAL combos that
our pair-synergy model (trained on deck co-occurrence) cannot learn on
its own: co-occurrence says "these belong in the same 75", Spellbook
says "these cards form a loop/lock/win".

Respectful use: they publish a single bulk export
(https://json.commanderspellbook.com/variants.json.gz, ~27 MB) exactly
so consumers don't crawl the paginated API. One fetch, distilled to
card names + produced effects, committed to
data/combos/spellbook-combos.jsonl.gz (~1-2 MB).

Usage:
    python3 experiments/fetch_spellbook_combos.py [--from-file cached.json.gz]
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

BULK_URL = "https://json.commanderspellbook.com/variants.json.gz"
OUT = Path(__file__).resolve().parent.parent / "data" / "combos" / "spellbook-combos.jsonl.gz"
UA = "mtg-deckbuilder-research/0.1 (one-time combo dataset fetch; github Mammal4963/MTGDeckBuilder)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-file", type=Path, default=None,
                    help="use an already-downloaded variants.json.gz")
    args = ap.parse_args()

    if args.from_file:
        raw = args.from_file.read_bytes()
    else:
        import requests
        print(f"downloading {BULK_URL} ...")
        resp = requests.get(BULK_URL, headers={"User-Agent": UA}, timeout=600)
        resp.raise_for_status()
        raw = resp.content

    data = json.loads(gzip.decompress(raw))
    variants = data["variants"]
    print(f"{len(variants)} variants (export version {data.get('version')}, "
          f"timestamp {data.get('timestamp')})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with gzip.open(OUT, "wt", encoding="utf-8") as out:
        for v in variants:
            if v.get("status") != "OK":
                continue
            cards = sorted(u["card"]["name"] for u in v["uses"])
            if len(cards) < 2:
                continue                     # single-card "combos" teach nothing about pairs
            out.write(json.dumps({
                "id": v["id"],
                "cards": cards,
                "produces": sorted(p["feature"]["name"] for p in v["produces"]),
                "identity": v.get("identity", ""),
                "popularity": v.get("popularity") or 0,
            }, separators=(",", ":")) + "\n")
            kept += 1
    print(f"wrote {kept} combos -> {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
