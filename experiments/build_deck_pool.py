"""Build a Forge-verified deck pool for generalist pilot training.

Samples corpus decks (mtgo-decks.jsonl.gz), writes them as pool_XX.dck,
and verifies each with a 1-game Forge sim (drops decks that fail to
load or produce no result). Verified decks land in the Forge decks dir
and experiments/decks/pool/.

Usage:
  FORGE_SIM_SERVER=1 python experiments/build_deck_pool.py --n 50
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from improve_deck import FORGE_DECKS, write_dck  # noqa: E402
import sim_server  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
POOL_DIR = Path(__file__).resolve().parent / "decks" / "pool"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--candidates", type=int, default=90)
    args = ap.parse_args()

    decks = []
    with gzip.open(ROOT / "data" / "decks" / "mtgo-decks.jsonl.gz",
                   "rt", encoding="utf-8") as f:
        for ln in f:
            d = json.loads(ln)
            main_pairs = [(n2, q) for n2, q in d.get("main", [])]
            total = sum(q for _n2, q in main_pairs)
            if 58 <= total <= 80:
                decks.append((d.get("player") or d.get("event") or "deck",
                              d.get("format", "?"), main_pairs))
    rng = np.random.default_rng(29)
    rng.shuffle(decks)
    print(f"{len(decks)} corpus decks; sampling {args.candidates} "
          f"candidates for {args.n} slots", flush=True)

    POOL_DIR.mkdir(parents=True, exist_ok=True)
    client = sim_server.shared_client("poolcheck")
    kept = 0
    seen_names = set()
    for name, fmt, pairs in decks[:args.candidates]:
        if kept >= args.n:
            break
        key = re.sub(r"\W+", "", name.lower())[:24]
        if key in seen_names:
            continue
        seen_names.add(key)
        dck = f"pool_{kept:02d}"
        write_dck(dck, pairs)
        out = client.run(dck, "burn", 1, quiet=True, timeout_s=120)
        ok = len(re.findall(r"Game Result", out)) == 1 \
            and "could not be loaded" not in out
        if ok:
            (POOL_DIR / f"{dck}.dck").write_text(
                (FORGE_DECKS / f"{dck}.dck").read_text())
            kept += 1
            print(f"[{kept}/{args.n}] {dck} <- {name} ({fmt})", flush=True)
        else:
            (FORGE_DECKS / f"{dck}.dck").unlink(missing_ok=True)
    client.close()
    print(f"pool complete: {kept} verified decks", flush=True)


if __name__ == "__main__":
    main()
