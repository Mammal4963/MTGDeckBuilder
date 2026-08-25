"""Pool v3: expand the training pool with Forge's own shipped decks.

Sources (sampled, then smoke-verified with a 1-game sim):
  - Forge res/geneticaidecks: lists evolved FOR the Forge AI
  - Forge quest/duels + decks/standard + quest/precons
  - MTGO corpus (more, via the existing sampler logic)

Verified decks land as poolv3_XXX.dck in the Forge constructed dir and
experiments/decks/pool/ (the training pool directory).

Usage:
  FORGE_SIM_SERVER=1 python experiments/build_pool_v3.py --target 450
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from improve_deck import FORGE_DIR, FORGE_DECKS  # noqa: E402
import sim_server  # noqa: E402

POOL_DIR = Path(__file__).resolve().parent / "decks" / "pool"

SOURCES = [
    ("res/geneticaidecks", 160),
    ("res/quest/duels", 120),
    ("res/decks/standard", 120),
    ("res/quest/precons", 60),
]


def main_size(text):
    m = re.search(r"\[Main\](.*?)(\[|$)", text, re.S | re.I)
    if not m:
        return 0
    total = 0
    for ln in m.group(1).splitlines():
        mm = re.match(r"\s*(\d+)\s+\S", ln)
        if mm:
            total += int(mm.group(1))
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=450)
    args = ap.parse_args()
    existing = len(list(POOL_DIR.glob("*.dck")))
    print(f"existing pool: {existing}; target total ~{args.target}",
          flush=True)

    rng = np.random.default_rng(53)
    candidates = []
    for rel, take in SOURCES:
        files = sorted((FORGE_DIR / rel).rglob("*.dck"))
        rng.shuffle(files)
        picked = 0
        for f in files:
            if picked >= take:
                break
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if 58 <= main_size(text) <= 80:
                candidates.append((rel, f, text))
                picked += 1
        print(f"{rel}: {picked} candidates", flush=True)

    rng.shuffle(candidates)
    client = sim_server.shared_client("poolv3")
    kept = 0
    need = max(0, args.target - existing)
    for rel, f, text in candidates:
        if kept >= need:
            break
        dck = f"poolv3_{kept:03d}"
        text2 = re.sub(r"(?mi)^Name=.*$", f"Name={dck}", text, count=1)
        if "Name=" not in text2.split("[Main]")[0]:
            text2 = f"[metadata]\nName={dck}\n" + text2
        (FORGE_DECKS / f"{dck}.dck").write_text(text2, encoding="utf-8")
        out = client.run(dck, "burn", 1, quiet=True, timeout_s=120)
        if len(re.findall(r"Game Result", out)) == 1:
            (POOL_DIR / f"{dck}.dck").write_text(text2, encoding="utf-8")
            kept += 1
            if kept % 25 == 0:
                print(f"[{kept}/{need}] latest: {f.name} ({rel})",
                      flush=True)
        else:
            (FORGE_DECKS / f"{dck}.dck").unlink(missing_ok=True)
    client.close()
    print(f"pool v3 complete: +{kept} decks "
          f"(total {existing + kept})", flush=True)


if __name__ == "__main__":
    main()
