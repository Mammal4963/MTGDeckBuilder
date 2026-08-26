"""Pool v4: ~450 decks the pilot has NEVER seen (round 21).

Fresh sources (decks/legends, remaining quest duels + geneticai, more
theme duels), deduped against the existing pool by a content hash of
the [Main] section - v3 renamed files, so names can't be trusted.
Output goes to experiments/decks/pool_v4 (a separate pool dir).

Usage:
  FORGE_SIM_SERVER=1 python experiments/build_pool_v4.py --target 450
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from improve_deck import FORGE_DIR, FORGE_DECKS  # noqa: E402
import sim_server  # noqa: E402

POOL_V1 = Path(__file__).resolve().parent / "decks" / "pool"
POOL_V4 = Path(__file__).resolve().parent / "decks" / "pool_v4"

SOURCES = [
    ("res/decks/legends", 200),
    ("res/geneticaidecks", 200),
    ("res/quest/duels", 200),
    ("res/adventures", 80),
]


def main_block(text):
    m = re.search(r"\[Main\](.*?)(\[|$)", text, re.S | re.I)
    return m.group(1) if m else ""


def main_size(text):
    total = 0
    for ln in main_block(text).splitlines():
        mm = re.match(r"\s*(\d+)\s+\S", ln)
        if mm:
            total += int(mm.group(1))
    return total


def main_hash(text):
    lines = sorted(ln.strip().lower()
                   for ln in main_block(text).splitlines() if ln.strip())
    return hashlib.sha1("\n".join(lines).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=450)
    args = ap.parse_args()
    POOL_V4.mkdir(exist_ok=True)

    seen = set()
    for f in list(POOL_V1.glob("*.dck")) + list(POOL_V4.glob("*.dck")):
        seen.add(main_hash(f.read_text(encoding="utf-8",
                                       errors="replace")))
    print(f"{len(seen)} existing deck hashes to exclude", flush=True)

    rng = np.random.default_rng(61)
    candidates = []
    for rel, take in SOURCES:
        base = FORGE_DIR / rel
        files = sorted(base.rglob("*.dck")) if base.exists() else []
        rng.shuffle(files)
        picked = 0
        for f in files:
            if picked >= take:
                break
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if not (58 <= main_size(text) <= 80):
                continue
            h = main_hash(text)
            if h in seen:
                continue
            seen.add(h)
            candidates.append((rel, f, text))
            picked += 1
        print(f"{rel}: {picked} fresh candidates", flush=True)

    rng.shuffle(candidates)
    client = sim_server.shared_client("poolv4")
    kept = len(list(POOL_V4.glob("*.dck")))
    while_start = kept
    for rel, f, text in candidates:
        if kept >= args.target:
            break
        dck = f"poolv4_{kept:03d}"
        text2 = re.sub(r"(?mi)^Name=.*$", f"Name={dck}", text, count=1)
        if "Name=" not in text2.split("[Main]")[0]:
            text2 = f"[metadata]\nName={dck}\n" + text2
        (FORGE_DECKS / f"{dck}.dck").write_text(text2, encoding="utf-8")
        out = client.run(dck, "burn", 1, quiet=True, timeout_s=120)
        if len(re.findall(r"Game Result", out)) == 1:
            (POOL_V4 / f"{dck}.dck").write_text(text2, encoding="utf-8")
            kept += 1
            if kept % 25 == 0:
                print(f"[{kept}/{args.target}] latest: {f.name} ({rel})",
                      flush=True)
        else:
            (FORGE_DECKS / f"{dck}.dck").unlink(missing_ok=True)
    client.close()
    print(f"pool v4 complete: +{kept - while_start} "
          f"(total {kept} never-seen decks)", flush=True)


if __name__ == "__main__":
    main()
