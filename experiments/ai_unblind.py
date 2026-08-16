"""Rung 1 of the custom-AI-pilot ladder: un-blind Forge's AI.

Forge's card scripts mark 2,525 cards `AI:RemoveDeck:All` - the built-in
AI is forbidden from playing them (data/forge-ai-unplayable.json;
discovered when Tainted Aether was cast in 0/18 baseline games). This
tool strips those hints for chosen cards, in place, inside the
install's cardsfolder.zip, so the AI at least *attempts* them with its
generic heuristics.

The original zip is backed up once (cardsfolder.zip.orig); --restore
puts it back. While applied, ALL sims see the modified scripts - so
apply, measure, restore.

Usage:
  python3 experiments/ai_unblind.py --apply "Tainted Aether" "Acorn Catapult"
  python3 experiments/ai_unblind.py --restore
  python3 experiments/ai_unblind.py --status
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from improve_deck import FORGE_DIR  # noqa: E402

ZIP = FORGE_DIR / "res" / "cardsfolder" / "cardsfolder.zip"
BACKUP = ZIP.with_suffix(".zip.orig")
AI_HINT = re.compile(r"^AI:RemoveDeck:(All|Random)\s*$", re.M)


def entry_for(name: str) -> str:
    base = re.sub(r"[^a-z0-9 _-]", "", name.lower()).replace(" ", "_")
    return f"{base[0]}/{base}.txt"


def apply(names: list[str]) -> None:
    if not BACKUP.exists():
        shutil.copy2(ZIP, BACKUP)
        print(f"backed up pristine zip -> {BACKUP.name}")
    targets = {entry_for(n): n for n in names}
    tmp = ZIP.with_suffix(".zip.tmp")
    changed = []
    with zipfile.ZipFile(ZIP) as zin, \
            zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for info in zin.infolist():
            data = zin.read(info)
            if info.filename in targets:
                txt = data.decode("utf-8", "replace")
                stripped, n = AI_HINT.subn("", txt)
                if n:
                    data = re.sub(r"\n{3,}", "\n\n", stripped).encode()
                    changed.append(targets[info.filename])
            zout.writestr(info, data)
    tmp.replace(ZIP)
    missing = set(names) - set(changed)
    print(f"un-blinded: {changed}")
    if missing:
        print(f"NOT FOUND or had no hint: {sorted(missing)}")


def restore() -> None:
    if BACKUP.exists():
        shutil.copy2(BACKUP, ZIP)
        print("restored pristine cardsfolder.zip")
    else:
        print("no backup present - nothing to restore")


def status() -> None:
    print(f"zip: {ZIP}")
    print(f"backup exists: {BACKUP.exists()}")
    if BACKUP.exists():
        print(f"sizes: current {ZIP.stat().st_size}, "
              f"pristine {BACKUP.stat().st_size} "
              f"({'MODIFIED' if ZIP.stat().st_size != BACKUP.stat().st_size else 'identical size'})")


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--apply", nargs="+", metavar="CARD")
    g.add_argument("--restore", action="store_true")
    g.add_argument("--status", action="store_true")
    args = ap.parse_args()
    if args.apply:
        apply(args.apply)
    elif args.restore:
        restore()
    else:
        status()


if __name__ == "__main__":
    main()
