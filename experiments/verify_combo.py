"""Verify combo hypotheses in Forge via mirror A/B simulation.

Method: build a 60-card shell around the pair, and a NULL twin that is
identical except the combo pieces are replaced by curve-matched vanilla
cards. Run combo-vs-null head-to-head (the mirror cancels everything
except the pair's contribution) plus both arms against a mid-power
reference deck. If the combo functions mechanically AND the Forge AI
can pilot it, the combo arm must win the mirror decisively.

Test 1 is a POSITIVE CONTROL - Sanguine Bond + Exquisite Blood, a
catalogued instant-win pair made of mandatory triggers (the thing
Forge's AI pilots best). It validates the methodology; only then does
the novel-miner candidate's result mean anything.

Usage: python3 experiments/verify_combo.py [--h2h 40] [--ref-games 20]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from improve_deck import write_dck, run_match, ci95  # noqa: E402

SHELL_BLACK = [
    ("Child of Night", 4), ("Vampire Nighthawk", 4),
    ("Gray Merchant of Asphodel", 4), ("Phyrexian Rager", 4),
    ("Doom Blade", 4), ("Sign in Blood", 4),
    ("Swamp", 24),
]
SHELL_WB = [
    ("Child of Night", 4), ("Phyrexian Rager", 4), ("Gravedigger", 4),
    ("Doom Blade", 4), ("Sign in Blood", 4),
    ("Scoured Barrens", 4), ("Swamp", 16), ("Plains", 4),
]

TESTS = [
    {
        "name": "control: Sanguine Bond + Exquisite Blood",
        "shell": SHELL_BLACK,
        "combo": [("Sanguine Bond", 4), ("Exquisite Blood", 4)],
        "null": [("Zombie Goliath", 4), ("Rotting Legion", 4)],
    },
    {
        "name": "novel: Zodiark, Umbral God + Magnanimous Magistrate",
        "shell": SHELL_WB,
        "combo": [("Zodiark, Umbral God", 4), ("Magnanimous Magistrate", 4)],
        "null": [("Zombie Goliath", 4), ("Serra Angel", 4)],
    },
]

# Mid-power reference: corpus pauper burn (already installed as burn.dck
# from the spike). Legacy gauntlet decks are too strong - both arms lose
# nearly every game and the signal drowns.
REFERENCE = "burn"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h2h", type=int, default=40)
    ap.add_argument("--ref-games", type=int, default=20)
    args = ap.parse_args()

    for i, test in enumerate(TESTS):
        a, b = f"combo_{i}", f"null_{i}"
        write_dck(a, test["shell"] + test["combo"])
        write_dck(b, test["shell"] + test["null"])

        print(f"\n=== {test['name']} ===", flush=True)
        w, n = run_match(a, b, args.h2h, 90 + 30 * args.h2h)
        p, half = ci95(w, n)
        print(f"mirror h2h: combo arm {w}/{n} = {p:.0%} ±{half:.0%}", flush=True)

        for arm in (a, b):
            w, n = run_match(arm, REFERENCE, args.ref_games,
                             90 + 30 * args.ref_games)
            p, half = ci95(w, n)
            print(f"vs {REFERENCE}: {arm} {w}/{n} = {p:.0%} ±{half:.0%}",
                  flush=True)


if __name__ == "__main__":
    main()
