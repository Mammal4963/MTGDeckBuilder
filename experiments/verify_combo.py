"""Verify combo hypotheses in Forge via assembly-conditional analysis.

v1 ran a mirror A/B (combo shell vs curve-matched null twin) and the
POSITIVE CONTROL came back 50% - diagnosis from verbose logs: with
4+4 pieces in 60 cards, games end before the combo is ever assembled,
so the mirror measures "8 situationally-dead cards vs 8 vanilla
bodies", not the combo.

v2 therefore measures the mechanically meaningful quantities from
verbose game logs:
  - assembly rate: fraction of games where BOTH pieces resolved
  - conditional winrate: of assembled games, how many the combo arm won
  - unassembled winrate: the baseline the shell provides on its own
Both arms get 4 Diabolic Tutor (symmetric shell change) to push the
assembly rate into measurable territory.

A combo VERIFIES if conditional winrate >> unassembled winrate.
Test 1 stays the positive control (Sanguine Bond + Exquisite Blood,
catalogued, mandatory triggers); test 2 is the miner's novel candidate.

Usage: python3 experiments/verify_combo.py [--games 40]
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from improve_deck import write_dck, ci95, FORGE_DIR  # noqa: E402

SHELL_BLACK = [
    ("Diabolic Tutor", 4), ("Vampire Nighthawk", 4),
    ("Gray Merchant of Asphodel", 4), ("Phyrexian Rager", 4),
    ("Doom Blade", 4), ("Sign in Blood", 4),
    ("Swamp", 24),
]
SHELL_WB = [
    ("Diabolic Tutor", 4), ("Phyrexian Rager", 4), ("Gravedigger", 4),
    ("Doom Blade", 4), ("Sign in Blood", 4),
    ("Scoured Barrens", 4), ("Swamp", 16), ("Plains", 4),
]

TESTS = [
    {
        "name": "control: Sanguine Bond + Exquisite Blood",
        "shell": SHELL_BLACK,
        "combo": [("Sanguine Bond", 4), ("Exquisite Blood", 4)],
        "null": [("Zombie Goliath", 4), ("Rotting Legion", 4)],
        # the drain loop visibly executing = chained Bond triggers
        "fired": (r"triggered Sanguine Bond", 3),
    },
    {
        "name": "novel: Zodiark, Umbral God + Magnanimous Magistrate",
        "shell": SHELL_WB,
        "combo": [("Zodiark, Umbral God", 4), ("Magnanimous Magistrate", 4)],
        "null": [("Zombie Goliath", 4), ("Serra Angel", 4)],
        # the interaction = Magistrate actually returning a creature
        "fired": (r"triggered Magnanimous Magistrate", 1),
    },
]


def run_verbose(deck_a: str, deck_b: str, games: int, timeout_s: int) -> str:
    import os
    if os.environ.get("FORGE_SIM_SERVER") == "1":
        import sim_server
        return sim_server.shared_client("verbose").run(
            deck_a, deck_b, games, quiet=False, timeout_s=timeout_s)
    from improve_deck import java_prefix
    cmd = java_prefix() + ["-Xmx3g",
           "-Dio.netty.tryReflectionSetAccessible=true",
           "-Dfile.encoding=UTF-8", "-jar",
           str(FORGE_DIR / "forge-gui-desktop-2.0.14-jar-with-dependencies.jar"),
           "sim", "-d", f"{deck_a}.dck", f"{deck_b}.dck", "-n", str(games)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout_s, cwd=FORGE_DIR)
        return proc.stdout + proc.stderr
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout or ""
        return out.decode("utf-8", "replace") if isinstance(out, bytes) else out


def analyze(log: str, combo_name: str, pieces, fired_pat=None) -> dict:
    """Walk the verbose log game by game; classify each by assembly.

    Only the combo arm plays the pieces, so any 'Resolve Stack: <piece>'
    inside a game segment is the combo player's. 'fired' is the subset
    of assembled games where the interaction visibly executed (pattern
    count >= threshold) - the decisive bucket: assembly can happen too
    late to matter, but a fired combo should win. Draws are counted
    separately so they can't misalign the winner bookkeeping.
    """
    res = {"assembled": [0, 0], "unassembled": [0, 0],
           "fired": [0, 0], "draws": 0}
    prev = 0
    for m in re.finditer(r"Game Result: Game \d+ ended[^\n]*", log):
        seg = log[prev:m.start()]
        prev = m.end()
        assembled = all(f"Resolve Stack: {p}" in seg for p in pieces)
        wm = re.search(r"Ai\(\d\)-(\S+) has won", m.group(0))
        if not wm:
            res["draws"] += 1
            continue
        won = wm.group(1) == combo_name
        bucket = "assembled" if assembled else "unassembled"
        res[bucket][1] += 1
        res[bucket][0] += won
        if assembled and fired_pat is not None:
            pat, thresh = fired_pat
            if len(re.findall(pat, seg)) >= thresh:
                res["fired"][1] += 1
                res["fired"][0] += won
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=40)
    args = ap.parse_args()

    for i, test in enumerate(TESTS):
        a, b = f"combo_{i}", f"null_{i}"
        write_dck(a, test["shell"] + test["combo"])
        write_dck(b, test["shell"] + test["null"])
        pieces = [n for n, _q in test["combo"]]

        print(f"\n=== {test['name']} ===", flush=True)
        log = run_verbose(a, b, args.games, 120 + 30 * args.games)
        (Path(__file__).resolve().parent / "output"
         / f"verify_{i}.log").write_text(log)
        res = analyze(log, a, pieces, test.get("fired"))
        for bucket in ("assembled", "fired", "unassembled"):
            w, n = res[bucket]
            if n == 0:
                print(f"{bucket}: no games", flush=True)
                continue
            p, half = ci95(w, n)
            print(f"{bucket}: combo arm {w}/{n} = {p:.0%} ±{half:.0%}",
                  flush=True)
        total = res["assembled"][1] + res["unassembled"][1] + res["draws"]
        if total:
            print(f"assembly rate: {res['assembled'][1]}/{total}"
                  + (f" ({res['draws']} draws)" if res["draws"] else ""),
                  flush=True)


if __name__ == "__main__":
    main()
