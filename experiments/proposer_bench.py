"""A/B benchmark for suggestion mechanisms.

Task: take a real corpus deck, remove one real playset, insert a junk
playset, and ask each proposer to rank candidate additions. The removed
card is ground truth; near-duplicate decks (same league lists) are
excluded from the neighborhood so the answer can't be copied verbatim.

Reports mean reciprocal rank and hit@5/@12 per proposer per format,
plus a qualitative side-by-side on the user's Tainted Aether jank deck.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from improve_deck import Improver, norm  # noqa: E402

RNG = np.random.default_rng(5)
N_PER_FORMAT = 60


def near_duplicates(imp, target_rows):
    """Corpus indexes sharing >=60% of distinct cards with the target."""
    tset = set(target_rows)
    out = set()
    for i, d in enumerate(imp.corpus):
        inter = len(tset & set(d["rows"]))
        union = len(tset | set(d["rows"]))
        if union and inter / union >= 0.6:
            out.add(i)
    return out


def bench_format(fmt: str):
    imp = Improver(fmt)
    junk_pool = [r for r in range(len(imp.meta))
                 if not imp.is_land[r] and imp.legal(r) and imp.playable(r)
                 and r not in imp._freq_cache()]
    results = {"v1": [], "v2": []}
    tested = 0
    order = RNG.permutation(len(imp.corpus))
    for di in order:
        if tested >= N_PER_FORMAT:
            break
        deck = dict(imp.corpus[di]["rows"])
        playsets = [r for r, q in deck.items()
                    if not imp.is_land[r] and q >= 3]
        if len(playsets) < 3:
            continue
        removed = int(RNG.choice(playsets))
        qty = deck.pop(removed)
        junk = int(RNG.choice(junk_pool))
        deck[junk] = qty
        exclude = near_duplicates(imp, imp.corpus[di]["rows"])
        for version in ("v1", "v2"):
            adds, _c, _p = imp.rank_adds(deck, 50, version=version,
                                         exclude=exclude)
            ranks = [r for r, _s in adds]
            rank = ranks.index(removed) + 1 if removed in ranks else None
            results[version].append(rank)
        tested += 1

    print(f"\n[{fmt}] {tested} degraded decks")
    for version, ranks in results.items():
        rr = [1 / r for r in ranks if r]
        mrr = float(np.mean(rr + [0.0] * (len(ranks) - len(rr))))
        h5 = sum(1 for r in ranks if r and r <= 5) / len(ranks)
        h12 = sum(1 for r in ranks if r and r <= 12) / len(ranks)
        print(f"  {version}: MRR {mrr:.3f}   hit@5 {h5:.0%}   hit@12 {h12:.0%}")
    return results


def freq_cache(imp):
    if not hasattr(imp, "_freqset"):
        seen = set()
        for d in imp.corpus:
            seen.update(d["rows"])
        imp._freqset = seen
    return imp._freqset


Improver._freq_cache = freq_cache


def tainted_aether_side_by_side():
    imp = Improver("legacy")
    deck_lines = {
        "Acorn Catapult": 4, "Clutch of the Undercity": 4,
        "Dimir House Guard": 4, "Echoing Truth": 4, "Forbidden Orchard": 4,
        "Hunted Horror": 3, "Hunted Phantasm": 3, "Island": 2,
        "Jwar Isle Refuge": 3, "Phyrexian Arena": 4, "Recoil": 3,
        "Sign in Blood": 2, "Swamp": 12, "Tainted Aether": 4,
        "Tainted Isle": 4,
    }
    deck = {}
    for n, q in deck_lines.items():
        deck[imp.name_to_row[norm(n)]] = q
    print("\n[Tainted Aether jank deck] top-10 suggestions:")
    for version in ("v1", "v2"):
        adds, _c, _p = imp.rank_adds(deck, 10, version=version)
        names = [imp.meta[r]["name"] for r, _s in adds]
        print(f"  {version}: {', '.join(names)}")


if __name__ == "__main__":
    for fmt in ("pauper", "modern", "legacy"):
        bench_format(fmt)
    tainted_aether_side_by_side()
