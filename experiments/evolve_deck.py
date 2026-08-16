"""Evolve a deck toward a local maximum: proposer + Forge racing, iterated.

The piece the improve-loop was missing: instead of one round of
candidate swaps, run a generational hill-climb -

  each generation:
    1. the proposer (auto: archetype/brew routing, backed by the
       co-occurrence spaces and the trained pair-synergy model) ranks
       candidate swaps against the current incumbent
    2. stage 1 racing: every variant plays a few games vs the gauntlet
    3. stage 2: the top finalists play more games vs the gauntlet
    4. accept gate: the best finalist plays the incumbent head-to-head;
       it becomes the new incumbent only if it clears --accept
  stop after --generations, or when --patience consecutive generations
  produce no accepted challenger (local maximum reached).

Honest-methodology notes (see FORGE-NOTES.md): sim winrates at these
budgets are a SCREEN, not a fine ranker; the h2h accept gate is what
keeps noise from walking the deck downhill. The AI-pilot bias caveat
applies - the loop optimizes "what Forge's AI wins with", which
undervalues precise burn math and activated-ability engines.

State is journaled to output/evolve-<name>.json after every generation;
rerun with the same --name to resume. The evolved list is written to
output/evolved-<name>.txt.

Example (overnight budget, ~10 min/generation on this box):
  python3 experiments/evolve_deck.py mydeck.txt --format legacy \
      --name tainted --generations 12 --lock "Tainted Aether"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from improve_deck import (Improver, norm, write_dck, run_match,  # noqa: E402
                          evaluate, ci95)
from mtg_deckbuilder.collection import parse_collection  # noqa: E402

OUT = Path(__file__).resolve().parent / "output"


def deck_to_pairs(imp, deck):
    return sorted((imp.meta[r]["name"], q) for r, q in deck.items())


def apply_swap(deck, cut, add, qty):
    new = dict(deck)
    take = min(qty, new[cut])
    new[cut] -= take
    if new[cut] <= 0:
        del new[cut]
    new[add] = min(4, new.get(add, 0) + take)
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("deck", help="decklist file")
    ap.add_argument("--format", default="legacy")
    ap.add_argument("--name", default="run", help="state/deck name key")
    ap.add_argument("--generations", type=int, default=8)
    ap.add_argument("--candidates", type=int, default=6,
                    help="variants raced per generation")
    ap.add_argument("--gauntlet", type=int, default=3)
    ap.add_argument("--stage1", type=int, default=4,
                    help="games per gauntlet deck per variant, stage 1")
    ap.add_argument("--stage2", type=int, default=10)
    ap.add_argument("--finalists", type=int, default=2)
    ap.add_argument("--h2h", type=int, default=40,
                    help="accept-gate games vs incumbent")
    ap.add_argument("--accept", type=float, default=0.55,
                    help="h2h winrate the challenger must clear")
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--lock", action="append", default=[])
    ap.add_argument("--proposer", choices=["v1", "v2", "brew", "auto"],
                    default="auto")
    args = ap.parse_args()

    imp = Improver(args.format)
    state_path = OUT / f"evolve-{args.name}.json"

    if state_path.exists():
        state = json.loads(state_path.read_text())
        deck = {imp.name_to_row[norm(n)]: q for n, q in state["incumbent"]}
        print(f"resuming {args.name} at generation {state['generation']}"
              f" ({len(state['history'])} events)", flush=True)
    else:
        coll = parse_collection(Path(args.deck).read_text(encoding="utf-8"))
        deck = {}
        for name, qty in coll.items():
            row = imp.name_to_row.get(norm(name))
            if row is None:
                print(f"UNMATCHED: {name} - aborting")
                return
            deck[row] = deck.get(row, 0) + qty
            if not imp.playable(row):
                print(f"NOT FORGE-PLAYABLE: {name} - aborting")
                return
        gauntlet_idx = imp.gauntlet(args.gauntlet)
        state = {
            "format": args.format,
            "incumbent": deck_to_pairs(imp, deck),
            "generation": 0,
            "history": [],
            "tried": [],
            # store the decklists, not corpus indices, so a resumed run
            # keeps the identical gauntlet even if the corpus changes
            "gauntlet": [imp.corpus[i]["main"] for i in gauntlet_idx],
        }

    for gi, g in enumerate(state["gauntlet"]):
        write_dck(f"evo_gauntlet_{gi}", g)
    gauntlet_names = [f"evo_gauntlet_{gi}"
                      for gi in range(len(state["gauntlet"]))]
    locked = {norm(n) for n in args.lock}
    tried = {tuple(t) for t in state["tried"]}
    stale = 0

    def save():
        state["incumbent"] = deck_to_pairs(imp, deck)
        state["tried"] = sorted(tried)
        state_path.write_text(json.dumps(state, indent=1))
        (OUT / f"evolved-{args.name}.txt").write_text(
            "\n".join(f"{q} {n}" for n, q in state["incumbent"]) + "\n")

    while state["generation"] < args.generations and stale < args.patience:
        gen = state["generation"] + 1
        print(f"\n=== generation {gen} ===", flush=True)

        swaps, _presence = imp.propose(deck, args.candidates * 2, locked,
                                       version=args.proposer)
        swaps = [(c, a, q) for c, a, q in swaps
                 if (imp.meta[c]["name"], imp.meta[a]["name"]) not in tried
                 and imp.playable(a)][:args.candidates]
        if not swaps:
            print("proposer exhausted - stopping", flush=True)
            break

        variants = {}
        write_dck("evo_base", deck_to_pairs(imp, deck))
        for vi, (cut, add, qty) in enumerate(swaps):
            vname = f"evo_cand_{vi}"
            variants[vname] = (cut, add, qty)
            write_dck(vname, deck_to_pairs(imp, apply_swap(deck, cut, add, qty)))
            print(f"  {vname}: -{qty} {imp.meta[cut]['name']}"
                  f" +{qty} {imp.meta[add]['name']}", flush=True)

        r1 = evaluate(list(variants) + ["evo_base"], gauntlet_names,
                      args.stage1)
        ranked = sorted(variants, key=lambda v: -(r1[v][0] / max(1, r1[v][1])))
        finalists = ranked[:args.finalists]
        print(f"stage1: " + ", ".join(
            f"{v} {r1[v][0]}/{r1[v][1]}" for v in ranked), flush=True)

        r2 = evaluate(finalists + ["evo_base"], gauntlet_names, args.stage2)
        best = max(finalists, key=lambda v: r2[v][0] / max(1, r2[v][1]))
        bp, bh = ci95(*r2[best])
        ip, ih = ci95(*r2["evo_base"])
        print(f"stage2: best {best} {bp:.0%}±{bh:.0%}"
              f" vs incumbent {ip:.0%}±{ih:.0%}", flush=True)

        w, n = run_match(best, "evo_base", args.h2h, 90 + 30 * args.h2h)
        p, half = ci95(w, n)
        cut, add, qty = variants[best]
        desc = (f"-{qty} {imp.meta[cut]['name']}"
                f" +{qty} {imp.meta[add]['name']}")
        print(f"accept gate: {desc} h2h {w}/{n} = {p:.0%}±{half:.0%}",
              flush=True)

        event = {"gen": gen, "swap": desc, "h2h": [w, n],
                 "stage2": {v: r2[v] for v in r2}}
        for cut2, add2, _q in swaps:
            tried.add((imp.meta[cut2]["name"], imp.meta[add2]["name"]))
        if n >= args.h2h // 2 and p >= args.accept:
            deck = apply_swap(deck, cut, add, qty)
            event["accepted"] = True
            stale = 0
            print(f"ACCEPTED -> incumbent updated", flush=True)
        else:
            event["accepted"] = False
            stale += 1
            print(f"rejected ({stale}/{args.patience} stale)", flush=True)

        state["generation"] = gen
        state["history"].append(event)
        save()

    save()
    print(f"\ndone after generation {state['generation']}"
          f" ({sum(1 for e in state['history'] if e['accepted'])} swaps"
          f" accepted). Final list: output/evolved-{args.name}.txt",
          flush=True)


if __name__ == "__main__":
    main()
