"""Command-line interface for mtg_deckbuilder.

    mtgdeck fetch-data
    mtgdeck analyze  my_collection.txt
    mtgdeck suggest  my_collection.txt [--format 60|commander]
    mtgdeck build    my_collection.txt                    # best 60-card deck
    mtgdeck build    my_collection.txt --colors BR --theme sacrifice
    mtgdeck build    my_collection.txt --commander "Krenko, Mob Boss"
    mtgdeck card     "Skullclamp"
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from .carddata import Card, CardDatabase
from .collection import load_collection
from .deckbuilder import DeckBuildError, build_deck
from .fetch import (
    DEFAULT_DATA_PATH,
    LEGACY_DATA_PATH,
    MANUAL_INSTRUCTIONS,
    FetchError,
    fetch_bulk_data,
)
from .synergy import (
    THEME_NAMES,
    filter_pauper,
    find_commander_decks,
    find_sixty_card_decks,
    pretend_playsets,
    resolve_pool,
    score_themes,
)
from .tags import tag_card

SAMPLE_DATA = Path(__file__).resolve().parent.parent / "examples" / "sample_cards.json"


def _data_path(arg: Optional[str]) -> Path:
    if arg:
        return Path(arg)
    env = os.environ.get("MTGDECK_DATA")
    if env:
        return Path(env)
    if DEFAULT_DATA_PATH.exists():
        return DEFAULT_DATA_PATH
    if LEGACY_DATA_PATH.exists():
        return LEGACY_DATA_PATH
    if SAMPLE_DATA.exists():
        print(
            "note: using the small bundled sample database "
            "(run `mtgdeck fetch-data` for the full card catalog)\n",
            file=sys.stderr,
        )
        return SAMPLE_DATA
    raise SystemExit(
        "No card database found. Run `mtgdeck fetch-data`, or:\n\n"
        + MANUAL_INSTRUCTIONS
    )


def _load(args) -> Tuple[CardDatabase, List[Tuple[Card, int]]]:
    db = CardDatabase.load(_data_path(args.data))
    collection = load_collection(Path(args.collection))
    pool, missing = resolve_pool(collection, db)
    if missing:
        print(
            f"warning: {len(missing)} name(s) not found in the card database: "
            + ", ".join(missing[:10])
            + (" ..." if len(missing) > 10 else ""),
            file=sys.stderr,
        )
    if not pool:
        raise SystemExit("No cards from the collection were found in the database.")
    return db, pool


def _color_label(colors) -> str:
    chars = [c for c in "WUBRG" if c in colors]
    return "".join(chars) or "C"


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_fetch_data(args) -> int:
    dest = Path(args.out) if args.out else DEFAULT_DATA_PATH
    print(f"Downloading Scryfall oracle-cards bulk data to {dest} ...")
    try:
        path = fetch_bulk_data(dest)
    except FetchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Done: {path} ({path.stat().st_size / 1e6:.0f} MB)")
    return 0


def cmd_analyze(args) -> int:
    _db, pool = _load(args)
    total = sum(n for _c, n in pool)
    print(f"Collection: {total} cards, {len(pool)} unique\n")

    role_counts = {}
    for card, count in pool:
        for tag, _w in tag_card(card).items():
            if ":" not in tag:
                role_counts.setdefault(tag, [0, []])
                role_counts[tag][0] += count
                role_counts[tag][1].append(card.name)

    print("Functional roles in your collection:")
    for role, (count, names) in sorted(role_counts.items(), key=lambda kv: -kv[1][0]):
        example = ", ".join(names[:4])
        print(f"  {role:<15} {count:>4} cards   e.g. {example}")

    print("\nSynergy themes (score = balance of enablers and payoffs):")
    for theme in score_themes(pool)[: args.top]:
        marker = "*" if theme.viable else " "
        print(
            f" {marker} {theme.display_name:<28} score {theme.score:6.1f}   "
            f"enablers {theme.enabler_weight:5.1f} / payoffs {theme.payoff_weight:5.1f}"
        )
        examples = [c.name for c, _w in (theme.payoffs[:3] + theme.enablers[:3])]
        print(f"      key cards: {', '.join(dict.fromkeys(examples))}")
    print("\n  * = enough cards on both sides to build around")
    return 0


def cmd_suggest(args) -> int:
    _db, pool = _load(args)
    if args.pauper:
        pool = filter_pauper(pool)
    if args.playsets:
        pool = pretend_playsets(pool)
    if args.format == "commander":
        ideas = find_commander_decks(pool, top=args.top)
        if not ideas:
            print(
                "No viable commander decks found - no legendary creature in the\n"
                "collection has enough synergy support in its color identity.\n"
                "Try `mtgdeck suggest --format 60` for 60-card ideas."
            )
            return 0
        print("Decks hiding in your collection (Commander):\n")
        for i, idea in enumerate(ideas, 1):
            print(
                f"{i}. {idea.commander.name}  [{idea.color_label}]  "
                f"- {idea.theme.display_name}  (score {idea.score:.1f})"
            )
            payoffs = ", ".join(c.name for c, _w in idea.theme.payoffs[:4])
            enablers = ", ".join(c.name for c, _w in idea.theme.enablers[:4])
            print(f"     payoffs:  {payoffs}")
            print(f"     enablers: {enablers}")
            support = idea.support
            print(
                f"     support:  {support['ramp']} ramp, {support['draw']} draw, "
                f"{support['removal']} removal, {support['board_wipe']} wipes\n"
            )
        best = ideas[0]
        print(
            "Build the top idea with:\n"
            f'  mtgdeck build {args.collection} --commander "{best.commander.name}"'
        )
    else:
        ideas = find_sixty_card_decks(pool, top=args.top)
        if not ideas:
            print("No viable 60-card decks found in this collection.")
            return 0
        print("Decks hiding in your collection (60-card casual):\n")
        for i, idea in enumerate(ideas, 1):
            print(
                f"{i}. [{idea.color_label}] {idea.theme.display_name}  "
                f"(score {idea.score:.1f})"
            )
            examples = [c.name for c, _w in (idea.theme.payoffs[:3] + idea.theme.enablers[:3])]
            print(f"     key cards: {', '.join(dict.fromkeys(examples))}\n")
        best = ideas[0]
        print(
            "Build the top idea with:\n"
            f"  mtgdeck build {args.collection} --format 60 "
            f"--colors {_color_label(best.colors)} --theme {best.theme.theme}"
        )
    return 0


def cmd_build(args) -> int:
    db, pool = _load(args)
    if args.pauper:
        pool = filter_pauper(pool)
    if args.playsets:
        pool = pretend_playsets(pool)

    commander = None
    if args.commander:
        commander = db.get(args.commander)
        if commander is None:
            raise SystemExit(f"Commander {args.commander!r} not found in card database.")
        owned = {c.name for c, _n in pool}
        if commander.name not in owned:
            print(
                f"note: {commander.name} is not in your collection file; "
                "building around it anyway.",
                file=sys.stderr,
            )
        if not commander.is_legendary_creature:
            raise SystemExit(f"{commander.name} cannot be a commander.")

    fmt = args.format or ("commander" if commander else "60")
    try:
        deck = build_deck(
            pool,
            fmt=fmt,
            commander=commander,
            theme=args.theme,
            colors=list(args.colors) if args.colors else None,
        )
    except DeckBuildError as exc:
        raise SystemExit(f"error: {exc}")

    label = "Commander" if deck.format == "commander" else "60-card"
    print(f"# {label} deck - {deck.theme_name}  [{_color_label(deck.colors)}]")
    if deck.commander:
        print(f"\n## Commander\n1 {deck.commander.name}")

    order = ["Ramp", "Card Draw", "Removal", "Board Wipes", "Theme", "Threats", "Flex", "Lands"]
    grouped = deck.by_category()
    for category in order:
        if category not in grouped:
            continue
        cards = grouped[category]
        count = sum(dc.count for dc in cards)
        print(f"\n## {category} ({count})")
        for deck_card in cards:
            reason = "; ".join(deck_card.reasons)
            explain = f"   # {reason}" if args.explain and reason else ""
            print(f"{deck_card.count} {deck_card.card.name}{explain}")

    print(f"\n# Total: {deck.total_cards} cards")
    for note in deck.notes:
        print(f"# Tuning: {note}")
    return 0


def cmd_card(args) -> int:
    db = CardDatabase.load(_data_path(args.data))
    card = db.get(args.name)
    if card is None:
        raise SystemExit(f"Card {args.name!r} not found.")
    print(f"{card.name}  {card.mana_cost}  (MV {card.mana_value:g})")
    print(card.type_line)
    if card.oracle_text:
        print()
        print(card.oracle_text)
    tags = tag_card(card)
    print("\nExtracted tags:")
    if not tags:
        print("  (none)")
    for tag, weight in sorted(tags.items(), key=lambda kv: -kv[1]):
        if ":" in tag:
            theme, side = tag.split(":", 1)
            label = f"{THEME_NAMES.get(theme, theme)} - {side}"
        else:
            label = tag
        print(f"  {label:<40} weight {weight}")
    return 0


# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mtgdeck",
        description="Find the decks hiding in your Magic: The Gathering collection.",
    )
    parser.add_argument(
        "--data",
        help="path to Scryfall oracle-cards bulk JSON (default: "
        "$MTGDECK_DATA, then ~/.cache/mtg_deckbuilder/oracle-cards.json, "
        "then the bundled sample)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch-data", help="download Scryfall bulk card data")
    p.add_argument("--out", help="where to save the file")
    p.set_defaults(func=cmd_fetch_data)

    p = sub.add_parser("analyze", help="extract keywords/roles and rank synergy themes")
    p.add_argument("collection", help="collection file (.txt or .csv)")
    p.add_argument("--top", type=int, default=12, help="themes to show")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("suggest", help="find complete decks hiding in the collection")
    p.add_argument("collection")
    p.add_argument("--format", choices=["60", "commander"], default="60")
    p.add_argument("--top", type=int, default=8)
    p.add_argument(
        "--playsets", action="store_true",
        help="pretend you own 4 of every card (proxy-friendly)",
    )
    p.add_argument(
        "--pauper", action="store_true",
        help="commons only (Pauper-legal cards)",
    )
    p.set_defaults(func=cmd_suggest)

    p = sub.add_parser("build", help="build and tune a deck from the collection")
    p.add_argument("collection")
    p.add_argument("--commander", help="commander name (implies --format commander)")
    p.add_argument("--format", choices=["60", "commander"], help="default: 60")
    p.add_argument(
        "--colors",
        help="colors for 60-card decks, e.g. WG or ubr "
        "(default: auto-pick the best combo for the collection)",
    )
    p.add_argument("--theme", help="force a theme key (see `analyze`)")
    p.add_argument(
        "--playsets", action="store_true",
        help="pretend you own 4 of every card (proxy-friendly)",
    )
    p.add_argument(
        "--pauper", action="store_true",
        help="commons only (Pauper-legal cards)",
    )
    p.add_argument(
        "--explain", action="store_true",
        help="annotate every card with why it was picked",
    )
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("card", help="show the tags extracted from one card")
    p.add_argument("name")
    p.set_defaults(func=cmd_card)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
