"""JSON-in/JSON-out interface for the web frontend.

The website runs this package in the browser via Pyodide.  The JavaScript
side parses nothing itself: it sends the raw collection text plus the
Scryfall card objects it fetched for those names, and gets back a plain
JSON document ready to render.

Every entry point is a single string-to-string function so no Pyodide
proxy objects cross the boundary.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

from .carddata import Card, CardDatabase
from .collection import parse_collection
from .deckbuilder import Deck, DeckBuildError, build_deck
from .synergy import (
    THEME_NAMES,
    find_commander_decks,
    find_sixty_card_decks,
    pretend_playsets,
    resolve_pool,
    score_themes,
)
from .tags import tag_card

CATEGORY_ORDER = ["Ramp", "Card Draw", "Removal", "Board Wipes", "Theme", "Flex", "Lands"]


def _color_label(colors) -> str:
    chars = [c for c in "WUBRG" if c in colors]
    return "".join(chars) or "C"


def collection_names(text: str) -> str:
    """Names (as written) in a pasted collection, as a JSON array."""
    return json.dumps([name for name, _count in parse_collection(text).items()])


def _pool(
    text: str, entries: List[dict], playsets: bool = False
) -> Tuple[CardDatabase, List[Tuple[Card, int]], List[str]]:
    db = CardDatabase.from_entries(entries)
    pool, missing = resolve_pool(parse_collection(text), db)
    if playsets:
        pool = pretend_playsets(pool)
    return db, pool, missing


def _theme_meta() -> List[Dict]:
    return [{"key": key, "name": name} for key, name in sorted(
        THEME_NAMES.items(), key=lambda kv: kv[1]
    )]


def _serialize_deck(deck: Deck, missing: List[str]) -> Dict:
    grouped = deck.by_category()
    categories = []
    for name in CATEGORY_ORDER:
        if name not in grouped:
            continue
        cards = [
            {
                "count": dc.count,
                "name": dc.card.name,
                "mana_cost": dc.card.mana_cost,
                "mana_value": dc.card.mana_value,
                "type_line": dc.card.type_line,
                "reasons": dc.reasons,
            }
            for dc in grouped[name]
        ]
        categories.append(
            {"name": name, "count": sum(c["count"] for c in cards), "cards": cards}
        )

    lines = []
    if deck.commander:
        lines.append(f"1 {deck.commander.name}")
    for category in categories:
        for card in category["cards"]:
            lines.append(f"{card['count']} {card['name']}")

    return {
        "format": deck.format,
        "color_label": _color_label(deck.colors),
        "theme_key": deck.theme,
        "theme_name": deck.theme_name,
        "commander": deck.commander.name if deck.commander else None,
        "total": deck.total_cards,
        "notes": deck.notes,
        "missing": missing,
        "categories": categories,
        "export_text": "\n".join(lines),
    }


def _do_analyze(req: Dict) -> Dict:
    _db, pool, missing = _pool(req["collection"], req["cards"])
    total = sum(n for _c, n in pool)

    role_counts: Dict[str, Dict] = {}
    for card, count in pool:
        for tag, _weight in tag_card(card).items():
            if ":" in tag:
                continue
            entry = role_counts.setdefault(tag, {"role": tag, "count": 0, "examples": []})
            entry["count"] += count
            if len(entry["examples"]) < 4:
                entry["examples"].append(card.name)
    roles = sorted(role_counts.values(), key=lambda r: -r["count"])

    themes = []
    for theme in score_themes(pool)[: req.get("top", 12)]:
        examples = [c.name for c, _w in (theme.payoffs[:3] + theme.enablers[:3])]
        themes.append(
            {
                "key": theme.theme,
                "name": theme.display_name,
                "score": round(theme.score, 1),
                "viable": theme.viable,
                "enabler_weight": round(theme.enabler_weight, 1),
                "payoff_weight": round(theme.payoff_weight, 1),
                "examples": list(dict.fromkeys(examples)),
            }
        )
    return {
        "total_cards": total,
        "unique_cards": len(pool),
        "roles": roles,
        "themes": themes,
        "missing": missing,
    }


def _do_suggest(req: Dict) -> Dict:
    _db, pool, missing = _pool(
        req["collection"], req["cards"], bool(req.get("playsets"))
    )
    fmt = req.get("format", "60")
    top = req.get("top", 8)
    if fmt == "commander":
        ideas = find_commander_decks(pool, top=top)
    else:
        ideas = find_sixty_card_decks(pool, top=top)
    out = []
    for idea in ideas:
        examples = [
            c.name for c, _w in (idea.theme.payoffs[:4] + idea.theme.enablers[:4])
        ]
        out.append(
            {
                "commander": idea.commander.name if idea.commander else None,
                "colors": list(idea.colors),
                "color_label": idea.color_label,
                "theme_key": idea.theme.theme,
                "theme_name": idea.theme.display_name,
                "score": round(idea.score, 1),
                "key_cards": list(dict.fromkeys(examples))[:6],
                "support": idea.support,
            }
        )
    return {"format": fmt, "ideas": out, "missing": missing}


def _do_build(req: Dict) -> Dict:
    db, pool, missing = _pool(
        req["collection"], req["cards"], bool(req.get("playsets"))
    )

    commander = None
    commander_name = (req.get("commander") or "").strip()
    if commander_name:
        commander = db.get(commander_name)
        if commander is None:
            raise DeckBuildError(
                f"commander {commander_name!r} was not found on Scryfall - "
                "check the spelling, or add it to the collection list"
            )
        if not commander.is_legendary_creature:
            raise DeckBuildError(f"{commander.name} cannot be a commander")

    fmt = req.get("format") or ("commander" if commander else "60")
    colors: Optional[List[str]] = None
    if req.get("colors"):
        colors = [c.upper() for c in str(req["colors"]) if c.upper() in "WUBRG"]
        if not colors:
            raise DeckBuildError(
                f"could not read any colors from {req['colors']!r} (use letters WUBRG)"
            )
    deck = build_deck(
        pool,
        fmt=fmt,
        commander=commander,
        theme=req.get("theme") or None,
        colors=colors,
    )
    return _serialize_deck(deck, missing)


_ACTIONS = {
    "meta": lambda req: {"themes": _theme_meta()},
    "analyze": _do_analyze,
    "suggest": _do_suggest,
    "build": _do_build,
}


def run_json(payload: str) -> str:
    """Dispatch one request: ``{"action": ..., "collection": ..., "cards": [...]}``."""
    try:
        req = json.loads(payload)
        action = _ACTIONS.get(req.get("action"))
        if action is None:
            raise ValueError(f"unknown action {req.get('action')!r}")
        if req.get("action") != "meta" and not req.get("cards"):
            raise DeckBuildError(
                "none of the collection's cards could be found on Scryfall"
            )
        return json.dumps({"ok": True, "result": action(req)})
    except DeckBuildError as exc:
        return json.dumps({"ok": False, "error": str(exc)})
    except Exception as exc:  # surfaced in the UI rather than a broken page
        return json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
