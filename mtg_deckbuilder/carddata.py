"""Card data model and Scryfall bulk-data loading.

The card database is expected in Scryfall "oracle cards" bulk format:
a JSON array of card objects (https://scryfall.com/docs/api/bulk-data).
Only a handful of fields are used, so trimmed-down files work too.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Multi-face layouts where the faces carry the rules text.
_FACED_LAYOUTS = {
    "transform", "modal_dfc", "adventure", "split", "flip", "battle", "prototype",
}


@dataclass
class Card:
    name: str
    mana_cost: str = ""
    mana_value: float = 0.0
    colors: Tuple[str, ...] = ()
    color_identity: Tuple[str, ...] = ()
    type_line: str = ""
    oracle_text: str = ""
    keywords: Tuple[str, ...] = ()
    power: Optional[str] = None
    toughness: Optional[str] = None
    produced_mana: Tuple[str, ...] = ()
    rarity: str = ""
    tags: Dict[str, int] = field(default_factory=dict)

    # ---- derived helpers -------------------------------------------------

    @property
    def front_name(self) -> str:
        return self.name.split(" // ")[0].strip()

    @property
    def types(self) -> List[str]:
        """Card types of every face (words left of the em-dash)."""
        found: List[str] = []
        for part in self.type_line.split(" // "):
            left = part.split("—")[0]
            found.extend(left.split())
        return found

    @property
    def subtypes(self) -> List[str]:
        found: List[str] = []
        for part in self.type_line.split(" // "):
            pieces = part.split("—")
            if len(pieces) > 1:
                found.extend(pieces[1].split())
        return found

    @property
    def is_land(self) -> bool:
        return "Land" in self.types

    @property
    def is_creature(self) -> bool:
        return "Creature" in self.types

    @property
    def is_basic_land(self) -> bool:
        return "Basic" in self.types and self.is_land

    @property
    def is_legendary_creature(self) -> bool:
        # A card can lead a Commander deck if it is a legendary creature or
        # explicitly says it can be your commander (planeswalker commanders).
        if "can be your commander" in self.oracle_text.lower():
            return True
        return "Legendary" in self.types and self.is_creature

    @property
    def numeric_power(self) -> Optional[int]:
        try:
            return int(self.power) if self.power is not None else None
        except ValueError:
            return None

    def fits_identity(self, colors: Iterable[str]) -> bool:
        return set(self.color_identity) <= set(colors)

    # ---- construction ----------------------------------------------------

    @classmethod
    def from_scryfall(cls, d: dict) -> "Card":
        oracle = d.get("oracle_text")
        type_line = d.get("type_line")
        mana_cost = d.get("mana_cost")
        power = d.get("power")
        toughness = d.get("toughness")

        faces = d.get("card_faces") or []
        if faces and (oracle is None or d.get("layout") in _FACED_LAYOUTS):
            oracle = "\n//\n".join(f.get("oracle_text", "") for f in faces)
            type_line = type_line or " // ".join(
                f.get("type_line", "") for f in faces
            )
            mana_cost = mana_cost or faces[0].get("mana_cost", "")
            power = power or faces[0].get("power")
            toughness = toughness or faces[0].get("toughness")

        return cls(
            name=d.get("name", ""),
            mana_cost=mana_cost or "",
            mana_value=float(d.get("cmc") or 0.0),
            colors=tuple(d.get("colors") or ()),
            color_identity=tuple(d.get("color_identity") or ()),
            type_line=type_line or "",
            oracle_text=oracle or "",
            keywords=tuple(d.get("keywords") or ()),
            power=power,
            toughness=toughness,
            produced_mana=tuple(d.get("produced_mana") or ()),
            rarity=d.get("rarity", ""),
        )


def _norm(name: str) -> str:
    return re.sub(r"\s+", " ", name).strip().lower()


class CardDatabase:
    """Name-indexed card lookup over a Scryfall bulk file."""

    def __init__(self, cards: Iterable[Card]):
        self.by_name: Dict[str, Card] = {}
        self.cards: List[Card] = []
        for card in cards:
            self.cards.append(card)
            self.by_name.setdefault(_norm(card.name), card)
            # Index double-faced / split cards by their front face too, since
            # collection exports often list only the front name.
            if " // " in card.name:
                self.by_name.setdefault(_norm(card.front_name), card)

    @classmethod
    def load(cls, path: Path) -> "CardDatabase":
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        cards = []
        seen = set()
        for entry in raw:
            # Skip non-playable layouts (tokens, art cards, ...)
            if entry.get("layout") in {"token", "double_faced_token", "art_series", "emblem"}:
                continue
            name = entry.get("name", "")
            if name in seen:
                continue
            seen.add(name)
            cards.append(Card.from_scryfall(entry))
        return cls(cards)

    def get(self, name: str) -> Optional[Card]:
        return self.by_name.get(_norm(name))

    def __len__(self) -> int:
        return len(self.cards)
