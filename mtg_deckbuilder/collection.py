"""Parse a card collection from plain-text or CSV exports.

Supported inputs:

* Plain text, one card per line::

      4 Llanowar Elves
      2x Sol Ring
      Krenko, Mob Boss        # a bare name means quantity 1
      // comments and blank lines are ignored

* CSV with a header row. The name column may be called Name, Card,
  or Card Name; the quantity column Count, Quantity, Qty, or Amount
  (this covers ManaBox, Moxfield, Archidekt and Deckbox exports).
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

_LINE_RE = re.compile(r"^\s*(?:(\d+)\s*[xX]?\s+)?(.+?)\s*$")
_COMMENT_RE = re.compile(r"^\s*(#|//)")

_NAME_COLUMNS = ("name", "card name", "card")
_COUNT_COLUMNS = ("count", "quantity", "qty", "amount")


@dataclass
class Collection:
    """A multiset of card names (as written by the user)."""

    counts: Dict[str, int] = field(default_factory=dict)

    def add(self, name: str, count: int = 1) -> None:
        name = re.sub(r"\s+", " ", name).strip()
        if name:
            self.counts[name] = self.counts.get(name, 0) + count

    def items(self) -> Iterator[Tuple[str, int]]:
        return iter(self.counts.items())

    @property
    def total_cards(self) -> int:
        return sum(self.counts.values())

    @property
    def unique_cards(self) -> int:
        return len(self.counts)


def _looks_like_csv(lines: List[str]) -> bool:
    if not lines:
        return False
    header = lines[0].lower()
    return "," in header and any(col in header for col in _NAME_COLUMNS)


def _parse_csv(lines: List[str]) -> Collection:
    collection = Collection()
    reader = csv.DictReader(lines)
    fields = {(f or "").strip().lower(): f for f in (reader.fieldnames or [])}

    name_key = next((fields[c] for c in _NAME_COLUMNS if c in fields), None)
    count_key = next((fields[c] for c in _COUNT_COLUMNS if c in fields), None)
    if name_key is None:
        raise ValueError(
            "CSV has no recognizable name column (expected one of: %s)"
            % ", ".join(_NAME_COLUMNS)
        )

    for row in reader:
        name = (row.get(name_key) or "").strip()
        if not name:
            continue
        count = 1
        if count_key:
            try:
                count = max(1, int(float(row.get(count_key) or 1)))
            except ValueError:
                count = 1
        collection.add(name, count)
    return collection


def _parse_text(lines: List[str]) -> Collection:
    collection = Collection()
    for line in lines:
        if not line.strip() or _COMMENT_RE.match(line):
            continue
        match = _LINE_RE.match(line)
        if not match:
            continue
        count = int(match.group(1)) if match.group(1) else 1
        collection.add(match.group(2), count)
    return collection


def parse_collection(text: str, assume_csv: bool = False) -> Collection:
    """Parse collection text (plain list or CSV export) into a Collection."""
    lines = text.splitlines()
    if assume_csv or _looks_like_csv(lines):
        return _parse_csv(lines)
    return _parse_text(lines)


def load_collection(path: Path) -> Collection:
    with open(path, encoding="utf-8-sig") as fh:
        text = fh.read()
    return parse_collection(text, assume_csv=path.suffix.lower() == ".csv")
