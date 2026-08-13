"""Assemble a tuned deck from a card pool.

The builder works in passes:

1. **Role quotas** - fill ramp, card draw, removal and board wipes from the
   best available cards (preferring cards that also serve the deck's theme).
2. **Theme package** - fill most remaining nonland slots with the highest
   synergy cards for the chosen theme, shaping the mana curve as it goes.
3. **Flex** - top up with the best generically useful leftovers.
4. **Mana base tuning** - land count from the finished curve using a
   Karsten-style heuristic (more lands for higher average mana value, fewer
   when the deck has plenty of cheap ramp), then nonbasic lands from the
   collection and basics split by colored pip counts.

Every selection records *why* the card was chosen, so the final list doubles
as an explanation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .carddata import Card
from .synergy import (
    THEME_NAMES,
    ThemeScore,
    filter_pool_by_colors,
    find_sixty_card_decks,
    score_themes,
)
from .tags import tag_card

BASIC_FOR_COLOR = {
    "W": "Plains", "U": "Island", "B": "Swamp", "R": "Mountain", "G": "Forest",
}

# Fraction of nonland slots we aim to have at each point on the curve.
_CURVE_TARGETS = {0: 0.08, 1: 0.10, 2: 0.24, 3: 0.22, 4: 0.16, 5: 0.11, 6: 0.09}


def _bucket(mana_value: float) -> int:
    return min(6, int(mana_value))


@dataclass
class DeckCard:
    card: Card
    count: int
    category: str
    reasons: List[str] = field(default_factory=list)


@dataclass
class Deck:
    format: str                      # "commander" or "60"
    colors: Tuple[str, ...]
    theme: str
    commander: Optional[Card] = None
    cards: List[DeckCard] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def theme_name(self) -> str:
        return THEME_NAMES.get(self.theme, self.theme)

    @property
    def total_cards(self) -> int:
        total = sum(dc.count for dc in self.cards)
        return total + (1 if self.commander else 0)

    def by_category(self) -> Dict[str, List[DeckCard]]:
        grouped: Dict[str, List[DeckCard]] = {}
        for deck_card in self.cards:
            grouped.setdefault(deck_card.category, []).append(deck_card)
        for cards in grouped.values():
            cards.sort(key=lambda dc: (dc.card.mana_value, dc.card.name))
        return grouped


class DeckBuildError(ValueError):
    pass


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _theme_weight(card: Card, theme: str) -> int:
    """How much this card contributes to `theme` (either side)."""
    weight = 0
    for tag, value in tag_card(card).items():
        if ":" not in tag:
            continue
        tag_theme, _side = tag.split(":", 1)
        if tag_theme == theme or (
            tag_theme == "tribal_any" and theme.startswith("tribal_")
        ):
            weight = max(weight, value)
        # Token producers pull their weight in sacrifice decks.
        if theme == "sacrifice" and tag == "tokens:enabler":
            weight = max(weight, 1)
    return weight


def _goodstuff_score(card: Card) -> float:
    """Generic usefulness for flex slots: cheap, tagged, evasive."""
    tags = tag_card(card)
    score = sum(tags.values()) * 0.5
    score += sum(
        1
        for k in card.keywords
        if k.lower() in {"flying", "trample", "haste", "vigilance", "deathtouch",
                         "lifelink", "menace", "first strike", "double strike"}
    ) * 0.4
    score -= max(0.0, card.mana_value - 4) * 0.5
    return score


_PIP_RE = re.compile(r"\{([WUBRG])(?:/[WUBRGP])?\}")


# ---------------------------------------------------------------------------
# The builder
# ---------------------------------------------------------------------------

class _Selection:
    """Tracks picked cards, curve buckets, and remaining availability."""

    def __init__(self, pool: Sequence[Tuple[Card, int]], singleton: bool):
        self.singleton = singleton
        self.available: Dict[str, int] = {}
        self.cards: Dict[str, Card] = {}
        for card, count in pool:
            self.cards[card.name] = card
            per_card_cap = 1 if singleton else 4
            self.available[card.name] = min(count, per_card_cap)
        self.picked: Dict[str, DeckCard] = {}
        self.buckets: Dict[int, int] = {b: 0 for b in _CURVE_TARGETS}

    @property
    def nonland_count(self) -> int:
        return sum(dc.count for dc in self.picked.values())

    def curve_penalty(self, card: Card, total_target: int) -> float:
        bucket = _bucket(card.mana_value)
        target = _CURVE_TARGETS[bucket] * total_target
        over = self.buckets[bucket] + 1 - target
        return 0.6 * over if over > 0 else 0.0

    def take(self, card: Card, category: str, reasons: List[str], copies: int = 1) -> int:
        stock = self.available.get(card.name, 0)
        copies = min(copies, stock)
        if copies <= 0:
            return 0
        self.available[card.name] = stock - copies
        if card.name in self.picked:
            self.picked[card.name].count += copies
        else:
            self.picked[card.name] = DeckCard(card, copies, category, reasons)
        self.buckets[_bucket(card.mana_value)] += copies
        return copies

    def consistency_bonus(self, card: Card, weight: float) -> float:
        """Reward cards we own multiples of: a focused 60-card deck would
        rather run a playset of a solid card than four different singles."""
        if self.singleton:
            return 0.0
        return weight * (self.available.get(card.name, 0) - 1)

    def candidates(self) -> List[Card]:
        return [
            self.cards[name]
            for name, stock in self.available.items()
            if stock > 0 and not self.cards[name].is_land
        ]


def _fill_role(
    selection: _Selection,
    role: str,
    quota: int,
    category: str,
    theme: str,
    total_nonland_target: int,
) -> None:
    while quota > 0:
        best: Optional[Tuple[float, Card, int]] = None
        for card in selection.candidates():
            tags = tag_card(card)
            role_weight = tags.get(role, 0)
            if role_weight <= 0:
                continue
            score = (
                role_weight * 2.0
                + _theme_weight(card, theme) * 0.8
                + selection.consistency_bonus(card, 0.4)
                - card.mana_value * 0.25
                - selection.curve_penalty(card, total_nonland_target)
            )
            if best is None or score > best[0]:
                best = (score, card, role_weight)
        if best is None:
            break
        _score, card, role_weight = best
        if selection.singleton:
            copies = 1
        else:
            # Take the whole usable set, even slightly past the quota:
            # a split playset (e.g. 1 of 4 owned copies) helps nobody.
            copies = 4 if role_weight >= 2 else 2
        reasons = [role.replace("_", " ")]
        theme_weight = _theme_weight(card, theme)
        if theme_weight:
            reasons.append(f"supports {THEME_NAMES.get(theme, theme)}")
        quota -= selection.take(card, category, reasons, copies)


def _fill_scored(
    selection: _Selection,
    quota: int,
    category: str,
    scorer,
    reason_for,
    total_nonland_target: int,
    consistency: float = 0.6,
) -> None:
    while quota > 0:
        best: Optional[Tuple[float, Card]] = None
        for card in selection.candidates():
            base = scorer(card)
            if base <= 0:
                continue
            score = (
                base
                + selection.consistency_bonus(card, consistency)
                - selection.curve_penalty(card, total_nonland_target)
            )
            if best is None or score > best[0]:
                best = (score, card)
        if best is None:
            break
        _score, card = best
        # Whole usable set at once (see _fill_role); trim rebalances later.
        copies = 1 if selection.singleton else 4
        quota -= selection.take(card, category, reason_for(card), copies)


def build_deck(
    pool: List[Tuple[Card, int]],
    fmt: str = "commander",
    commander: Optional[Card] = None,
    theme: Optional[str] = None,
    colors: Optional[Sequence[str]] = None,
) -> Deck:
    """Build a deck from ``pool`` (already resolved (card, owned-count) pairs)."""
    fmt = fmt.lower()
    if fmt not in ("commander", "60"):
        raise DeckBuildError(f"unknown format: {fmt!r} (expected 'commander' or '60')")

    auto_note: Optional[str] = None
    if fmt == "commander":
        if commander is None:
            raise DeckBuildError("commander format needs a commander")
        deck_colors: Tuple[str, ...] = tuple(commander.color_identity)
        deck_size = 99
        singleton = True
    else:
        if not colors:
            # No colors given: pick the best color combo (and theme, if the
            # caller didn't force one) that the collection supports.
            ideas = find_sixty_card_decks(pool, top=8)
            if theme is not None:
                ideas = [i for i in ideas if i.theme.theme == theme] or ideas
            if not ideas:
                raise DeckBuildError(
                    "could not auto-pick colors for a 60-card deck from this "
                    "pool; pass colors explicitly (e.g. WG)"
                )
            best = ideas[0]
            colors = best.colors
            if theme is None:
                theme = best.theme.theme
            auto_note = (
                f"Auto-picked {''.join(best.colors)} "
                f"{best.theme.display_name} (score {best.score:.1f})."
            )
        deck_colors = tuple(dict.fromkeys(c.upper() for c in colors))
        deck_size = 60
        singleton = False

    pool = filter_pool_by_colors(pool, deck_colors or ("C",))
    if commander is not None:
        pool = [(c, n) for c, n in pool if c.name != commander.name]

    # Pick the theme: the best-scoring viable one unless the user chose.
    if theme is None:
        ranked = score_themes(pool)
        preferred = [t for t in ranked if t.viable] or ranked
        if not preferred:
            raise DeckBuildError("no synergy themes found in this card pool")
        if commander is not None:
            commander_themes = {
                t.split(":", 1)[0] for t in tag_card(commander) if ":" in t
            }
            led = [t for t in preferred if t.theme in commander_themes]
            theme = (led[0] if led else preferred[0]).theme
        else:
            theme = preferred[0].theme

    deck = Deck(format=fmt, colors=deck_colors, theme=theme, commander=commander)
    if auto_note:
        deck.notes.append(auto_note)

    # Provisional land count; retuned from the real curve afterwards.
    lands_target = 37 if fmt == "commander" else 23
    nonland_target = deck_size - lands_target

    if fmt == "commander":
        scale = nonland_target / 62.0
        quotas = {
            "ramp": max(2, round(10 * scale)),
            "draw": max(2, round(9 * scale)),
            "removal": max(2, round(6 * scale)),
            "board_wipe": max(1, round(3 * scale)),
        }
    else:
        # Constructed 60-card decks lean on playset consistency, not a big
        # ramp package: light ramp, some draw, a real removal suite, and at
        # most a couple of sweepers.
        quotas = {
            "ramp": 3,
            "draw": 5,
            "removal": 7,
            "board_wipe": 2,
        }

    selection = _Selection(pool, singleton)
    _fill_role(selection, "ramp", quotas["ramp"], "Ramp", theme, nonland_target)
    _fill_role(selection, "draw", quotas["draw"], "Card Draw", theme, nonland_target)
    _fill_role(selection, "removal", quotas["removal"], "Removal", theme, nonland_target)
    _fill_role(selection, "board_wipe", quotas["board_wipe"], "Board Wipes", theme, nonland_target)

    theme_quota = round((nonland_target - selection.nonland_count) * 0.75)
    display = THEME_NAMES.get(theme, theme)

    def theme_reasons(card: Card) -> List[str]:
        sides = [
            tag.split(":", 1)[1]
            for tag in tag_card(card)
            if ":" in tag and tag.split(":", 1)[0] in (theme, "tribal_any")
        ]
        side = "payoff" if "payoff" in sides else "enabler"
        return [f"{display} {side}"]

    _fill_scored(
        selection, theme_quota, "Theme",
        lambda c: _theme_weight(c, theme) * 2.0 - c.mana_value * 0.15,
        theme_reasons, nonland_target,
    )
    _fill_scored(
        selection, nonland_target - selection.nonland_count, "Flex",
        _goodstuff_score, lambda c: ["solid playable"], nonland_target,
    )

    # ------------------------------------------------------------------
    # Tune the mana base from the actual curve.
    # ------------------------------------------------------------------
    picked = list(selection.picked.values())
    nonland_total = sum(dc.count for dc in picked)
    if nonland_total:
        avg_mv = sum(dc.card.mana_value * dc.count for dc in picked) / nonland_total
    else:
        avg_mv = 3.0
    cheap_ramp = sum(
        dc.count for dc in picked
        if "ramp" in tag_card(dc.card) and dc.card.mana_value <= 2
    )

    if fmt == "commander":
        lands_target = round(31.5 + 3.1 * avg_mv - 0.5 * cheap_ramp)
        lands_target = max(33, min(40, lands_target))
    else:
        lands_target = round(16.5 + 2.6 * avg_mv - 0.4 * cheap_ramp)
        lands_target = max(18, min(27, lands_target))
    deck.notes.append(
        f"Average mana value {avg_mv:.2f} with {cheap_ramp} cheap ramp sources "
        f"-> {lands_target} lands."
    )

    # Trim or extend nonlands so lands fit exactly.
    nonland_target = deck_size - lands_target
    if selection.nonland_count > nonland_target:
        removable = sorted(
            (dc for dc in selection.picked.values() if dc.category in ("Flex", "Theme")),
            key=lambda dc: (_goodstuff_score(dc.card), -dc.card.mana_value),
        )
        excess = selection.nonland_count - nonland_target
        # Cut whole singleton entries first; break up playsets only as a
        # last resort, so the deck stays focused on its multi-copy cards.
        for pass_singletons_only in (True, False):
            for deck_card in removable:
                if excess <= 0:
                    break
                if pass_singletons_only and deck_card.count != 1:
                    continue
                cut = min(excess, deck_card.count)
                deck_card.count -= cut
                excess -= cut
        selection.picked = {
            name: dc for name, dc in selection.picked.items() if dc.count > 0
        }
    elif selection.nonland_count < nonland_target:
        shortfall = nonland_target - selection.nonland_count
        _fill_scored(
            selection, shortfall, "Flex",
            lambda c: _goodstuff_score(c) + 5.0,  # take anything playable
            lambda c: ["fills out the deck"], nonland_target,
        )
        still_short = nonland_target - selection.nonland_count
        if still_short:
            lands_target += still_short
            deck.notes.append(
                f"Pool ran out of playables; {still_short} extra lands added."
            )

    # Note multi-copy picks in the explanations, from the final counts.
    if not singleton:
        for deck_card in selection.picked.values():
            if deck_card.count > 1:
                deck_card.reasons = deck_card.reasons + [
                    f"{deck_card.count} copies for consistency"
                ]

    deck.cards.extend(selection.picked.values())

    # ------------------------------------------------------------------
    # Mana base: nonbasic lands from the pool, then basics by pip count.
    # ------------------------------------------------------------------
    lands_needed = lands_target
    nonbasics = [
        (card, count)
        for card, count in pool
        if card.is_land and not card.is_basic_land
    ]

    def land_quality(card: Card) -> float:
        ours = len(set(card.produced_mana) & set(deck_colors))
        tags = tag_card(card)
        return ours * 2.0 + sum(tags.values()) * 0.3

    nonbasics.sort(key=lambda cn: land_quality(cn[0]), reverse=True)
    min_basics = max(3, round(lands_needed * (0.3 if fmt == "commander" else 0.5)))
    for card, count in nonbasics:
        allowance = lands_needed - min_basics
        if allowance <= 0:
            break
        copies = min(1 if singleton else 4, count, allowance)
        if copies > 0:
            deck.cards.append(DeckCard(card, copies, "Lands", ["mana base"]))
            lands_needed -= copies

    pips: Dict[str, int] = {c: 0 for c in deck_colors}
    for deck_card in selection.picked.values():
        for pip in _PIP_RE.findall(deck_card.card.mana_cost):
            if pip in pips:
                pips[pip] += deck_card.count
    total_pips = sum(pips.values())
    if deck_colors and total_pips:
        remaining = lands_needed
        shares = []
        for color in deck_colors:
            share = pips[color] / total_pips
            shares.append((color, share))
        counts = {c: max(1 if pips[c] else 0, round(s * lands_needed)) for c, s in shares}
        # Adjust rounding drift against the largest share.
        drift = sum(counts.values()) - remaining
        largest = max(counts, key=lambda c: counts[c]) if counts else None
        if largest is not None:
            counts[largest] = max(0, counts[largest] - drift)
        for color in deck_colors:
            n = counts.get(color, 0)
            if n > 0:
                basic = Card(
                    name=BASIC_FOR_COLOR[color],
                    type_line=f"Basic Land — {BASIC_FOR_COLOR[color]}",
                    color_identity=(color,),
                    produced_mana=(color,),
                )
                deck.cards.append(DeckCard(basic, n, "Lands", ["basic land"]))
    elif lands_needed > 0:
        wastes = Card(name="Wastes", type_line="Basic Land", produced_mana=("C",))
        deck.cards.append(DeckCard(wastes, lands_needed, "Lands", ["basic land"]))

    return deck
