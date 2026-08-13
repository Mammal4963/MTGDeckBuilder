"""Score synergy themes in a card pool and find the decks hiding in it.

A *theme* is a named synergy package with two sides: enablers produce a
resource and payoffs reward you for it.  A pool scores well in a theme only
when it has a healthy amount of **both** sides - twenty token producers with
no payoffs is not a deck, and the geometric mean in `ThemeScore` captures
that: the score is ``2 * sqrt(enablers * payoffs)``, which is maximized when
the two sides are balanced.

Cross-feeds encode synergies *between* themes (token producers are also
sacrifice fodder, treasures feed artifact payoffs, ...).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from .carddata import Card, CardDatabase
from .collection import Collection
from .tags import TRIBAL_TYPES, tag_card

THEME_NAMES: Dict[str, str] = {
    "tokens": "Go-wide tokens",
    "sacrifice": "Sacrifice / aristocrats",
    "counters": "+1/+1 counters",
    "graveyard": "Graveyard value",
    "lifegain": "Lifegain",
    "spells": "Spellslinger",
    "artifacts": "Artifacts matter",
    "enchantments": "Enchantments matter",
    "lands": "Lands matter / landfall",
    "blink": "Blink / flicker",
    "voltron": "Voltron (equipment & auras)",
    "mill": "Mill",
    "discard": "Discard",
    "power": "Big power matters",
}
for _t in TRIBAL_TYPES:
    THEME_NAMES[f"tribal_{_t}"] = f"{_t.capitalize()} tribal"

# (source_theme, side) -> [(target_theme, side, factor)]
# "A token producer is 60% of a sacrifice enabler", etc.
CROSS_FEEDS: Dict[Tuple[str, str], List[Tuple[str, str, float]]] = {
    ("tokens", "enabler"): [("sacrifice", "enabler", 0.6)],
    ("graveyard", "enabler"): [("spells", "enabler", 0.0)],  # placeholder, no-op
    ("artifacts", "enabler"): [],
    ("spells", "enabler"): [("graveyard", "enabler", 0.3)],  # spells fill the yard
    ("blink", "payoff"): [("tokens", "payoff", 0.0)],
}


@dataclass
class ThemeScore:
    theme: str
    enabler_weight: float = 0.0
    payoff_weight: float = 0.0
    enablers: List[Tuple[Card, int]] = field(default_factory=list)
    payoffs: List[Tuple[Card, int]] = field(default_factory=list)

    @property
    def score(self) -> float:
        return 2.0 * math.sqrt(self.enabler_weight * self.payoff_weight)

    @property
    def viable(self) -> bool:
        # A playable package needs several cards on each side.
        return self.enabler_weight >= 6 and self.payoff_weight >= 6

    @property
    def display_name(self) -> str:
        return THEME_NAMES.get(self.theme, self.theme)


def resolve_pool(collection: Collection, db: CardDatabase) -> Tuple[List[Tuple[Card, int]], List[str]]:
    """Match collection names against the database.

    Returns ``([(card, count), ...], [unmatched names])``.
    """
    pool: List[Tuple[Card, int]] = []
    missing: List[str] = []
    for name, count in sorted(collection.items()):
        card = db.get(name)
        if card is None:
            missing.append(name)
        else:
            pool.append((card, count))
    return pool, missing


def pretend_playsets(
    pool: List[Tuple[Card, int]]
) -> List[Tuple[Card, int]]:
    """Treat every card as at least a playset, as if proxies were allowed.

    Basic lands keep their real count (the builder adds basics freely anyway).
    """
    return [
        (card, count if card.is_basic_land else max(count, 4))
        for card, count in pool
    ]


def _theme_contributions(card: Card) -> Dict[Tuple[str, str], float]:
    """Theme contributions of one card, including cross-feeds and changelings."""
    out: Dict[Tuple[str, str], float] = {}
    for tag, weight in tag_card(card).items():
        if ":" not in tag:
            continue
        theme, side = tag.split(":", 1)
        if theme == "tribal_any":
            for tribe in TRIBAL_TYPES:
                key = (f"tribal_{tribe}", side)
                out[key] = max(out.get(key, 0.0), float(weight))
            continue
        out[(theme, side)] = max(out.get((theme, side), 0.0), float(weight))
        for target_theme, target_side, factor in CROSS_FEEDS.get((theme, side), []):
            if factor <= 0:
                continue
            key = (target_theme, target_side)
            out[key] = max(out.get(key, 0.0), weight * factor)
    return out


def score_themes(pool: Iterable[Tuple[Card, int]]) -> List[ThemeScore]:
    """Score every theme present in the pool, best first."""
    scores: Dict[str, ThemeScore] = {}
    for card, count in pool:
        # Extra copies help, but with diminishing returns; a playset is not
        # four independent synergy pieces.
        copies = min(count, 4)
        multiplier = 1.0 + 0.35 * (copies - 1)
        for (theme, side), weight in _theme_contributions(card).items():
            entry = scores.setdefault(theme, ThemeScore(theme))
            contribution = weight * multiplier
            if side == "enabler":
                entry.enabler_weight += contribution
                if weight >= 1:
                    entry.enablers.append((card, int(weight)))
            else:
                entry.payoff_weight += contribution
                if weight >= 1:
                    entry.payoffs.append((card, int(weight)))
    ranked = sorted(scores.values(), key=lambda s: s.score, reverse=True)
    for entry in ranked:
        entry.enablers.sort(key=lambda cw: cw[1], reverse=True)
        entry.payoffs.sort(key=lambda cw: cw[1], reverse=True)
    return ranked


def filter_pool_by_colors(
    pool: Iterable[Tuple[Card, int]], colors: Iterable[str]
) -> List[Tuple[Card, int]]:
    colorset = set(colors)
    return [(c, n) for c, n in pool if c.fits_identity(colorset)]


@dataclass
class DeckIdea:
    commander: Optional[Card]
    colors: Tuple[str, ...]
    theme: ThemeScore
    support: Dict[str, int]          # role -> available card count in colors
    score: float

    @property
    def color_label(self) -> str:
        order = "WUBRG"
        chars = [c for c in order if c in self.colors]
        return "".join(chars) if chars else "C"


_ROLE_KEYS = ("ramp", "draw", "removal", "board_wipe")


def _role_support(pool: Iterable[Tuple[Card, int]]) -> Dict[str, int]:
    support = {role: 0 for role in _ROLE_KEYS}
    for card, count in pool:
        card_tags = tag_card(card)
        for role in _ROLE_KEYS:
            if role in card_tags:
                support[role] += min(count, 4)
    return support


def _support_bonus(support: Dict[str, int]) -> float:
    # Reward having the boring-but-necessary staples available, up to a cap.
    return (
        min(support["ramp"], 10) * 0.4
        + min(support["draw"], 10) * 0.4
        + min(support["removal"], 8) * 0.3
        + min(support["board_wipe"], 3) * 0.3
    )


def find_commander_decks(
    pool: List[Tuple[Card, int]], top: int = 10
) -> List[DeckIdea]:
    """Rank (commander, theme) pairs available in the pool."""
    ideas: List[DeckIdea] = []
    commanders = [c for c, _ in pool if c.is_legendary_creature]
    for commander in commanders:
        sub_pool = filter_pool_by_colors(pool, commander.color_identity or ("C",))
        themes = score_themes(sub_pool)
        support = _role_support(sub_pool)
        commander_themes = {
            tag.split(":", 1)[0] for tag in tag_card(commander) if ":" in tag
        }
        for theme in themes[:6]:
            if not theme.viable:
                continue
            # A commander that participates in the theme is worth a lot: it is
            # always available, so the deck's engine starts assembled.
            leader_bonus = 1.6 if theme.theme in commander_themes else 1.0
            ideas.append(
                DeckIdea(
                    commander=commander,
                    colors=tuple(commander.color_identity),
                    theme=theme,
                    support=support,
                    score=theme.score * leader_bonus + _support_bonus(support),
                )
            )
    ideas.sort(key=lambda i: i.score, reverse=True)

    # Keep the best idea per commander, then the best overall.
    seen = set()
    unique: List[DeckIdea] = []
    for idea in ideas:
        if idea.commander.name in seen:
            continue
        seen.add(idea.commander.name)
        unique.append(idea)
    return unique[:top]


# All mono colors and color pairs; enough to find focused 60-card decks.
_COLOR_COMBOS: List[Tuple[str, ...]] = [
    ("W",), ("U",), ("B",), ("R",), ("G",),
    ("W", "U"), ("W", "B"), ("W", "R"), ("W", "G"), ("U", "B"),
    ("U", "R"), ("U", "G"), ("B", "R"), ("B", "G"), ("R", "G"),
]


def find_sixty_card_decks(
    pool: List[Tuple[Card, int]], top: int = 8
) -> List[DeckIdea]:
    """Rank (color combo, theme) pairs for 60-card casual decks."""
    ideas: List[DeckIdea] = []
    for combo in _COLOR_COMBOS:
        sub_pool = filter_pool_by_colors(pool, combo)
        if sum(n for c, n in sub_pool if not c.is_land) < 30:
            continue
        support = _role_support(sub_pool)
        for theme in score_themes(sub_pool)[:4]:
            if not theme.viable:
                continue
            ideas.append(
                DeckIdea(
                    commander=None,
                    colors=combo,
                    theme=theme,
                    support=support,
                    score=theme.score + _support_bonus(support),
                )
            )
    ideas.sort(key=lambda i: i.score, reverse=True)

    # Prefer distinct themes over the same theme in five color pairs.
    seen = set()
    unique: List[DeckIdea] = []
    for idea in ideas:
        key = idea.theme.theme
        if key in seen and len(unique) < top - 2:
            continue
        seen.add(key)
        unique.append(idea)
    return unique[:top]
