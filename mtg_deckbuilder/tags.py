"""Extract functional roles and synergy tags from card text.

Two kinds of tags are produced for every card:

* **Role tags** (plain strings like ``ramp``, ``draw``, ``removal``) describe
  what job a card does in any deck.
* **Theme tags** (``theme:side`` strings like ``tokens:enabler`` or
  ``sacrifice:payoff``) describe which synergy package a card belongs to and
  which side of the synergy it sits on.  *Enablers* produce a resource
  (tokens, cards in graveyard, life, ...); *payoffs* reward you for having it.

Tag weights (1-3) express how strongly a card supports the tag; the synergy
scorer sums them.  Rules are ordinary data - to teach the tool a new
mechanic, add a `Rule` to `RULES` (or a keyword mapping to `KEYWORD_TAGS`)
and every command picks it up automatically.

All patterns are matched against a normalized oracle text: lowercased, with
the card's own name replaced by ``~`` (so "When Krenko, Mob Boss enters"
becomes "when ~ enters", matching the same rule as every other ETB card).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .carddata import Card


@dataclass(frozen=True)
class Rule:
    tag: str                       # "ramp" or "theme:enabler" / "theme:payoff"
    pattern: str                   # regex against normalized oracle text
    weight: int = 1
    type_ok: Optional[str] = None  # type_line must match this regex ...
    type_not: Optional[str] = None # ... and must not match this one


# Number words used in token/counter templating.
_N = r"(?:a|an|x|one|two|three|four|five|six|seven|eight|nine|ten|\d+|that many)"

RULES: List[Rule] = [
    # ------------------------------------------------------------------
    # Functional roles
    # ------------------------------------------------------------------
    # Mana rocks / dorks: a nonland permanent with a mana ability.
    Rule("ramp", r"\{t\}[^:\n]*: add ", 2, type_not=r"Land"),
    Rule("ramp", r"sacrifice [^:\n]*: add \{", 1, type_not=r"Land"),
    # Land ramp spells: fetch lands onto the battlefield.
    Rule("ramp", r"search your librar(?:y|ies) for [^.\n]*land[^.\n]*put[^.\n]* onto the battlefield", 2),
    # Extra land drops / land recursion.
    Rule("ramp", r"play (?:an|\w+) additional lands?", 2),
    Rule("ramp", r"put [^.\n]*land[^.\n]* from your hand onto the battlefield", 2),
    # Treasure and other mana tokens.
    Rule("ramp", r"create[^.\n]* treasure token", 2),
    # Cost reduction.
    Rule("ramp", r"spells? (?:you cast )?costs? (?:up to )?\{\d\} less to cast", 1),
    # Ritual-style burst mana.
    Rule("ramp", r"add \{[wubrgc]\}\{[wubrgc]\}", 1, type_ok=r"Instant|Sorcery"),

    Rule("draw", r"draws? (?:a|two|three|four|x|that many) cards?", 2),
    Rule("draw", r"draws? (?:a card|cards) (?:for each|equal to)", 3),
    Rule("draw", r"whenever [^.\n]*, draw a card", 3),
    # Impulse draw.
    Rule("draw", r"exile the top (?:card|two cards|three cards) of your library[^.\n]*(?:may )?(?:play|cast)", 2),
    Rule("card_selection", r"\bscry \d", 1),
    Rule("card_selection", r"look at the top [^.\n]* of your library", 1),

    Rule("removal", r"destroy target (?:[a-z-]+ )*(?:creature|artifact|enchantment|planeswalker|permanent)", 2),
    Rule("removal", r"exile target (?:[a-z-]+ )*(?:creature|artifact|enchantment|planeswalker|permanent)", 2),
    Rule("removal", r"deals? (?:\d+|x) damage to (?:any target|target creature|target attacking|each of up to)", 2),
    Rule("removal", r"target creature gets? [+-]?\d*x?/-", 2),   # -N/-N effects
    Rule("removal", r"\bfights? (?:target|up to|another target)", 1),
    Rule("removal", r"return target (?:creature|nonland permanent) to its owner'?s hand", 1),
    Rule("removal", r"put target (?:creature|nonland permanent) (?:on top|into its owner'?s library)", 1),

    Rule("board_wipe", r"destroy(?:s)? all (?:creatures|artifacts|enchantments|nonland permanents|permanents|other creatures)", 3),
    Rule("board_wipe", r"exiles? all (?:creatures|nonland permanents|other creatures)", 3),
    Rule("board_wipe", r"deals? (?:\d+|x) damage to each creature", 3),
    Rule("board_wipe", r"each player sacrifices (?:all|\d+|the rest)", 3),
    Rule("board_wipe", r"all creatures get -", 3),

    Rule("counterspell", r"counter target", 2),
    Rule("tutor", r"search your librar(?:y|ies) for (?!.*land)[^.\n]*card", 2),
    Rule("protection", r"(?:hexproof|indestructible|protection from|can't be countered|ward)", 1),
    Rule("recursion", r"return [^.\n]* from (?:your|a) graveyard to (?:your hand|the battlefield)", 2),
    Rule("recursion", r"put target [^.\n]*card from (?:a|your) graveyard onto the battlefield", 3),

    # ------------------------------------------------------------------
    # Token theme
    # ------------------------------------------------------------------
    Rule("tokens:enabler", rf"create {_N} [^.\n]*token", 2),
    Rule("tokens:enabler", r"create (?:a|x) (?:tapped )?[\d/]+ [^.\n]*creature tokens?", 2),
    Rule("tokens:payoff", r"creatures? you control (?:get|gets|gain|have) ", 2),
    Rule("tokens:payoff", r"whenever (?:a|another) (?:creature|nontoken creature)(?: you control)? enters", 3),
    Rule("tokens:payoff", r"whenever one or more [^.\n]*tokens? (?:you control )?enter", 3),
    Rule("tokens:payoff", r"for each creature you control", 2),
    Rule("tokens:payoff", r"tokens? you control", 2),
    Rule("tokens:payoff", r"\bpopulate\b", 2),

    # ------------------------------------------------------------------
    # Sacrifice / aristocrats theme
    # ------------------------------------------------------------------
    Rule("sacrifice:enabler", r"sacrifice (?:a|another) (?:creature|permanent|artifact)\s?(?::|,|\.)", 2),
    Rule("sacrifice:enabler", r"(?:^|\n|, )sacrifice (?:a|another) creature:", 3),
    # "Sacrifice a Goblin:" and other typed sacrifice costs.
    Rule("sacrifice:enabler", r"sacrifice (?:a|another) [a-z-]+:", 2),
    Rule("sacrifice:payoff", r"whenever (?:a|another) creature(?: you control)?(?: or planeswalker)? dies", 3),
    Rule("sacrifice:payoff", r"whenever you sacrifice", 3),
    Rule("sacrifice:payoff", r"whenever ~ or another creature (?:you control )?dies", 3),
    Rule("sacrifice:payoff", r"when(?:ever)? ~ dies", 1),
    Rule("sacrifice:payoff", r"each (?:player|opponent) sacrifices", 2),

    # ------------------------------------------------------------------
    # +1/+1 counters theme
    # ------------------------------------------------------------------
    Rule("counters:enabler", rf"put {_N} \+1/\+1 counters?", 2),
    Rule("counters:enabler", r"enters (?:the battlefield )?with [^.\n]*\+1/\+1 counters?", 1),
    Rule("counters:payoff", r"\bproliferate\b", 3),
    Rule("counters:payoff", r"whenever (?:one or more )?\+1/\+1 counters? (?:is|are) put", 3),
    Rule("counters:payoff", r"(?:each|every) creature(?: you control)? with a \+1/\+1 counter", 2),
    Rule("counters:payoff", r"creatures you control with \+1/\+1 counters", 2),
    Rule("counters:payoff", r"if [^.\n]* would (?:be placed|put) [^.\n]*counters?[^.\n]*(?:instead|twice that many|plus)", 3),

    # ------------------------------------------------------------------
    # Graveyard theme
    # ------------------------------------------------------------------
    Rule("graveyard:enabler", rf"\bmills? {_N} cards?", 2),
    Rule("graveyard:enabler", r"put(?:s)? the top [^.\n]* of (?:your|their) library into (?:your|their) graveyard", 2),
    Rule("graveyard:enabler", r"discard (?:a|two|three|your) cards?", 1),
    Rule("graveyard:payoff", r"return [^.\n]* from (?:your|a) graveyard to (?:your hand|the battlefield)", 2),
    Rule("graveyard:payoff", r"put target [^.\n]*card from (?:a|your) graveyard onto the battlefield", 2),
    Rule("graveyard:payoff", r"(?:for each|equal to the number of) [^.\n]*cards? in your graveyard", 3),
    Rule("graveyard:payoff", r"cast [^.\n]* from your graveyard", 2),
    Rule("graveyard:payoff", r"exile [^.\n]* from your graveyard(?:[:,]| as)", 1),

    # ------------------------------------------------------------------
    # Lifegain theme
    # ------------------------------------------------------------------
    Rule("lifegain:enabler", r"you (?:gain|gains) (?:\d+|x|that much) life", 2),
    Rule("lifegain:enabler", r"gain life equal to", 2),
    Rule("lifegain:payoff", r"whenever you gain life", 3),
    Rule("lifegain:payoff", r"if you (?:gained life|have more life)", 2),
    Rule("lifegain:payoff", r"you have (?:\d+ or more|more) life", 2),

    # ------------------------------------------------------------------
    # Spellslinger theme (instants & sorceries matter)
    # ------------------------------------------------------------------
    Rule("spells:payoff", r"whenever you cast (?:an instant or sorcery|a noncreature|an instant, sorcery)", 3),
    Rule("spells:payoff", r"instant and sorcery spells (?:you cast )?cost", 2),
    Rule("spells:payoff", r"copy (?:target|that|it)[^.\n]*(?:instant|sorcery|spell)", 2),
    Rule("spells:payoff", r"whenever you cast your (?:first|second) spell", 2),

    # ------------------------------------------------------------------
    # Artifacts theme
    # ------------------------------------------------------------------
    Rule("artifacts:enabler", r"create [^.\n]*(?:treasure|clue|food|blood|artifact) token", 2),
    Rule("artifacts:payoff", r"whenever (?:an|another) artifact (?:you control )?enters", 3),
    Rule("artifacts:payoff", r"artifacts? you control", 2),
    Rule("artifacts:payoff", r"(?:for each|equal to the number of) artifacts? you control", 3),
    Rule("artifacts:payoff", r"artifact spells? (?:you cast )?cost", 2),

    # ------------------------------------------------------------------
    # Enchantments theme
    # ------------------------------------------------------------------
    Rule("enchantments:payoff", r"whenever (?:an|another) enchantment (?:you control )?enters", 3),
    Rule("enchantments:payoff", r"enchantments? you control", 2),
    Rule("enchantments:payoff", r"(?:for each|equal to the number of) enchantments? you control", 3),
    Rule("enchantments:payoff", r"enchantment spells? (?:you cast )?cost", 2),

    # ------------------------------------------------------------------
    # Lands theme
    # ------------------------------------------------------------------
    Rule("lands:payoff", r"whenever a land enters the battlefield under your control", 3),
    Rule("lands:payoff", r"(?:for each|equal to the number of) lands? you control", 2),
    Rule("lands:payoff", r"play (?:an|\w+) additional lands?", 2),
    Rule("lands:enabler", r"search your librar(?:y|ies) for [^.\n]*land[^.\n]*put[^.\n]* onto the battlefield", 2),
    Rule("lands:enabler", r"return [^.\n]*land[^.\n]* from your graveyard", 1),

    # ------------------------------------------------------------------
    # Blink / ETB theme
    # ------------------------------------------------------------------
    Rule("blink:enabler", r"exile [^.\n]*(?:creature|permanent)s? you (?:own|control)[^.\n]*, then return", 3),
    Rule("blink:enabler", r"exile [^.\n]*(?:creature|permanent)[^.\n]*return (?:it|that card|them) to the battlefield", 2),
    Rule("blink:payoff", r"when(?:ever)? ~ enters(?:the battlefield)?[, ]", 1),

    # ------------------------------------------------------------------
    # Voltron (equipment / auras)
    # ------------------------------------------------------------------
    Rule("voltron:enabler", r"equipped creature", 2, type_ok=r"Equipment"),
    Rule("voltron:enabler", r"enchant creature", 1, type_ok=r"Aura"),
    Rule("voltron:payoff", r"whenever you (?:attach|cast a spell that targets)", 3),
    Rule("voltron:payoff", r"(?:equipment|aura)s? you control", 2),
    Rule("voltron:payoff", r"equip costs? (?:you pay )?cost", 2),
    Rule("voltron:payoff", r"(?:enchanted|equipped) creature (?:you control )?(?:gets|has|gains)", 1),

    # ------------------------------------------------------------------
    # Mill (opponents) and discard themes
    # ------------------------------------------------------------------
    Rule("mill:enabler", r"(?:target (?:player|opponent)|each opponent) mills", 2),
    Rule("mill:payoff", r"(?:for each|equal to the number of) cards? in (?:their|each opponent's|all) graveyards?", 2),
    Rule("discard:enabler", r"(?:target (?:player|opponent)|each opponent) discards", 2),
    Rule("discard:payoff", r"whenever an opponent discards", 3),

    # ------------------------------------------------------------------
    # Big power matters
    # ------------------------------------------------------------------
    Rule("power:payoff", r"power (?:4|5) or greater", 3),
    Rule("power:payoff", r"greatest power among", 2),
]

# Official keywords (Scryfall's `keywords` array) that imply a tag.
KEYWORD_TAGS: Dict[str, Tuple[str, int]] = {
    "lifelink": ("lifegain:enabler", 1),
    "extort": ("lifegain:enabler", 1),
    "prowess": ("spells:payoff", 2),
    "magecraft": ("spells:payoff", 3),
    "storm": ("spells:payoff", 3),
    "landfall": ("lands:payoff", 3),
    "flashback": ("graveyard:payoff", 2),
    "jump-start": ("graveyard:payoff", 2),
    "escape": ("graveyard:payoff", 2),
    "delve": ("graveyard:payoff", 2),
    "threshold": ("graveyard:payoff", 2),
    "delirium": ("graveyard:payoff", 2),
    "unearth": ("graveyard:payoff", 2),
    "dredge": ("graveyard:enabler", 2),
    "affinity": ("artifacts:payoff", 2),
    "metalcraft": ("artifacts:payoff", 3),
    "improvise": ("artifacts:payoff", 2),
    "constellation": ("enchantments:payoff", 3),
    "bestow": ("enchantments:enabler", 1),
    "proliferate": ("counters:payoff", 3),
    "populate": ("tokens:payoff", 2),
    "convoke": ("tokens:payoff", 1),
    "exploit": ("sacrifice:enabler", 2),
    "devour": ("sacrifice:enabler", 1),
    "ferocious": ("power:payoff", 2),
    "formidable": ("power:payoff", 2),
    "equip": ("voltron:enabler", 2),
    "mill": ("graveyard:enabler", 1),
    "treasure": ("ramp", 1),
}

# Creature types with common tribal support.  A card of one of these types
# gets `tribal_<type>:enabler`; text caring about the type gets `:payoff`.
TRIBAL_TYPES = [
    "angel", "beast", "bird", "cat", "cleric", "demon", "dinosaur", "dog",
    "dragon", "druid", "elemental", "elf", "faerie", "goblin", "human",
    "hydra", "knight", "merfolk", "ninja", "pirate", "rat", "rogue",
    "samurai", "shaman", "skeleton", "sliver", "snake", "soldier", "spider",
    "spirit", "squirrel", "treefolk", "vampire", "warrior", "wizard", "wolf",
    "zombie",
]

_COMPILED = [(rule, re.compile(rule.pattern)) for rule in RULES]
_TRIBAL_PAYOFF = {
    t: re.compile(
        rf"(?:(?:other )?{t}s? (?:you control )?(?:get|gets|gain|have|has)\b"
        rf"|whenever (?:a|another) {t}(?: you control)? (?:enters|dies|attacks)"
        rf"|for each {t}"
        rf"|number of {t}s? you control"
        rf"|whenever you cast an? {t} spell"
        rf"|{t} spells? (?:you cast )?cost)"
    )
    for t in TRIBAL_TYPES
}


def normalize_text(card: Card) -> str:
    """Lowercase oracle text with self-references replaced by ``~``."""
    text = card.oracle_text
    for face in card.name.split(" // "):
        face = face.strip()
        if face:
            text = text.replace(face, "~")
        # Cards often refer to themselves by first name ("Krenko can't block").
        short = face.split(",")[0].strip()
        if len(short) > 2:
            text = text.replace(short, "~")
    text = text.lower()
    # Newer templating says "this creature" / "this spell" instead of the name.
    text = re.sub(r"\bthis (?:creature|spell|permanent|artifact|enchantment|land|card)\b", "~", text)
    return text


def tag_card(card: Card) -> Dict[str, int]:
    """Return ``{tag: weight}`` for a card. Also cached on ``card.tags``."""
    if card.tags:
        return card.tags

    tags: Dict[str, int] = {}

    def bump(tag: str, weight: int) -> None:
        tags[tag] = max(tags.get(tag, 0), weight)

    text = normalize_text(card)

    for rule, compiled in _COMPILED:
        if rule.type_ok and not re.search(rule.type_ok, card.type_line):
            continue
        if rule.type_not and re.search(rule.type_not, card.type_line):
            continue
        if compiled.search(text):
            bump(rule.tag, rule.weight)

    for keyword in card.keywords:
        mapped = KEYWORD_TAGS.get(keyword.lower())
        if mapped:
            bump(*mapped)

    # Tribal membership and payoffs.
    subtypes = {s.lower() for s in card.subtypes}
    if "changeling" in {k.lower() for k in card.keywords}:
        bump("tribal_any:enabler", 1)
    for tribe in TRIBAL_TYPES:
        if card.is_creature and tribe in subtypes:
            bump(f"tribal_{tribe}:enabler", 1)
        if _TRIBAL_PAYOFF[tribe].search(text):
            bump(f"tribal_{tribe}:payoff", 3)

    # Big creatures enable power-matters decks.
    power = card.numeric_power
    if power is not None and power >= 4:
        bump("power:enabler", 1 if power < 6 else 2)

    # Cheap instants/sorceries are the fuel for spellslinger payoffs.
    if ("Instant" in card.types or "Sorcery" in card.types) and card.mana_value <= 3:
        bump("spells:enabler", 1)
    if "Artifact" in card.types and not card.is_land:
        bump("artifacts:enabler", 1)
    if "Enchantment" in card.types:
        bump("enchantments:enabler", 1)
    # Creatures with ETB triggers are what blink decks want to blink.
    if card.is_creature and re.search(r"when(?:ever)? ~ enters", text):
        bump("blink:payoff", 2)

    card.tags = tags
    return tags


def role_tags(tags: Dict[str, int]) -> Dict[str, int]:
    return {t: w for t, w in tags.items() if ":" not in t}


def theme_tags(tags: Dict[str, int]) -> Dict[str, int]:
    return {t: w for t, w in tags.items() if ":" in t}
