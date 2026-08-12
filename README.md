# MTG Deck Builder

Find the decks hiding in your Magic: The Gathering collection.

You can't hold a thousand cards in your head at once — by the time you've read
the tenth card you've forgotten the first. This tool reads your whole
collection at once: it extracts keywords and mechanics from every card's text,
groups cards by synergy (enablers vs. payoffs), tells you which decks you
already own the pieces for, and then builds a tuned list — ramp package, card
draw, removal, and a land count matched to the mana curve — explaining why
every card made the cut.

Pure Python, no dependencies, works offline.

## Quick start

No install needed (Python 3.9+). Try it immediately on the bundled sample:

```bash
python -m mtg_deckbuilder analyze examples/sample_collection.txt
python -m mtg_deckbuilder suggest examples/sample_collection.txt
python -m mtg_deckbuilder build examples/sample_collection.txt --commander "Krenko, Mob Boss" --explain
```

Or install it to get the `mtgdeck` command:

```bash
pip install .
mtgdeck analyze my_collection.txt
```

## Setting up real card data

The bundled sample database only knows ~60 cards. For your real collection,
download Scryfall's free card catalog (updated daily):

```bash
mtgdeck fetch-data          # ~150 MB, saved to ~/.cache/mtg_deckbuilder/
```

If your machine can't reach the internet, download the **Oracle Cards** bulk
file from <https://scryfall.com/docs/api/bulk-data> in a browser and either
save it as `~/.cache/mtg_deckbuilder/oracle-cards.json` or pass
`--data /path/to/oracle-cards.json` (or set `MTGDECK_DATA`).

## Your collection file

Plain text, one card per line — counts optional, comments ignored:

```
4 Llanowar Elves
2x Sol Ring
Krenko, Mob Boss
# binder page 2
```

CSV exports also work: any file with a `Name`/`Card` column and an optional
`Count`/`Quantity`/`Qty` column (covers ManaBox, Moxfield, Archidekt, and
Deckbox exports).

## Commands

### `mtgdeck analyze <collection>`

The "keep it all in your head" view. Counts every functional role in your
collection (ramp, draw, removal, wipes, tutors...) and scores every synergy
theme it can detect, with the key cards for each:

```
 * Go-wide tokens        score 35.3   enablers 17.5 / payoffs 17.8
      key cards: Impact Tremors, Purphoros, Krenko, Mob Boss, ...
```

A theme only scores well when you own both halves of it — twenty token
producers with nothing that rewards tokens is not a deck, and the score
(a geometric mean of the two sides) reflects that.

### `mtgdeck suggest <collection> [--format commander|60]`

Finds complete decks hiding in the collection. For Commander it tries every
legendary creature you own as a leader, scores every theme inside that
commander's color identity (commanders that personally participate in the
theme score higher), and checks you own enough ramp/draw/removal support.
For 60-card it sweeps every mono-color and color pair.

### `mtgdeck build <collection> [options]`

Builds and tunes an actual decklist:

```bash
mtgdeck build cards.txt --commander "Krenko, Mob Boss"          # 100-card EDH
mtgdeck build cards.txt --format 60 --colors BR --theme sacrifice
mtgdeck build cards.txt --commander "..." --explain             # why each card?
```

The builder fills role quotas first (ramp package, card draw, removal, board
wipes — preferring cards that also serve the theme, like Skirk Prospector in
a goblin sacrifice deck), then packs the theme's enablers and payoffs while
shaping the mana curve, then tunes the land count from the finished curve:
more lands for a high average mana value, fewer when you have plenty of cheap
ramp. Nonbasic lands come from your collection; basics are split by colored
pip counts. Only cards you own are used (respecting how many copies you own,
singleton for Commander).

### `mtgdeck card "<name>"`

Shows exactly which tags the analyzer extracted from one card — useful for
understanding (or distrusting) a suggestion:

```
Skullclamp  {1}  (MV 1)
  draw                                     weight 2
  Voltron (equipment & auras) - enabler    weight 2
```

## How it works

1. **Tag extraction** (`tags.py`) — every card is run through a rule set that
   combines Scryfall's official keyword list with regexes over oracle text
   (self-references normalized to `~`, so "Whenever Blood Artist or another
   creature dies" matches the same rule as every other death trigger). Tags
   are either *roles* (`ramp`, `draw`, `removal`...) or *theme sides*
   (`tokens:enabler`, `sacrifice:payoff`...), each weighted 1–3.
2. **Synergy scoring** (`synergy.py`) — a theme's score is
   `2·√(enablers × payoffs)`, maximized when the two sides are balanced.
   Cross-feeds link related themes: token producers count as sacrifice
   fodder, cheap spells feed the graveyard, changelings join every tribe.
   Extra copies of a card help with diminishing returns.
3. **Deck building** (`deckbuilder.py`) — quota-driven greedy selection with
   a curve-shaping penalty, then a Karsten-style land formula
   (`31.5 + 3.1·avgMV − 0.5·cheapRamp` for Commander, clamped 33–40).

## Teaching it new mechanics

All detection is data: add a `Rule` to `RULES` in `mtg_deckbuilder/tags.py`
(a tag name, a regex, a weight), a keyword to `KEYWORD_TAGS`, or a creature
type to `TRIBAL_TYPES`, and every command picks it up automatically.

## Tests

```bash
python -m unittest discover -s tests
```

## Repository layout

```
mtg_deckbuilder/
  carddata.py     card model + Scryfall bulk loading
  collection.py   collection file parsing (txt / CSV)
  tags.py         keyword & theme extraction rules
  synergy.py      theme scoring, hidden-deck discovery
  deckbuilder.py  deck assembly, ramp package, land tuning
  fetch.py        Scryfall bulk-data download
  cli.py          the mtgdeck command
examples/         sample card DB + sample collection
tests/            unit tests (no network needed)
```
