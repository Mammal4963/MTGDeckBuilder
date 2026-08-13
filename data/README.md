# Card data snapshot

`scryfall-oracle-cards-2026-08-12.jsonl.gz` is a snapshot of Scryfall's
**Oracle Cards** bulk data file (one JSON object per Oracle ID, gzipped
JSON Lines), downloaded 2026-08-12 from <https://scryfall.com/docs/api/bulk-data>.

It is committed to the repo so that experiments never need to re-download
it — please be respectful of Scryfall's free API and reuse this snapshot
instead of fetching your own copy unless you actually need fresher data
(`mtgdeck fetch-data` refreshes into ~/.cache, not here).

Card data is copyright Wizards of the Coast, provided by Scryfall under
their data guidelines (<https://scryfall.com/docs/api>). This project is
unofficial Fan Content permitted under the Fan Content Policy; neither the
data nor this project is approved or endorsed by Wizards or Scryfall.

Everything in `mtg_deckbuilder/` and `experiments/` can load this file
directly (the loader handles `.jsonl.gz`):

```python
from mtg_deckbuilder.carddata import CardDatabase
db = CardDatabase.load("data/scryfall-oracle-cards-2026-08-12.jsonl.gz")
```
