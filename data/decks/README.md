# MTGO decklist corpus

`mtgo-decks.jsonl.gz` - tournament decklists from the officially published
MTGO results at <https://www.mtgo.com/decklists> (league 5-0s, challenges,
preliminaries) for the constructed 60-card formats: standard, pioneer,
modern, legacy, vintage, pauper.

Fetched once by `experiments/fetch_mtgo_decks.py` (>=1.2s between
requests) and committed here so experiments never re-crawl the site.
Decklists and card names are copyright Wizards of the Coast; published
results are used here for non-commercial research under the Fan Content
Policy. This project is unofficial and not endorsed by Wizards or
Daybreak Games.

One JSON object per line:

```json
{"event": "/decklist/pauper-league-...", "format": "pauper",
 "date": "2026-08-13", "player": "...", "wins": 5,
 "main": [["Lightning Bolt", 4], ...], "side": [...]}
```
