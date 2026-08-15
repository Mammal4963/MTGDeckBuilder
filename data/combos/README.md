# Combo dataset

`spellbook-combos.jsonl.gz` — distilled from the
[Commander Spellbook](https://commanderspellbook.com) bulk export
(`https://json.commanderspellbook.com/variants.json.gz`), a
community-curated database of verified combos. Fetched once by
`experiments/fetch_spellbook_combos.py` and committed here so the
export is never re-downloaded; only card names, produced effects,
color identity and popularity are kept (~2.5 MB vs 630 MB raw).

Commander Spellbook content is community-maintained and made freely
available; this snapshot is used for research (training the combo-head
model in `experiments/combo_model.py`) with attribution and gratitude.

Snapshot: export version 6.1.1, 2026-08-15, 105,692 combos.
