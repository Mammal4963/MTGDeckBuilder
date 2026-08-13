# Geometry-driven deck building experiments

Research track: replace the hand-written regex tagger with learned card
geometry, and see whether deck construction itself can be driven by it.
Lives on its own branch; nothing here touches the website until an
experiment earns its way in.

## The idea

1. **Text embeddings** capture what a card *does*: embed every oracle
   card's composed rules text (type | cost | stats | normalized text) so
   functionally similar cards are neighbors. But text similarity is not
   synergy - synergy is *complementarity* (Blood Artist + Viscera Seer
   share a deck, not a sentence).
2. **Co-occurrence embeddings** capture what a card is *played with*:
   train on real decklists (each deck a document, cards as words). In
   that space the famous pairs actually are neighbors.
3. The interesting objects live in the disagreement: pairs with high
   co-occurrence but low text similarity are true synergy pairs, and a
   projection from text-space into co-occurrence-space generalizes
   synergy predictions to cards no meta deck has ever played.
4. **Cluster recipes**: express meta decks as histograms over card
   clusters, learn the recipe distribution, instantiate recipes from a
   player's collection -> geometry-driven deck building.

## Phases

- [x] **1. Embed + map** - `embed_cards.py` (MiniLM, local, CPU),
  `project_umap.py`, `make_map.py` -> `output/card-map.html`, an
  interactive map of every playable card, colored by tagger function /
  color identity / card type.
- [ ] **2. Meta-deck corpus** - gather tournament decklists (respectful
  scraping or existing dumps), overlay them on the map, measure whether
  decks are "tight" in embedding space vs random baselines.
- [ ] **3. Co-occurrence space** - PMI/word2vec over decklists; mine the
  high-co-occurrence / low-text-similarity synergy list.
- [ ] **4. Recipes -> generation** - cluster recipes from meta decks,
  generate decks from a collection, evaluate by held-out deck
  reconstruction; wire the winner into the builder.

## Data

Card data comes from the committed snapshot in `data/` (see its README) -
do not re-download Scryfall bulk data for experiments.

## Running

```bash
pip install sentence-transformers umap-learn
python experiments/embed_cards.py    # ~30k cards, a few minutes on CPU
python experiments/project_umap.py
python experiments/make_map.py       # -> experiments/output/card-map.html
```

`output/embeddings.npy` is gitignored (regenerable, 50+ MB); the map and
coords are small enough to commit.
