from pathlib import Path

from mtg_deckbuilder.carddata import CardDatabase

SAMPLE_DB = Path(__file__).resolve().parent.parent / "examples" / "sample_cards.json"
SAMPLE_COLLECTION = (
    Path(__file__).resolve().parent.parent / "examples" / "sample_collection.txt"
)

_db = None


def sample_db() -> CardDatabase:
    global _db
    if _db is None:
        _db = CardDatabase.load(SAMPLE_DB)
    return _db
