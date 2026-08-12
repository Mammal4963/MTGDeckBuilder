import gzip
import json
import tempfile
import unittest
from pathlib import Path

from mtg_deckbuilder.carddata import CardDatabase

from tests.helpers import SAMPLE_DB, sample_db


class CardDataTests(unittest.TestCase):
    def setUp(self):
        self.db = sample_db()

    def test_lookup_is_case_insensitive(self):
        self.assertIsNotNone(self.db.get("llanowar elves"))
        self.assertIsNotNone(self.db.get("  Llanowar   Elves "))

    def test_double_faced_card_front_name_lookup(self):
        card = self.db.get("Malakir Rebirth")
        self.assertIsNotNone(card)
        self.assertEqual(card.name, "Malakir Rebirth // Malakir Bloodchasm")
        # Both faces' text is available for tagging.
        self.assertIn("return it to the battlefield", card.oracle_text)
        self.assertIn("Add {B}", card.oracle_text)

    def test_type_parsing(self):
        krenko = self.db.get("Krenko, Mob Boss")
        self.assertTrue(krenko.is_legendary_creature)
        self.assertIn("Goblin", krenko.subtypes)
        self.assertFalse(krenko.is_land)

        mountain = self.db.get("Mountain")
        self.assertTrue(mountain.is_basic_land)
        self.assertFalse(mountain.is_creature)

    def test_god_is_commander_candidate(self):
        purphoros = self.db.get("Purphoros, God of the Forge")
        self.assertTrue(purphoros.is_legendary_creature)

    def test_loads_jsonl_and_gzipped_files(self):
        # Scryfall's bulk data is now JSON Lines (optionally gzipped); the
        # loader must read those as well as the legacy JSON array.
        with open(SAMPLE_DB, encoding="utf-8") as fh:
            entries = json.load(fh)
        lines = "\n".join(json.dumps(e) for e in entries)
        with tempfile.TemporaryDirectory() as tmp:
            jsonl = Path(tmp) / "cards.jsonl"
            jsonl.write_text(lines, encoding="utf-8")
            db = CardDatabase.load(jsonl)
            self.assertEqual(len(db), len(self.db))
            self.assertIsNotNone(db.get("Llanowar Elves"))

            gz = Path(tmp) / "cards.jsonl.gz"
            with gzip.open(gz, "wt", encoding="utf-8") as fh:
                fh.write(lines)
            db_gz = CardDatabase.load(gz)
            self.assertEqual(len(db_gz), len(self.db))

    def test_color_identity_fit(self):
        vito = self.db.get("Vito, Thorn of the Dusk Rose")
        self.assertTrue(vito.fits_identity({"B", "W"}))
        self.assertFalse(vito.fits_identity({"W"}))
        sol_ring = self.db.get("Sol Ring")
        self.assertTrue(sol_ring.fits_identity(set()))


if __name__ == "__main__":
    unittest.main()
