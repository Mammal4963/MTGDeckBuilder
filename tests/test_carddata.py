import unittest

from tests.helpers import sample_db


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

    def test_color_identity_fit(self):
        vito = self.db.get("Vito, Thorn of the Dusk Rose")
        self.assertTrue(vito.fits_identity({"B", "W"}))
        self.assertFalse(vito.fits_identity({"W"}))
        sol_ring = self.db.get("Sol Ring")
        self.assertTrue(sol_ring.fits_identity(set()))


if __name__ == "__main__":
    unittest.main()
