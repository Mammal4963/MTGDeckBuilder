import unittest

from mtg_deckbuilder.collection import load_collection
from mtg_deckbuilder.deckbuilder import DeckBuildError, build_deck
from mtg_deckbuilder.synergy import resolve_pool

from tests.helpers import SAMPLE_COLLECTION, sample_db


class DeckBuilderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = sample_db()
        collection = load_collection(SAMPLE_COLLECTION)
        cls.pool, _missing = resolve_pool(collection, cls.db)

    def build_krenko(self):
        return build_deck(
            self.pool, fmt="commander", commander=self.db.get("Krenko, Mob Boss")
        )

    def test_commander_deck_is_exactly_100_cards(self):
        deck = self.build_krenko()
        self.assertEqual(deck.total_cards, 100)

    def test_commander_deck_is_singleton_except_basics(self):
        deck = self.build_krenko()
        for deck_card in deck.cards:
            if not deck_card.card.is_basic_land:
                self.assertEqual(
                    deck_card.count, 1, f"{deck_card.card.name} is not singleton"
                )

    def test_commander_deck_respects_color_identity(self):
        deck = self.build_krenko()
        for deck_card in deck.cards:
            self.assertTrue(
                deck_card.card.fits_identity({"R"}),
                f"{deck_card.card.name} is outside Krenko's color identity",
            )

    def test_commander_not_in_the_99(self):
        deck = self.build_krenko()
        names = [dc.card.name for dc in deck.cards]
        self.assertNotIn("Krenko, Mob Boss", names)

    def test_ramp_package_is_included(self):
        deck = self.build_krenko()
        ramp = [dc for dc in deck.cards if dc.category == "Ramp"]
        self.assertGreaterEqual(sum(dc.count for dc in ramp), 2)

    def test_land_count_is_reasonable(self):
        # The sample pool has few mono-red playables, so the commander deck
        # backfills with lands - but it must say so in the tuning notes.
        deck = self.build_krenko()
        lands = sum(dc.count for dc in deck.cards if dc.card.is_land)
        self.assertGreaterEqual(lands, 33)
        if lands > 40:
            self.assertTrue(any("ran out" in note for note in deck.notes))
        # With a deep enough pool (4-ofs allowed), the tuned count stays in
        # the formula's clamped range.
        deck60 = build_deck(self.pool, fmt="60", colors=["W", "B"], theme="lifegain")
        lands60 = sum(dc.count for dc in deck60.cards if dc.card.is_land)
        self.assertGreaterEqual(lands60, 18)
        self.assertLessEqual(lands60, 27)

    def test_deck_respects_collection_counts(self):
        deck = self.build_krenko()
        owned = {card.name: count for card, count in self.pool}
        for deck_card in deck.cards:
            if deck_card.card.is_basic_land:
                continue
            self.assertLessEqual(
                deck_card.count,
                owned.get(deck_card.card.name, 0),
                f"deck uses more copies of {deck_card.card.name} than owned",
            )

    def test_sixty_card_deck(self):
        deck = build_deck(self.pool, fmt="60", colors=["W", "B"], theme="lifegain")
        self.assertEqual(deck.total_cards, 60)
        for deck_card in deck.cards:
            self.assertTrue(deck_card.card.fits_identity({"W", "B"}))
            if not deck_card.card.is_basic_land:
                self.assertLessEqual(deck_card.count, 4)

    def test_sixty_card_needs_colors(self):
        with self.assertRaises(DeckBuildError):
            build_deck(self.pool, fmt="60")

    def test_every_nonland_pick_has_a_reason(self):
        deck = self.build_krenko()
        for deck_card in deck.cards:
            self.assertTrue(deck_card.reasons, f"{deck_card.card.name} has no reason")

    def test_notes_explain_tuning(self):
        deck = self.build_krenko()
        self.assertTrue(any("lands" in note for note in deck.notes))


if __name__ == "__main__":
    unittest.main()
