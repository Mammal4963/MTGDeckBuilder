import unittest
from pathlib import Path

from mtg_deckbuilder.collection import load_collection
from mtg_deckbuilder.synergy import (
    find_commander_decks,
    find_sixty_card_decks,
    resolve_pool,
    score_themes,
)

from tests.helpers import SAMPLE_COLLECTION, sample_db


class SynergyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = sample_db()
        collection = load_collection(SAMPLE_COLLECTION)
        cls.pool, missing = resolve_pool(collection, cls.db)
        assert not missing, f"sample collection has unknown cards: {missing}"

    def test_sample_collection_fully_resolves(self):
        self.assertGreater(len(self.pool), 50)

    def test_token_and_sacrifice_themes_score_high(self):
        themes = {t.theme: t for t in score_themes(self.pool)}
        self.assertIn("tokens", themes)
        self.assertIn("sacrifice", themes)
        self.assertTrue(themes["tokens"].viable)
        self.assertTrue(themes["sacrifice"].viable)

    def test_theme_score_requires_both_sides(self):
        # A pool of only token payoffs with no producers should score ~0.
        payoff_only = [
            (card, count)
            for card, count in self.pool
            if card.name in ("Intangible Virtue", "Impact Tremors")
        ]
        themes = {t.theme: t for t in score_themes(payoff_only)}
        if "tokens" in themes:
            self.assertEqual(themes["tokens"].enabler_weight, 0)
            self.assertEqual(themes["tokens"].score, 0)

    def test_commander_suggestions_include_krenko_tokens(self):
        ideas = find_commander_decks(self.pool)
        self.assertTrue(ideas, "expected at least one commander deck idea")
        by_name = {idea.commander.name: idea for idea in ideas}
        self.assertIn("Krenko, Mob Boss", by_name)
        krenko = by_name["Krenko, Mob Boss"]
        self.assertIn(krenko.theme.theme, ("tokens", "tribal_goblin"))
        # Krenko is mono-red; every suggested pool stat must respect that.
        self.assertEqual(set(krenko.colors), {"R"})

    def test_commander_decks_are_ranked(self):
        ideas = find_commander_decks(self.pool)
        scores = [idea.score for idea in ideas]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_sixty_card_suggestions(self):
        ideas = find_sixty_card_decks(self.pool)
        self.assertTrue(ideas)
        for idea in ideas:
            self.assertIsNone(idea.commander)
            self.assertTrue(idea.theme.viable)


if __name__ == "__main__":
    unittest.main()
