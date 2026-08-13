import unittest

from mtg_deckbuilder.carddata import Card
from mtg_deckbuilder.collection import load_collection
from mtg_deckbuilder.deckbuilder import (
    DeckBuildError,
    _fill_scored,
    _goodstuff_score,
    _Selection,
    _threat_power,
    build_deck,
)
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

    def test_sixty_card_auto_picks_colors(self):
        deck = build_deck(self.pool, fmt="60")
        self.assertEqual(deck.total_cards, 60)
        self.assertTrue(deck.colors, "auto-pick should choose at least one color")
        self.assertTrue(
            any("Auto-picked" in note for note in deck.notes),
            "auto-picked colors should be explained in the notes",
        )

    def test_sixty_card_auto_pick_honors_forced_theme(self):
        deck = build_deck(self.pool, fmt="60", theme="lifegain")
        self.assertEqual(deck.theme, "lifegain")
        self.assertEqual(deck.total_cards, 60)

    def test_sixty_card_auto_pick_fails_on_tiny_pool(self):
        with self.assertRaises(DeckBuildError):
            build_deck(self.pool[:3], fmt="60")

    def test_sixty_card_prefers_playsets_over_equal_singles(self):
        # Same-quality creatures; we own 4 of one and 1 each of two others.
        # The focused builder should take the playset first.
        def bear(name):
            return Card(
                name=name, mana_cost="{1}{G}", mana_value=2.0,
                color_identity=("G",), type_line="Creature — Bear",
                keywords=("Trample",), power="2", toughness="2",
            )

        playset, single_a, single_b = bear("Bearset"), bear("Lone Bear A"), bear("Lone Bear B")
        selection = _Selection(
            [(playset, 4), (single_a, 1), (single_b, 1)], singleton=False
        )
        _fill_scored(
            selection, 4, "Flex", _goodstuff_score, lambda c: ["test"], 40
        )
        self.assertIn("Bearset", selection.picked)
        self.assertEqual(selection.picked["Bearset"].count, 4)

    def test_sixty_card_does_not_split_playsets(self):
        deck = build_deck(self.pool, fmt="60", colors=["B", "R"], theme="sacrifice")
        owned = {card.name: count for card, count in self.pool}
        for deck_card in deck.cards:
            if deck_card.card.is_basic_land or deck_card.card.is_land:
                continue
            usable = min(owned.get(deck_card.card.name, 0), 4)
            if usable >= 4:
                self.assertEqual(
                    deck_card.count, usable,
                    f"{deck_card.card.name}: took {deck_card.count} of {usable} usable",
                )

    def test_multi_copy_picks_explain_consistency(self):
        deck = build_deck(self.pool, fmt="60", colors=["B", "R"], theme="sacrifice")
        for deck_card in deck.cards:
            if deck_card.card.is_land or deck_card.count <= 1:
                continue
            self.assertTrue(
                any("copies for consistency" in r for r in deck_card.reasons),
                f"{deck_card.count}x {deck_card.card.name} lacks a consistency note",
            )
            # The note must match the final count, even after trimming.
            note = next(r for r in deck_card.reasons if "copies for consistency" in r)
            self.assertTrue(
                note.startswith(str(deck_card.count)),
                f"{deck_card.card.name}: count {deck_card.count} but note {note!r}",
            )

    def test_commander_deck_has_no_consistency_notes(self):
        deck = self.build_krenko()
        for deck_card in deck.cards:
            for reason in deck_card.reasons:
                self.assertNotIn("consistency", reason)

    def test_pretend_playsets_keeps_basic_land_counts(self):
        from mtg_deckbuilder.synergy import pretend_playsets

        boosted = dict(
            (c.name, n) for c, n in pretend_playsets(self.pool)
        )
        for card, count in self.pool:
            if card.is_basic_land:
                self.assertEqual(boosted[card.name], count)
            else:
                self.assertEqual(boosted[card.name], max(count, 4))

    def test_threat_power_scoring(self):
        attacker = Card(name="Bear", type_line="Creature — Bear",
                        power="2", toughness="2")
        self.assertGreater(_threat_power(attacker), 0)
        wall = Card(name="Wall", type_line="Creature — Wall",
                    keywords=("Defender",), power="0", toughness="4")
        self.assertEqual(_threat_power(wall), 0)
        chip = Card(name="Sprite", type_line="Creature — Faerie",
                    power="1", toughness="1")
        self.assertEqual(_threat_power(chip), 0)
        burn = Card(name="Zap", type_line="Instant",
                    oracle_text="Zap deals 3 damage to any target.")
        self.assertGreater(_threat_power(burn), 0)
        flyer = Card(name="Hawk", type_line="Creature — Bird",
                     keywords=("Flying",), power="2", toughness="1")
        self.assertGreater(_threat_power(flyer), _threat_power(attacker))

    def test_threatless_pool_gets_a_warning(self):
        def spell(name, text):
            return Card(name=name, mana_cost="{1}{U}", mana_value=2.0,
                        color_identity=("U",), type_line="Instant",
                        oracle_text=text)
        pool = [
            (spell("Deny", "Counter target spell."), 4),
            (spell("Ponderous", "Draw two cards."), 4),
            (spell("Grasp", "Destroy target creature."), 4),
            (spell("Twitch", "Untap target permanent. Draw a card."), 4),
        ]
        deck = build_deck(pool, fmt="60", colors=["U"], theme="spells")
        self.assertTrue(
            any(note.startswith("⚠") for note in deck.notes),
            f"expected a low-threat warning, notes: {deck.notes}",
        )

    def test_deck_with_attackers_gets_no_threat_warning(self):
        deck = build_deck(self.pool, fmt="60", colors=["B", "R"], theme="sacrifice")
        self.assertFalse(any(note.startswith("⚠") for note in deck.notes),
                         f"unexpected warning: {deck.notes}")

    def test_every_nonland_pick_has_a_reason(self):
        deck = self.build_krenko()
        for deck_card in deck.cards:
            self.assertTrue(deck_card.reasons, f"{deck_card.card.name} has no reason")

    def test_notes_explain_tuning(self):
        deck = self.build_krenko()
        self.assertTrue(any("lands" in note for note in deck.notes))


if __name__ == "__main__":
    unittest.main()
