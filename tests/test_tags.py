import unittest

from mtg_deckbuilder.tags import tag_card

from tests.helpers import sample_db


class TagExtractionTests(unittest.TestCase):
    def setUp(self):
        self.db = sample_db()

    def tags_of(self, name):
        card = self.db.get(name)
        self.assertIsNotNone(card, f"{name} missing from sample DB")
        return tag_card(card)

    def test_mana_dorks_and_rocks_are_ramp(self):
        for name in ("Llanowar Elves", "Sol Ring", "Arcane Signet", "Elvish Archdruid"):
            self.assertIn("ramp", self.tags_of(name), name)

    def test_land_ramp_spells(self):
        self.assertIn("ramp", self.tags_of("Cultivate"))
        self.assertIn("ramp", self.tags_of("Rampant Growth"))

    def test_basic_lands_are_not_ramp(self):
        self.assertNotIn("ramp", self.tags_of("Forest"))

    def test_typed_sacrifice_cost_is_ramp_and_sac_outlet(self):
        tags = self.tags_of("Skirk Prospector")
        self.assertIn("ramp", tags)
        self.assertIn("sacrifice:enabler", tags)

    def test_draw_spells(self):
        self.assertIn("draw", self.tags_of("Harmonize"))
        self.assertIn("draw", self.tags_of("Sign in Blood"))       # "draws two cards"
        self.assertIn("draw", self.tags_of("Night's Whisper"))
        self.assertIn("draw", self.tags_of("Skullclamp"))

    def test_removal_handles_adjectives(self):
        self.assertIn("removal", self.tags_of("Doom Blade"))       # "nonblack creature"
        self.assertIn("removal", self.tags_of("Swords to Plowshares"))
        self.assertIn("removal", self.tags_of("Lightning Bolt"))

    def test_board_wipes(self):
        self.assertIn("board_wipe", self.tags_of("Wrath of God"))
        self.assertIn("board_wipe", self.tags_of("Blasphemous Act"))

    def test_counterspell(self):
        self.assertIn("counterspell", self.tags_of("Counterspell"))

    def test_token_theme_sides(self):
        self.assertIn("tokens:enabler", self.tags_of("Krenko, Mob Boss"))
        self.assertIn("tokens:enabler", self.tags_of("Raise the Alarm"))
        self.assertIn("tokens:payoff", self.tags_of("Impact Tremors"))
        self.assertIn("tokens:payoff", self.tags_of("Intangible Virtue"))
        self.assertIn("tokens:payoff", self.tags_of("Purphoros, God of the Forge"))

    def test_sacrifice_theme_sides(self):
        self.assertIn("sacrifice:enabler", self.tags_of("Viscera Seer"))
        self.assertIn("sacrifice:enabler", self.tags_of("Goblin Bombardment"))
        self.assertIn("sacrifice:payoff", self.tags_of("Blood Artist"))
        self.assertIn("sacrifice:payoff", self.tags_of("Zulaport Cutthroat"))
        self.assertIn("sacrifice:payoff", self.tags_of("Grave Pact"))

    def test_lifegain_theme_sides(self):
        self.assertIn("lifegain:enabler", self.tags_of("Soul Warden"))
        self.assertIn("lifegain:payoff", self.tags_of("Ajani's Pridemate"))
        self.assertIn("lifegain:payoff", self.tags_of("Vito, Thorn of the Dusk Rose"))

    def test_spellslinger_theme_sides(self):
        self.assertIn("spells:payoff", self.tags_of("Talrand, Sky Summoner"))
        self.assertIn("spells:payoff", self.tags_of("Young Pyromancer"))
        self.assertIn("spells:payoff", self.tags_of("Archmage Emeritus"))  # Magecraft
        self.assertIn("spells:enabler", self.tags_of("Opt"))

    def test_graveyard_theme_sides(self):
        self.assertIn("graveyard:enabler", self.tags_of("Stitcher's Supplier"))
        self.assertIn("graveyard:payoff", self.tags_of("Reanimate"))
        self.assertIn("graveyard:payoff", self.tags_of("Gravecrawler"))
        self.assertIn("recursion", self.tags_of("Reanimate"))

    def test_counters_theme(self):
        self.assertIn("counters:enabler", self.tags_of("Carrion Feeder"))
        self.assertIn("counters:enabler", self.tags_of("Ajani's Pridemate"))

    def test_tribal_tags(self):
        krenko = self.tags_of("Krenko, Mob Boss")
        self.assertIn("tribal_goblin:enabler", krenko)   # is a Goblin
        self.assertIn("tribal_goblin:payoff", krenko)    # counts Goblins
        chieftain = self.tags_of("Goblin Chieftain")
        self.assertIn("tribal_goblin:payoff", chieftain) # lords Goblins
        huntmaster = self.tags_of("Lys Alana Huntmaster")
        self.assertIn("tribal_elf:payoff", huntmaster)   # cast an Elf spell

    def test_tutor(self):
        self.assertIn("tutor", self.tags_of("Goblin Matron"))

    def test_etb_creature_is_blink_payoff(self):
        self.assertIn("blink:payoff", self.tags_of("Goblin Matron"))

    def test_self_name_normalization(self):
        # "Whenever Blood Artist or another creature dies" must match the
        # generic "whenever ~ or another creature dies" pattern.
        self.assertIn("sacrifice:payoff", self.tags_of("Blood Artist"))


if __name__ == "__main__":
    unittest.main()
