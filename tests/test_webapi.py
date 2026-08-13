import json
import unittest

from mtg_deckbuilder import webapi

from tests.helpers import SAMPLE_COLLECTION, SAMPLE_DB


def _load_entries():
    with open(SAMPLE_DB, encoding="utf-8") as fh:
        return json.load(fh)


class WebApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.entries = _load_entries()
        cls.collection_text = SAMPLE_COLLECTION.read_text(encoding="utf-8")

    def call(self, action, **kwargs):
        payload = {"action": action, "collection": self.collection_text,
                   "cards": self.entries}
        payload.update(kwargs)
        response = json.loads(webapi.run_json(json.dumps(payload)))
        return response

    def test_collection_names_returns_json_array(self):
        names = json.loads(webapi.collection_names("4 Sol Ring\nKrenko, Mob Boss"))
        self.assertEqual(sorted(names), ["Krenko, Mob Boss", "Sol Ring"])

    def test_meta_lists_themes(self):
        response = self.call("meta")
        self.assertTrue(response["ok"])
        keys = {t["key"] for t in response["result"]["themes"]}
        self.assertIn("sacrifice", keys)
        self.assertIn("lifegain", keys)

    def test_analyze(self):
        response = self.call("analyze")
        self.assertTrue(response["ok"])
        result = response["result"]
        self.assertGreater(result["total_cards"], 0)
        self.assertTrue(any(r["role"] == "removal" for r in result["roles"]))
        self.assertTrue(result["themes"])

    def test_suggest_sixty(self):
        response = self.call("suggest")
        self.assertTrue(response["ok"])
        ideas = response["result"]["ideas"]
        self.assertTrue(ideas)
        self.assertIsNone(ideas[0]["commander"])
        self.assertTrue(ideas[0]["color_label"])

    def test_suggest_commander(self):
        response = self.call("suggest", format="commander")
        self.assertTrue(response["ok"])
        ideas = response["result"]["ideas"]
        self.assertTrue(ideas)
        self.assertTrue(all(i["commander"] for i in ideas))

    def test_build_auto_is_sixty_cards(self):
        response = self.call("build")
        self.assertTrue(response["ok"], response.get("error"))
        deck = response["result"]
        self.assertEqual(deck["format"], "60")
        self.assertEqual(deck["total"], 60)
        self.assertEqual(
            sum(c["count"] for cat in deck["categories"] for c in cat["cards"]), 60
        )
        self.assertTrue(deck["export_text"].splitlines())

    def test_build_playsets_pretends_four_of_everything(self):
        response = self.call("build", playsets=True, colors="BR", theme="sacrifice")
        self.assertTrue(response["ok"], response.get("error"))
        deck = response["result"]
        self.assertEqual(deck["total"], 60)
        # Cards owned as singles in the sample collection (e.g. Viscera Seer,
        # 1x) can now appear as multiples; at least one pick must exceed the
        # owned count.
        from mtg_deckbuilder.collection import parse_collection
        owned = dict(parse_collection(self.collection_text).items())
        exceeded = [
            c["name"]
            for cat in deck["categories"] if cat["name"] != "Lands"
            for c in cat["cards"]
            if c["count"] > owned.get(c["name"], 0)
        ]
        self.assertTrue(exceeded, "playsets mode never used more copies than owned")

    def test_build_commander(self):
        response = self.call("build", commander="Krenko, Mob Boss")
        self.assertTrue(response["ok"], response.get("error"))
        deck = response["result"]
        self.assertEqual(deck["format"], "commander")
        self.assertEqual(deck["commander"], "Krenko, Mob Boss")
        self.assertEqual(deck["total"], 100)

    def test_bad_commander_is_a_clean_error(self):
        response = self.call("build", commander="Not A Real Card")
        self.assertFalse(response["ok"])
        self.assertIn("not found", response["error"])

    def test_bad_colors_is_a_clean_error(self):
        response = self.call("build", colors="xyz")
        self.assertFalse(response["ok"])
        self.assertIn("WUBRG", response["error"])

    def test_empty_cards_is_a_clean_error(self):
        payload = {"action": "build", "collection": "1 Sol Ring", "cards": []}
        response = json.loads(webapi.run_json(json.dumps(payload)))
        self.assertFalse(response["ok"])

    def test_malformed_payload_is_a_clean_error(self):
        response = json.loads(webapi.run_json("this is not json"))
        self.assertFalse(response["ok"])


if __name__ == "__main__":
    unittest.main()
