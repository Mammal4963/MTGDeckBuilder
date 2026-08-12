import tempfile
import unittest
from pathlib import Path

from mtg_deckbuilder.collection import load_collection


def _write(tmpdir: str, name: str, content: str) -> Path:
    path = Path(tmpdir) / name
    path.write_text(content, encoding="utf-8")
    return path


class CollectionParsingTests(unittest.TestCase):
    def test_plain_text_with_counts_and_comments(self):
        text = """
# my binder
4 Llanowar Elves
2x Sol Ring
Krenko, Mob Boss
// trailing comment
1 Llanowar Elves
"""
        with tempfile.TemporaryDirectory() as tmp:
            collection = load_collection(_write(tmp, "cards.txt", text))
        self.assertEqual(collection.counts["Llanowar Elves"], 5)
        self.assertEqual(collection.counts["Sol Ring"], 2)
        self.assertEqual(collection.counts["Krenko, Mob Boss"], 1)
        self.assertEqual(collection.total_cards, 8)

    def test_csv_with_quantity_column(self):
        text = 'Count,Name,Set\n3,"Krenko, Mob Boss",M13\n2,Shock,M20\n'
        with tempfile.TemporaryDirectory() as tmp:
            collection = load_collection(_write(tmp, "cards.csv", text))
        self.assertEqual(collection.counts["Krenko, Mob Boss"], 3)
        self.assertEqual(collection.counts["Shock"], 2)

    def test_csv_detected_without_csv_extension(self):
        text = "Name,Quantity\nOpt,4\n"
        with tempfile.TemporaryDirectory() as tmp:
            collection = load_collection(_write(tmp, "export.txt", text))
        self.assertEqual(collection.counts["Opt"], 4)

    def test_csv_without_name_column_raises(self):
        text = "Foo,Bar\n1,2\n"
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                load_collection(_write(tmp, "bad.csv", text))


if __name__ == "__main__":
    unittest.main()
