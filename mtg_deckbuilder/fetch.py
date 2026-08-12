"""Download Scryfall bulk card data (the "Oracle Cards" file).

Scryfall's bulk data is free and updated daily: https://scryfall.com/docs/api/bulk-data
If this machine has no network access, download the file in a browser and
point the tool at it with ``--data`` or the ``MTGDECK_DATA`` env var.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

BULK_INDEX_URL = "https://api.scryfall.com/bulk-data"
DEFAULT_DATA_DIR = Path.home() / ".cache" / "mtg_deckbuilder"
DEFAULT_DATA_PATH = DEFAULT_DATA_DIR / "oracle-cards.json"

_HEADERS = {
    "User-Agent": "mtg-deckbuilder/0.1 (offline collection analysis tool)",
    "Accept": "application/json",
}

MANUAL_INSTRUCTIONS = f"""\
Could not download card data automatically. To set it up manually:

  1. Visit https://scryfall.com/docs/api/bulk-data in a browser.
  2. Download the "Oracle Cards" bulk file (a .json file, ~150 MB).
  3. Save it as: {DEFAULT_DATA_PATH}
     (or anywhere, and pass --data /path/to/oracle-cards.json)
"""


class FetchError(RuntimeError):
    pass


def fetch_bulk_data(dest: Path = DEFAULT_DATA_PATH, timeout: int = 120) -> Path:
    """Download the oracle-cards bulk file to ``dest``. Returns the path."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        request = urllib.request.Request(BULK_INDEX_URL, headers=_HEADERS)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            index = json.load(response)
        oracle = next(
            (e for e in index.get("data", []) if e.get("type") == "oracle_cards"),
            None,
        )
        if oracle is None:
            raise FetchError("Scryfall bulk index had no oracle_cards entry")
        download = urllib.request.Request(oracle["download_uri"], headers=_HEADERS)
        tmp = dest.with_suffix(".part")
        with urllib.request.urlopen(download, timeout=timeout) as response, open(
            tmp, "wb"
        ) as out:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
        tmp.replace(dest)
        return dest
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise FetchError(f"{exc}\n\n{MANUAL_INSTRUCTIONS}") from exc
