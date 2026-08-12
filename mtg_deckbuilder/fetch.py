"""Download Scryfall bulk card data (the "Oracle Cards" file).

Scryfall's bulk data is free and updated daily: https://scryfall.com/docs/api/bulk-data
The bulk index now serves gzipped JSON Lines files (``jsonl_download_uri``);
older indexes served a plain JSON array (``download_uri``). Both are handled,
and the file is stored decompressed so ``CardDatabase.load`` can read it
directly.

If this machine has no network access, download the file in a browser and
point the tool at it with ``--data`` or the ``MTGDECK_DATA`` env var.
"""
from __future__ import annotations

import gzip
import json
import urllib.error
import urllib.request
from pathlib import Path

BULK_INDEX_URL = "https://api.scryfall.com/bulk-data"
DEFAULT_DATA_DIR = Path.home() / ".cache" / "mtg_deckbuilder"
DEFAULT_DATA_PATH = DEFAULT_DATA_DIR / "oracle-cards.jsonl"
# Where older versions of this tool stored the (JSON-array) bulk file.
LEGACY_DATA_PATH = DEFAULT_DATA_DIR / "oracle-cards.json"

_HEADERS = {
    "User-Agent": "mtg-deckbuilder/0.1 (offline collection analysis tool)",
    "Accept": "application/json",
}

MANUAL_INSTRUCTIONS = f"""\
Could not download card data automatically. To set it up manually:

  1. Visit https://scryfall.com/docs/api/bulk-data in a browser.
  2. Download the "Oracle Cards" bulk file (a .jsonl.gz file, ~25 MB).
  3. Decompress it (e.g. `gunzip oracle-cards-*.jsonl.gz`) and save it as:
     {DEFAULT_DATA_PATH}
     (or anywhere, and pass --data /path/to/oracle-cards.jsonl)
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
        uri = oracle.get("jsonl_download_uri") or oracle.get("download_uri")
        if not uri:
            raise FetchError("Scryfall oracle_cards entry had no download URI")
        download = urllib.request.Request(uri, headers=_HEADERS)
        tmp = dest.with_suffix(".part")
        with urllib.request.urlopen(download, timeout=timeout) as response:
            stream = (
                gzip.GzipFile(fileobj=response) if uri.endswith(".gz") else response
            )
            with open(tmp, "wb") as out:
                while True:
                    chunk = stream.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
        tmp.replace(dest)
        return dest
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise FetchError(f"{exc}\n\n{MANUAL_INSTRUCTIONS}") from exc
