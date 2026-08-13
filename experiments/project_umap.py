"""Project the card embeddings to 2D with UMAP for the map.

The 2D picture is a map for exploration - real analysis (similarity,
clustering, recipes) happens in the full-dimensional space; flattened
distances are only approximate.

Reads  experiments/output/embeddings.npy
Writes experiments/output/coords.npy  (float32 [n, 2])
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "output"


def main() -> None:
    import umap

    embeddings = np.load(OUT / "embeddings.npy")
    print("embeddings:", embeddings.shape)
    reducer = umap.UMAP(
        n_neighbors=25,
        min_dist=0.08,
        metric="cosine",
        random_state=42,
        verbose=True,
    )
    coords = reducer.fit_transform(embeddings).astype(np.float32)
    np.save(OUT / "coords.npy", coords)
    print("wrote", OUT / "coords.npy", coords.shape)


if __name__ == "__main__":
    main()
