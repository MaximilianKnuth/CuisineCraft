"""
fsa_embedder.py
===============
Encode UK/IE FSA traffic‑light labels per 100g into a 7×3 one‑hot
flattened to 21‑d vector, ordered like `NUTR_KEYS` in nutrition_embedder.

Class labels: green=0, amber=1, red=2.
"""
import json, os, logging
from typing import List, Tuple

import numpy as np
import pandas as pd


logger = logging.getLogger("CuisineCraft.FSAEmbed")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(name)s: %(message)s")

NUTR_KEYS = ['energy', 'fat', 'saturates', 'carbohydrate',
             'sugars', 'fibre', 'protein', 'salt']
FSA_KEYS = ['fat', 'saturates', 'sugars', 'salt']
LABEL2IDX = {"green": 0, "amber": 1, "red": 2}

class FsaEmbedder:
    """Ordinal one‑hot encoder for FSA traffic‑light labels (21d)."""

    DIM = len(NUTR_KEYS) * 3  # 7 * 3 = 21

    @staticmethod
    def run_fsa_embedding(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Convert traffic‑light dicts into 21‑d one‑hot vectors."""
        rows, ids = [], []
        for _, row in df.iterrows():
            fsa = json.loads(row["fsa_lights_per_100g"])
            vec = np.zeros((len(NUTR_KEYS), 3), dtype=np.float32)
            for i, k in enumerate(NUTR_KEYS):
                label = fsa.get(k, "green")  # default safest
                vec[i, LABEL2IDX.get(label, 0)] = 1.0
            rows.append(vec.flatten())
            ids.append(str(row["id"]))
        return np.vstack(rows), ids

    @staticmethod
    def store_embeddings(embeddings: np.ndarray, ids: List[str], out_dir: str) -> None:
        """Save FSA embeddings + IDs to disk."""
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "fsa_embeddings.npy"), embeddings)
        np.save(os.path.join(out_dir, "recipe_ids.npy"), np.array(ids))
        logger.info("FSA embeddings saved to %s", out_dir)
