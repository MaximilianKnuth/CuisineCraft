"""
nutrition_embedder.py
=====================
Encode `nutrition_per_100g` dicts into a 7‑d numeric vector in the order:
['calories','fat','saturated_fat','carbs','sugar','protein','salt'].
Missing keys default to 0.
"""
import json, os, logging
from typing import List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("CuisineCraft.NutrEmbed")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(name)s: %(message)s")

NUTR_KEYS = ["energy", "fat", "saturated_fat", "carbs", "sugar", "protein", "salt"]

class NutritionEmbedder:
    """Simple numeric encoder for 7 macro‑nutrition values."""

    DIM = 7

    @staticmethod
    def run_nutrition_embedding(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Extract and standardise per‑100g nutrition dicts.

        Args:
            df: DataFrame with column `nutrition_per_100g`.

        Returns:
            embeddings: float32 *(N, 7)*
            recipe_ids: list[str]
        """
        rows, ids = [], []
        for _, row in df.iterrows():
            nutr = json.loads(row["nutrition_per_100g"])
            vec  = [float(nutr.get(k, 0.0)) for k in NUTR_KEYS]
            rows.append(vec)
            ids.append(str(row["id"]))
        return np.array(rows, dtype=np.float32), ids

    @staticmethod
    def store_embeddings(embeddings: np.ndarray, ids: List[str], out_dir: str) -> None:
        """Persist `.npy` files for nutrition vectors."""
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "nutr_embeddings.npy"), embeddings)
        np.save(os.path.join(out_dir, "recipe_ids.npy"), np.array(ids))
        logger.info("Nutrition embeddings saved to %s", out_dir)
