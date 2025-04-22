"""
ingredients_embedder_class.py
=============================

Ingredient–embedding pipeline refactored into a class with **Google‑style
docstrings** (Args, Returns, Raises).

• Sentence‑BERT (`all‑MiniLM‑L6‑v2`) encodes ingredient names (384 d)
• Log‑scaled quantity scalar appended (+1 d) → 385 d / ingredient
• Mean‑pool across set → 385 d / recipe
• Helper `store_embeddings()` writes `.npy` artefacts
"""

import json
import logging
import os
import re
from fractions import Fraction
from typing import List, Tuple
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# -------------------------- logging ---------------------------------
logger = logging.getLogger("CuisineCraft.IngEmbed")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(name)s: %(message)s")

# -------------------------- constants -------------------------------
_UNIT2G = {
    "g": 1, "gram": 1, "grams": 1,
    "kg": 1000, "kilogram": 1000,
    "mg": 0.001,
    "cup": 240, "cups": 240,
    "tbsp": 15, "tablespoon": 15,
    "tsp": 5, "teaspoon": 5,
    "oz": 28.3495, "ounce": 28.3495,
    "lb": 453.592,
}
_QTY_RX = re.compile(r"(?P<num>\d+(?:\.\d+)?)(?:\s*)(?P<unit>[a-zA-Z]+)?")


class IngredientsEmbedder:
    """
    Compute dense ingredient embeddings for Recipe1M+ rows.

    Attributes
    ----------
    model_name : str
        HuggingFace identifier of the Sentence‑Transformer backbone.
    embedder : sentence_transformers.SentenceTransformer
        Loaded model instance used for all name encodings.
    """

    DIM_NAME = 384
    DIM_TOTAL = 385  # 384 + quantity scalar

    # -----------------------------------------------------------------
    # constructor
    # -----------------------------------------------------------------
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """
        Instantiate the embedder and load SBERT.

        Args:
            model_name: Pre‑trained SBERT model ID.  Defaults to
                ``"all-MiniLM-L6-v2"``.
        """
        logger.info("Loading Sentence‑Transformer '%s' …", model_name)
        self.model_name = model_name
        self.embedder = SentenceTransformer(model_name)

    # -----------------------------------------------------------------
    # static helpers for quantities
    # -----------------------------------------------------------------
    @staticmethod
    def _qty_to_float(q) -> float:
        """
        Convert quantity token to float.

        Supports integers, decimals, pure and mixed fractions.

        Args:
            q: Raw quantity token (str | int | float).

        Returns:
            float: Parsed number (default 1.0 on failure).
        """
        if isinstance(q, (int, float)):
            return float(q)
        s = str(q).strip()
        try:
            if " " in s:
                whole, frac = s.split()
                return float(whole) + float(Fraction(frac))
            if "/" in s:
                return float(Fraction(s))
            return float(s)
        except (ValueError, ZeroDivisionError):
            logger.debug("Quantity '%s' unparsable → 1.0", s)
            return 1.0

    @staticmethod
    def _qty_to_grams(qty: float, unit: str | None) -> float:
        """Convert numeric qty + unit into grams."""
        return qty * _UNIT2G.get(unit.lower(), 1.0) if unit else qty

    @staticmethod
    def _sanitise(tuples: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Remove blank names and clip extreme gram values."""
        return [(n, g if 0 < g <= 10_000 else 1.0) for n, g in tuples if n]

    # -----------------------------------------------------------------
    # row‑level parsers
    # -----------------------------------------------------------------
    def _parse_structured(self, jstring: str) -> List[Tuple[str, float]]:
        """
        Parse ``ingredients_structured`` JSON string.

        Args:
            jstring: JSON‑encoded list[dict] with keys
                ``ingredient``, ``quantity``, ``unit``.

        Returns:
            List of (name, grams) tuples.
        """
        data = json.loads(jstring)
        tuples = []
        for d in data:
            name = (d.get("ingredient") or "").strip().lower()
            qty_f = self._qty_to_float(d.get("quantity", 1))
            grams = self._qty_to_grams(qty_f, d.get("unit") or "g")
            tuples.append((name, grams))
        return self._sanitise(tuples)

    def _parse_fallback(self,
                        names: List[str],
                        raw_text: str) -> List[Tuple[str, float]]:
        """
        Heuristic parser for rows lacking structured quantities.

        Args:
            names: Clean ingredient tokens.
            raw_text: Concatenated free‑text ingredient phrases.

        Returns:
            List of (name, grams) tuples (grams may default to 1).
        """
        raw = raw_text.lower()
        tuples = []
        for name in names:
            n = name.strip().lower()
            m = re.search(rf"(\d+(?:\.\d+)?\s*[a-zA-Z]*)\s+{re.escape(n[:10])}", raw)
            if m and (g := _QTY_RX.match(m.group(1))):
                grams = self._qty_to_grams(float(g.group("num")), g.group("unit") or "g")
            else:
                grams = 1.0
            tuples.append((n, grams))
        return self._sanitise(tuples)

    # -----------------------------------------------------------------
    # public API
    # -----------------------------------------------------------------
    def run_ingredient_embedding(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Generate 385‑d recipe embeddings.

        Args:
            df: Merged Recipe1M+ DataFrame with columns ``id`` and either
                ``ingredients_structured`` *or*
                (``ingredients_clean`` and ``ingredients_raw``).

        Returns:
            Tuple[np.ndarray, List[str]]:
                • embeddings  –float32 array shape (N,385)
                • recipe_ids –hexadecimal IDs aligned with embeddings
        """
        recipe_vecs: List[np.ndarray] = []
        ids: List[str] = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Embedding recipes"):
            # choose data source
            if pd.notnull(row.get("ingredients_structured", np.nan)):
                tuples = self._parse_structured(row["ingredients_structured"])
            elif pd.notnull(row.get("ingredients_clean", np.nan)):
                tuples = self._parse_fallback(json.loads(row["ingredients_clean"]),
                                              row.get("ingredients_raw", ""))
            else:
                continue
            if not tuples:
                continue

            # encode names + quantity scalar
            names = [t[0] for t in tuples]
            qtys = np.array([t[1] for t in tuples], dtype=np.float32)
            name_vecs = self.embedder.encode(names, convert_to_numpy=True, show_progress_bar=True)
            qty_scaled = (np.log1p(qtys) / 10.0)[:, None]
            fused = np.hstack([name_vecs, qty_scaled])              # (k, 385)

            recipe_vecs.append(fused.mean(axis=0).astype(np.float32))
            ids.append(str(row["id"]))

        return np.vstack(recipe_vecs), ids

    @staticmethod
    def store_embeddings(embeddings: np.ndarray,
                         ids: List[str],
                         out_dir: str) -> None:
        """
        Save embeddings and IDs to disk.

        Args:
            embeddings: Output array from `run_ingredient_embedding`.
            ids: Matching recipe ID list.
            out_dir: Destination directory (created if missing).

        Raises:
            OSError: If `out_dir` cannot be created.
        """
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "recipe_embeddings.npy"), embeddings)
        np.save(os.path.join(out_dir, "recipe_ids.npy"), np.array(ids))
        logger.info("Embeddings saved to %s", out_dir)

