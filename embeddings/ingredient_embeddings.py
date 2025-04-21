"""
ingredients_embedder.py
=======================

Embeds Recipe1M+ ingredient lists into fixed‑size dense vectors and (optionally)
builds an approximate–nearest‑neighbour (ANN) index.

Key design points
-----------------
* **Sentence‑Transformers** (`all‑MiniLM‑L6‑v2`) for ingredient‑name semantics.
* **Quantity scalar** (log‐scaled grams) concatenated to each name vector.
* **Mean pooling** → one 385‑d vector per recipe row.
* ANN index: FAISS HNSW if available; falls back to **hnswlib** otherwise.
* Fully instrumented with a module‑level logger.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
from typing import List, Tuple
from fractions import Fraction
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from load_recipe1m_datasets import load_and_merge_recipe_frames

# ---------------------------------------------------------------------#
#  Logger configuration                                                #
# ---------------------------------------------------------------------#
logger = logging.getLogger("ingredients_embedder")
logger.setLevel(logging.INFO)  # default; caller can raise to DEBUG

if not logger.handlers:  # ensure at least one handler exists
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    logger.addHandler(_handler)

# ---------------------------------------------------------------------#
#  Unit–to–gram conversion helpers                                     #
# ---------------------------------------------------------------------#
_UNIT2G = {  # **very** coarse – extend as needed
    "g": 1,
    "gram": 1,
    "grams": 1,
    "kg": 1_000,
    "kilogram": 1_000,
    "mg": 0.001,
    "cup": 240,
    "cups": 240,
    "tbsp": 15,
    "tablespoon": 15,
    "tsp": 5,
    "teaspoon": 5,
    "oz": 28.3495,
    "ounce": 28.3495,
    "lb": 453.592,
}

# regex captures patterns like "2.5cup", "3 tbsp", "150g"
_QTY_RX = re.compile(r"(?P<num>\d+(?:\.\d+)?)(?:\s*)(?P<unit>[a-zA-Z]+)?")


def _qty_to_grams(qty: float, unit: str | None) -> float:
    """
    Convert a ``quantity`` + ``unit`` pair into grams.

    Args:
        qty: Numeric amount parsed from the ingredient string.
        unit: Unit token (case‑insensitive) or *None*.
              Unknown units default to a factor of 1 (i.e. assume grams).

    Returns:
        Grams equivalent of the original amount.

    Example:
        >>> _qty_to_grams(2, "cup")
        480.0
    """
    logger.debug("→ _qty_to_grams(qty=%s, unit=%s)", qty, unit)

    if unit is None:
        grams = qty
    else:
        grams = qty * _UNIT2G.get(unit.lower(), 1.0)

    logger.debug("← _qty_to_grams -> %.2f g", grams)
    return grams


def _qty_to_float(q: str | float | int) -> float:
    """
    Convert a quantity token into a ``float``.

    Supports:
        * simple numbers   → "2", "3.5"
        * pure fractions   → "1/4", "3/8"
        * mixed fractions  → "1 1/2", "2 3/4"

    Args:
        q: Raw quantity (string or already numeric).

    Returns:
        float: Numeric quantity; falls back to 1.0 on parsing failure.
    """
    if isinstance(q, (int, float)):
        return float(q)

    q_str = str(q).strip()
    try:
        # mixed fraction "1 1/2" → ["1", "1/2"]
        if " " in q_str:
            whole, frac = q_str.split()
            return float(whole) + float(Fraction(frac))
        # simple fraction "1/4"
        if "/" in q_str:
            return float(Fraction(q_str))
        # plain or decimal number
        return float(q_str)
    except (ValueError, ZeroDivisionError):
        logger.debug("Quantity '%s' unparsable → default 1.0", q_str)
        return 1.0


# ---------------------------------------------------------------------#
#  Ingredient‑parsing helpers                                           #
# ---------------------------------------------------------------------#
def _sanity(tuples: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    """
    Drop empty ingredient names and clip insane gram values.

    Args:
        tuples: List of (name, grams) tuples.

    Returns:
        Sanitised version of the same list.
    """
    logger.debug("→ _sanity")
    cleaned = []
    for name, grams in tuples:
        if not name:  # skip blanks
            continue
        if grams <= 0 or grams > 10_000:  # >10 kg is unlikely
            logger.debug("Clipping qty %.2f for '%s' to 1 g", grams, name)
            grams = 1.0
        cleaned.append((name, grams))
    logger.debug("← _sanity (%d kept)", len(cleaned))
    return cleaned


def _parse_structured(jstring: str) -> List[Tuple[str, float]]:
    """
    Parse the *ingredients_structured* JSON string.

    For each ingredient dict it extracts:
        • ``ingredient`` (text) → lower‑cased name
        • ``quantity`` + ``unit`` → grams via :func:`_qty_to_grams`

    Handles fractional quantities like ``"1/4"`` or ``"1 1/2"``.

    Args:
        jstring: JSON‑encoded list of ingredient dictionaries.

    Returns:
        List of (ingredient_name, quantity_g) tuples after sanity checks.
    """
    logger.debug("→ _parse_structured (fraction‑aware)")

    data = json.loads(jstring)
    tuples: List[Tuple[str, float]] = []

    for d in data:
        # ---------- extract & clean name ----------
        name = (d.get("ingredient") or "").strip().lower()

        # ---------- convert quantity string → float ----------
        qty_raw = _qty_to_float(d.get("quantity", 1))

        # ---------- convert to grams ----------
        unit = d.get("unit") or "g"
        qty_g = _qty_to_grams(qty_raw, unit)

        tuples.append((name, qty_g))

    cleaned = _sanity(tuples)
    logger.debug("← _parse_structured (%d tuples)", len(cleaned))
    return cleaned


def _parse_fallback(
    names: List[str], raw_text: str
) -> List[Tuple[str, float]]:
    """
    Fallback parser when *ingredients_structured* is unavailable.

    Args:
        names: Cleaned ingredient names from ``ingredients_clean``.
        raw_text: Concatenated free‑form strings from ``ingredients_raw``.

    Returns:
        List of (name, qty_g) derived via a regex heuristic; quantities
        default to 1g when not detectable.
    """
    logger.debug("→ _parse_fallback (len=%d)", len(names))
    tuples: List[Tuple[str, float]] = []
    raw_lower = raw_text.lower()

    # try to find "<number><unit>? <ingredient>" pattern
    for name in names:
        name_clean = name.strip().lower()

        pattern = rf"(\d+(?:\.\d+)?\s*[a-zA-Z]*)\s+{re.escape(name_clean[:10])}"
        match = re.search(pattern, raw_lower)

        if match and (m := _QTY_RX.match(match.group(1))):
            qty = float(m.group("num"))
            unit = m.group("unit") or "g"
            qty_g = _qty_to_grams(qty, unit)
        else:
            qty_g = 1.0  # fallback

        tuples.append((name_clean, qty_g))

    cleaned = _sanity(tuples)
    logger.debug("← _parse_fallback (%d tuples)", len(cleaned))
    return cleaned


# ---------------------------------------------------------------------#
#  Core embedding routine                                              #
# ---------------------------------------------------------------------#
def _embed_ingredients(
    df: pd.DataFrame,
    model_name: str = "all-MiniLM-L6-v2",
) -> Tuple[np.ndarray, List[int]]:
    """
    Convert ingredient columns of each DataFrame row into a 385‑d vector.

    Workflow
    --------
    1. For each recipe choose the best ingredient source:
          ingredients_structured  >  ingredients_clean  >  skip row
    2. Produce (name, qty_g) tuples.
    3. Encode *all* names at once with a Sentence‑Transformer (384d each).
    4. Concatenate log‑scaled qty scalar → 385d vectors.
    5. Mean‑pool these vectors → recipe embedding.

    Args:
        df: DataFrame containing at least the columns
            ``id`` and *one* of the ingredient fields.
        model_name: Huggingface/SBERT model identifier.

    Returns:
        embeddings: 2‑D float32 array with shape (N_recipes, 385).
        recipe_ids: List of recipe IDs (order matches *embeddings*).
    """
    logger.debug("→ _embed_ingredients (rows=%d)", len(df))

    # Load transformer once (CPU is fine; model is <50 MB)
    logger.info("Loading SentenceTransformer '%s' …", model_name)
    embedder = SentenceTransformer(model_name)

    recipe_vecs: List[np.ndarray] = []
    recipe_ids: List[int] = []

    # Iterate row‑wise with nice progress bar
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Embedding recipes", leave=False):
        # --------------------------------------------------------------
        # 1) Select ingredient source
        # --------------------------------------------------------------
        if pd.notnull(row.get("ingredients_structured", np.nan)):
            tuples = _parse_structured(row["ingredients_structured"])
        elif pd.notnull(row.get("ingredients_clean", np.nan)):
            names = json.loads(row["ingredients_clean"])
            tuples = _parse_fallback(names, row.get("ingredients_raw", ""))
        else:
            logger.debug("Row %d skipped (no ingredient info)", idx)
            continue

        if not tuples:
            logger.debug("Row %d skipped (parser returned 0 tuples)", idx)
            continue

        # --------------------------------------------------------------
        # 2) Embed names; concat qty scalar
        # --------------------------------------------------------------
        names = [t[0] for t in tuples]
        qtys = np.array([t[1] for t in tuples], dtype=np.float32)

        name_vecs = embedder.encode(names, convert_to_numpy=True)  # (k, 384)
        qty_scaled = (np.log1p(qtys) / 10.0)[:, None]              # (k, 1)

        fused_ing = np.hstack([name_vecs, qty_scaled])             # (k, 385)

        # --------------------------------------------------------------
        # 3) Aggregate to recipe‑level representation
        # --------------------------------------------------------------
        recipe_vecs.append(fused_ing.mean(axis=0))
        recipe_ids.append(str(row["id"]))

    logger.info("Embedded %d recipes (%.1f%% of DataFrame).",
                len(recipe_vecs), 100 * len(recipe_vecs) / len(df))
    logger.debug("← _embed_ingredients done")
    return np.vstack(recipe_vecs), recipe_ids


# ---------------------------------------------------------------------#
#  Orchestrator (vectors + optional ANN index)                         #
# ---------------------------------------------------------------------#
def run(
    input_df: pd.DataFrame,
    output_dir: str,
    build_index: bool = True,
    prefer_faiss: bool = True,
) -> None:
    """
    Complete pipeline: DataFrame → ``*.npy`` vectors → ANN index (optional).

    Args:
        input_df: Merged Recipe1M+ DataFrame from the loader util.
        output_dir: Target directory (will be created if missing).
        build_index: ``True`` → build ANN index (FAISS > hnswlib fallback).
        prefer_faiss: When *True* attempt FAISS first.

    Raises:
        OSError: When ``output_dir`` cannot be created.
    """
    logger.debug("→ run()")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Embedding stage
    embeddings, recipe_ids = _embed_ingredients(input_df)

    # 2. Persist vectors
    logger.info("Saving numpy arrays to '%s' …", output_dir)
    np.save(os.path.join(output_dir, "recipe_embeddings.npy"), embeddings)
    np.save(os.path.join(output_dir, "recipe_ids.npy"), np.array(recipe_ids))

    # 3. Build ANN index if requested
    if build_index:
        logger.info("Normalising vectors for cosine similarity …")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings_norm = embeddings / norms

        try:
            if prefer_faiss:
                import faiss  # local import

                logger.info("Building FAISS HNSW index …")
                dim = embeddings_norm.shape[1]
                index = faiss.index_factory(dim, "HNSW32", faiss.METRIC_INNER_PRODUCT)
                index.add(embeddings_norm)
                faiss.write_index(index, os.path.join(output_dir, "faiss.index"))
                logger.info("FAISS index saved")
            else:
                raise ImportError  # force fallback branch
        except (ModuleNotFoundError, ImportError):
            import hnswlib

            logger.warning("FAISS unavailable – using hnswlib fallback")
            dim = embeddings_norm.shape[1]
            index = hnswlib.Index(space="cosine", dim=dim)
            index.init_index(
                max_elements=embeddings_norm.shape[0], ef_construction=200, M=32
            )
            index.add_items(embeddings_norm)
            index.save_index(os.path.join(output_dir, "hnswlib.bin"))
            logger.info("hnswlib index saved ✔︎")
    else:
        logger.info("build_index=False → ANN step skipped")

    logger.debug("← run() finished")


# ---------------------------------------------------------------------#
#  CLI entry‑point (for standalone use)                                #
# ---------------------------------------------------------------------#
if __name__ == "__main__":
    '''parser = argparse.ArgumentParser(description="Embed Recipe1M+ ingredients")
    parser.add_argument("merged_dataframe", help="Path to merged .parquet or .csv")
    parser.add_argument("-o", "--output_dir", default="embeddings_out")
    parser.add_argument("--no-index", action="store_true", help="Skip ANN index build")
    parser.add_argument("--no-faiss", action="store_true", help="Force hnswlib index")
    parser.add_argument("-v", "--verbose", action="store_true", help="DEBUG logging")
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Load merged DataFrame
    df_merged = (
        pd.read_parquet(args.merged_dataframe)
        if args.merged_dataframe.endswith(".parquet")
        else pd.read_csv(args.merged_dataframe)
    )'''

    output_dir = "/Users/timseeberger/PycharmProjects/6.C511-Project/embeddings/ingredients/"
    base_dir = "/Users/timseeberger/PycharmProjects/6.C511-Project/data/recipe1m+/"
    df_merged = load_and_merge_recipe_frames(base_dir=base_dir)

    # Execute pipeline
    run(
        input_df=df_merged,
        output_dir=output_dir,
        build_index=False,
        prefer_faiss=False,
    )
