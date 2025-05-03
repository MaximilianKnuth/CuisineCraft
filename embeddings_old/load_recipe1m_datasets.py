"""
Utility to load the three core Recipe1M+ JSON layers and merge them
into a single tidy ``pandas.DataFrame`` that is compatible with the
ingredient‑embedding pipeline.

The merge is performed on the unique recipe ``id``.  Columns containing
lists or dictionaries are **JSON‑stringified** so they can safely survive
CSV/Parquet round‑trips and be parsed later with ``json.loads``.
"""

from __future__ import annotations
import json
import os
from typing import Dict, List
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


# -------------------------------------------------------------------- #
#                           helper functions                           #
# -------------------------------------------------------------------- #
def _parse_nutrition_entry(entry: Dict) -> Dict:
    """
    Convert a single dict from *recipes_with_nutritional_info.json*
    into a *thin* dictionary containing only the fields required by later
    stages (embedding, filtering, etc.).

    Args:
        entry (dict): Raw JSON object for one recipe.

    Returns:
        dict: Keys
            - id (int)
            - ingredients_structured (list[dict])
            - nutrition_per_100g (dict[str, float])
            - fsa_lights_per_100g (dict[str, str])
    """
    n = len(entry["ingredients"])

    # Build per‑ingredient records with robust index checks
    ingredients_structured: List[Dict] = []
    for i in range(n):
        ingredients_structured.append(
            {
                "ingredient": entry["ingredients"][i]["text"]
                if i < len(entry["ingredients"])
                else None,
                "quantity": entry["quantity"][i]["text"] if i < len(entry["quantity"]) else None,
                "unit": entry["unit"][i]["text"] if i < len(entry["unit"]) else None,
                "weight_g": entry["weight_per_ingr"][i] if i < len(entry["weight_per_ingr"]) else None,
                "nutrition": entry["nutr_per_ingredient"][i]
                if i < len(entry["nutr_per_ingredient"])
                else None,
            }
        )

    return {
        "id": entry["id"],
        "ingredients_structured": ingredients_structured,
        "nutrition_per_100g": entry.get("nutr_values_per100g", {}),
        "fsa_lights_per_100g": entry.get("fsa_lights_per100g", {}),
    }


def load_and_merge_recipe_frames(base_dir: str) -> pd.DataFrame:
    """
    Load *layer1.json*, *det_ingrs.json*, and
    *recipes_with_nutritional_info.json* from ``base_dir``, transform each
    into a compact DataFrame, and inner‑merge them on the recipe ``id``.

    The final DataFrame is ready to be consumed by the ingredient‑embedding
    script (`ingredients_embedder.py`).  All list/dict columns are JSON
    strings so that downstream code can simply call ``json.loads``.

    Args:
        base_dir (str):
            Directory that contains the three original Recipe1M+ files::

                layer1.json
                det_ingrs.json
                recipes_with_nutritional_info.json

    Returns:
        pandas.DataFrame:
            One row per recipe.  Guaranteed columns:

            * id (int)
            * title (str)
            * ingredients_raw (str  – JSON list)
            * instructions_full (str)
            * ingredients_clean (str – JSON list, may be NaN)
            * ingredients_structured (str – JSON list[dict], may be NaN)
            * nutrition_per_100g (str – JSON dict, may be NaN)
            * fsa_lights_per_100g (str – JSON dict, may be NaN)

    Example:
        >>> df = load_and_merge_recipe_frames("recipe1m_downloads")
        >>> df.head(1)
    """
    # ---------- ingredients_layer --------------------------------------
    with open(os.path.join(base_dir, "instructions_layer.json"), encoding="utf-8") as f:
        instructions_layer_json = json.load(f)

    df_instructions_layer = pd.DataFrame(
        {
            "id": r["id"],
            "title": r["title"],
            "ingredients_raw": [i["text"] for i in r.get("ingredients", [])],
            "instructions_full": " ".join(s["text"] for s in r.get("instructions", [])),
        }
        for r in instructions_layer_json
    )

    # ---------- detected‑ingredients ----------------------------------
    with open(os.path.join(base_dir, "ingredients_layer.json"), encoding="utf-8") as f:
        ingredients_layer_json = json.load(f)

    df_ingredients_layer = pd.DataFrame(
        {
            "id": r["id"],
            "ingredients_clean": [
                ing["text"]
                for ing, valid in zip(r["ingredients"], r["valid"])
                if valid
            ],
        }
        for r in ingredients_layer_json
    )

    # ---------- nutrition layer ---------------------------------------
    with open(
        os.path.join(base_dir, "recipes_with_nutritional_info.json"),
        encoding="utf-8",
    ) as f:
        nutr_json = json.load(f)

    df_nutr = pd.DataFrame(_parse_nutrition_entry(r) for r in nutr_json)

    # ---------- merge + tidy ------------------------------------------
    df = (
        df_instructions_layer.merge(df_ingredients_layer, on="id", how="inner")
        .merge(df_nutr, on="id", how="inner")
        .drop_duplicates(subset="id")
        .reset_index(drop=True)
    )

    # ---------- stringify complex cols --------------------------------
    complex_cols = (
        "ingredients_raw",
        "ingredients_clean",
        "ingredients_structured",
        "nutrition_per_100g",
        "fsa_lights_per_100g",
    )
    for col in complex_cols:
        if col in df.columns:
            df[col] = df[col].apply(json.dumps)

    return df
