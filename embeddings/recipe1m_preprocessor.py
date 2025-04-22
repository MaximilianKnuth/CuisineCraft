"""
recipe1m_preprocessor.py
========================
Recipe1M+ data–cleaning utility collected in **RecipePreprocessor**.

* Lower‑cases & de‑noises titles and instructions.
* Builds a single `text_for_embedding` column (title + instructions).
* Extracts & standardises numeric nutrition vectors (energy, fat, …).
* Maps FSA traffic‑light colours to ordinal integers and packs them
  into a fixed‑length vector.

Google‑style docstrings for every public method.
"""

import logging
import re
import json
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("CuisineCraft.Preprocessor")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(name)s: %(message)s")


class Recipe1MPlusPreprocessor:
    """Apply text and numeric cleaning to a merged Recipe1M+ DataFrame."""

    # canonical nutrition ordering (7 values)
    NUTR_KEYS: List[str] = [
        "energy", # kcal or kJ ‑ whatever recipe1m uses
        "fat",
        "saturates",
        "carbohydrate",
        "sugars",
        "protein",
        "salt",
    ]

    # FSA four‑nutrient traffic‑light scheme
    FSA_KEYS: List[str] = ["fat", "saturates", "sugars", "salt"]
    COLOR_MAP: Dict[str, int] = {"green": 0, "amber": 1, "orange": 1, "red": 2}

    # -----------------------------------------------------------------
    # public orchestrator
    # -----------------------------------------------------------------
    def run_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run all cleaning sub‑routines in sequence.

        Args:
            df (pd.DataFrame): Output of `load_and_merge_recipe_frames`
                containing at least columns `title`, `instructions_full`,
                `nutrition_per_100g`, `fsa_lights_per_100g`.

        Returns:
            pd.DataFrame: Cleaned frame with new columns
              • `clean_title`, `clean_instructions`, `text_for_embedding`
              • `nutrition_vector` (7‑d list[float])
              • `fsa_vector` (4‑d list[int])
        """
        logger.info("Preprocessing %d rows …", len(df))
        df = df.copy()

        # 1. clean free‑text
        df["clean_title"] = df["title"].apply(self._clean_text)
        df["clean_instructions"] = df["instructions_full"].apply(self._clean_text)
        df["text_for_embedding"] = df["clean_title"] + " " + df["clean_instructions"]

        # 2. nutrition vectors
        df["nutrition_vector"] = df["nutrition_per_100g"].apply(self._extract_nutrition)

        # scale nutrition across dataset (z‑score)
        nutr_mat = np.vstack(df["nutrition_vector"].to_numpy())
        scaler = StandardScaler()
        nutr_scaled = scaler.fit_transform(nutr_mat)
        df["nutrition_vector"] = nutr_scaled.tolist()

        # 3. FSA traffic lights
        df["fsa_vector"] = df["fsa_lights_per_100g"].apply(self._extract_fsa)

        # 4. drop recipes with empty instructions after cleaning
        before = len(df)
        df = df[df["clean_instructions"].astype(bool)].reset_index(drop=True)
        logger.info("Dropped %d rows with empty instructions after cleaning.", before - len(df))

        return df

    # -----------------------------------------------------------------
    # text cleaning
    # -----------------------------------------------------------------
    @staticmethod
    def _clean_text(text: str) -> str:
        """Aggressive text normalisation.

        Args:
            text: Raw string (can be NaN/None).

        Returns:
            Cleaned lower‑case string.
        """
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r"http\S+", "", text)          # remove URLs
        text = re.sub(r"<.*?>", "", text)            # strip HTML
        text = re.sub(r"[^\w\s.,!?;:]", " ", text)   # drop special chars
        text = re.sub(r"([.,!?;:])\1+", r"\1", text) # dedup punctuation
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # -----------------------------------------------------------------
    # numeric nutrition parsing
    # -----------------------------------------------------------------
    def _extract_nutrition(self, jstr: str) -> List[float]:
        """Parse `nutrition_per_100g` JSON → ordered numeric list."""
        try:
            d = json.loads(jstr)
        except Exception:
            d = {}
        return [float(d.get(k, 0.0)) for k in self.NUTR_KEYS]

    # -----------------------------------------------------------------
    # FSA traffic‑light mapping
    # -----------------------------------------------------------------
    def _extract_fsa(self, jstr: str) -> List[int]:
        """Parse FSA lights dict → 4‑element ordinal list."""
        try:
            d = json.loads(jstr)
        except Exception:
            d = {}
        vec = []
        for key in self.FSA_KEYS:
            # tolerate mixed case in keys
            val = next((d.get(k_alt) for k_alt in (key, key.capitalize(), key.upper()) if k_alt in d), "green")
            vec.append(self.COLOR_MAP.get(str(val).lower(), 0))
        return vec
