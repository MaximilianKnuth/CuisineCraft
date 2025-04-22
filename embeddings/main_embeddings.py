"""
main_create_embeddings.py
=========================

Master script that orchestrates the complete preprocessing‑and‑embedding
workflow for four modalities of Recipe1M+:

    • Ingredients  (385 d / recipe)
    • Instructions (512 d / recipe)
    • Nutrition    (7 d  / recipe)
    • FSA lights   (21 d / recipe)

Outputs
-------
Each embedder writes two `.npy` files in its own sub‑directory:

    <out_root>/
        ingredients/
            recipe_embeddings.npy   (N, 385)
            recipe_ids.npy          (N,)
        instructions/
            instr_embeddings.npy    (N, 512)
            recipe_ids.npy
        nutrition/
            nutr_embeddings.npy     (N, 7)
            recipe_ids.npy
        fsa/
            fsa_embeddings.npy      (N, 21)
            recipe_ids.npy

The script also returns these arrays in Python‐scope so that users who
`import` this module can access them programmatically.
"""

import argparse
import logging
import os

from load_recipe1m_datasets import load_and_merge_recipe_frames
from recipe1m_preprocessor import Recipe1MPlusPreprocessor
from ingredient_embedder import IngredientsEmbedder
from instructions_embedder import InstructionsEmbedder
from nutritions_embedder import NutritionEmbedder
from fsa_embedder import FsaEmbedder

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(name)s: %(message)s")
logger = logging.getLogger("CuisineCraft.Main")


def main(base_dir: str, out_root: str) -> None:
    """Run full pipeline end‑to‑end.

    Args:
        base_dir: Folder containing the three Recipe1M+ JSON layers.
        out_root: Directory to store all embedding sub‑folders.
    """

    # ------------------------------------------------------------------
    # 1. Load + preprocess dataframe
    # ------------------------------------------------------------------
    logger.info("Step 1/5 —Load & merge raw JSON layers")
    df_raw = load_and_merge_recipe_frames(base_dir)
    logger.info("Merged frame shape: %s", df_raw.shape)

    logger.info("Step 2/5 —Preprocess")
    df_clean = Recipe1MPlusPreprocessor().run_preprocessing(df_raw)

    # ------------------------------------------------------------------
    # 2. Ingredient embeddings
    # ------------------------------------------------------------------
    logger.info("Step 3/5 —Ingredient embeddings")
    ing_embedder = IngredientsEmbedder()
    ing_vecs, ids_ing = ing_embedder.run_ingredient_embedding(df_clean)
    ing_embedder.store_embeddings(ing_vecs, ids_ing, os.path.join(out_root, "ingredients"))

    # ------------------------------------------------------------------
    # 3. Instruction embeddings
    # ------------------------------------------------------------------
    logger.info("Step 4/5 —Instruction embeddings")
    instr_embedder = InstructionsEmbedder()
    instr_vecs, ids_instr = instr_embedder.run_instruction_embedding(df_clean)
    instr_embedder.store_embeddings(instr_vecs, ids_instr, os.path.join(out_root, "instructions"))

    # ------------------------------------------------------------------
    # 4. Nutrition & FSA embeddings
    # ------------------------------------------------------------------
    logger.info("Step 5/5 —Nutrition & FSA embeddings")
    nutr_embedder = NutritionEmbedder()
    nutr_vecs, ids_nutr = nutr_embedder.run_nutrition_embedding(df_clean)
    nutr_embedder.store_embeddings(nutr_vecs, ids_nutr, os.path.join(out_root, "nutrition"))

    fsa_embedder = FsaEmbedder()
    fsa_vecs, ids_fsa = fsa_embedder.run_fsa_embedding(df_clean)
    fsa_embedder.store_embeddings(fsa_vecs, ids_fsa, os.path.join(out_root, "fsa"))

    # ------------------------------------------------------------------
    # Verify all ID sequences match (sanity check)
    # ------------------------------------------------------------------
    assert ids_ing == ids_instr == ids_nutr == ids_fsa, "Recipe‑ID sets diverged!"

    logger.info("All embeddings computed and stored.")


if __name__ == "__main__":
    base_dir = "/Users/timseeberger/PycharmProjects/6.C511-Project/data/recipe1m+"
    out_dir = "/Users/timseeberger/PycharmProjects/6.C511-Project/embeddings"

    main(base_dir, out_dir)
