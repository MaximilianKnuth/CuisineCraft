"""
instructions_embedder.py
========================
Embeds the `instructions_full` column with DistilBERT (512‑d).
"""
import os, logging
from typing import List, Tuple
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("CuisineCraft.InstrEmbed")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(name)s: %(message)s")

class InstructionsEmbedder:
    """Encode free‑text instructions into 512‑d dense vectors."""

    DIM = 512

    def __init__(self, model_name: str = "sentence-transformers/distilbert-base-nli-mean-tokens"):
        """
        Args:
            model_name: Any SBERT/DistilBERT that outputs 512‑d vectors.
        """
        logger.info("Loading instruction encoder '%s' …", model_name)
        self.embedder = SentenceTransformer(model_name)

    def run_instruction_embedding(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Embed every recipe’s `instructions_full` text.

        Args:
            df: DataFrame with columns `id` and `instructions_full`.

        Returns:
            embeddings: float32 *(N, 512)*,
            recipe_ids: list[str]
        """
        texts = df["instructions_full"].tolist()
        ids   = df["id"].astype(str).tolist()
        embs  = self.embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        return embs.astype(np.float32), ids

    @staticmethod
    def store_embeddings(embeddings: np.ndarray, ids: List[str], out_dir: str) -> None:
        """Write embeddings + ids to `.npy` files."""
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "instr_embeddings.npy"), embeddings)
        np.save(os.path.join(out_dir, "recipe_ids.npy"), np.array(ids))
        logger.info("Instruction embeddings saved to %s", out_dir)
