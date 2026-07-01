# agent/retriever.py

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


class Retriever:
    """
    Hybrid Retriever

    - BM25 keyword search
    - FAISS semantic retrieval
    - Reciprocal Rank Fusion
    - Metadata filtering
    """

    def __init__(self, data_dir: str = "data"):

        self.data_dir = Path(data_dir)

        # -----------------------------
        # Load embedding model
        # -----------------------------
        with open(self.data_dir / "model_name.pkl", "rb") as f:
            model_name = pickle.load(f)

        self.embed_model = SentenceTransformer(model_name, local_files_only=True)

        # -----------------------------
        # Load FAISS
        # -----------------------------
        self.faiss_index = faiss.read_index(
            str(self.data_dir / "catalog.faiss")
        )

        # -----------------------------
        # Load BM25
        # -----------------------------
        with open(self.data_dir / "bm25.pkl", "rb") as f:
            self.bm25: BM25Okapi = pickle.load(f)

        # -----------------------------
        # Load catalog
        # -----------------------------
        with open(self.data_dir / "catalog_lookup.pkl", "rb") as f:
            self.catalog: list[dict] = pickle.load(f)


        self.url_set = {x["link"] for x in self.catalog}

    # ==========================================================
    # Shared tokenizer
    # ==========================================================

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return re.findall(r"[a-zA-Z0-9\+\#\.]+", text.lower())

    # ==========================================================
    # BM25
    # ==========================================================

    def _bm25_search(
        self,
        query: str,
        top_k: int,
    ) -> tuple[np.ndarray, np.ndarray]:

        tokens = self.tokenize(query)

        scores = self.bm25.get_scores(tokens)

        indices = np.argsort(scores)[::-1][:top_k]

        return scores, indices

    # ==========================================================
    # Semantic Search
    # ==========================================================

    def _semantic_search(
        self,
        query: str,
        top_k: int,
    ) -> tuple[np.ndarray, np.ndarray]:

        embedding = self.embed_model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        embedding = embedding.astype(np.float32)

        scores, indices = self.faiss_index.search(
            embedding,
            top_k,
        )

        return scores[0], indices[0]

    # ==========================================================
    # Reciprocal Rank Fusion
    # ==========================================================

    @staticmethod
    def _rrf(
        bm25_rank: np.ndarray,
        semantic_rank: np.ndarray,
        k: int = 60,
    ) -> dict[int, float]:

        fused = {}

        for rank, idx in enumerate(bm25_rank):

            idx = int(idx)

            fused[idx] = fused.get(idx, 0.0) + 1 / (rank + k)

        for rank, idx in enumerate(semantic_rank):

            idx = int(idx)

            fused[idx] = fused.get(idx, 0.0) + 1 / (rank + k)

        return fused

    # ==========================================================
    # Main Search
    # ==========================================================

    def search(
        self,
        query: str,
        top_k: int = 20,
    ) -> list[dict]:

        _, bm25_rank = self._bm25_search(
            query,
            top_k * 2,
        )

        _, semantic_rank = self._semantic_search(
            query,
            top_k * 2,
        )

        fused = self._rrf(
            bm25_rank,
            semantic_rank,
        )

        ranked = sorted(
            fused.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        results = []

        for idx, score in ranked[:top_k]:

            item = self.catalog[idx].copy()

            item["_retrieval_score"] = round(score, 6)

            results.append(item)

        return results

    # ==========================================================
    # Metadata Filter
    # ==========================================================

    def filter(
        self,
        results: list[dict],
        *,
        job_level: Optional[str] = None,
        remote: Optional[bool] = None,
        language: Optional[str] = None,
        category: Optional[str] = None,
    ) -> list[dict]:

        filtered = results

        if job_level:

            filtered = [
                r
                for r in filtered
                if any(
                    job_level.lower() in level.lower()
                    for level in r.get("job_levels", [])
                )
            ]

        if remote is not None:

            want = "yes" if remote else "no"

            filtered = [
                r
                for r in filtered
                if r.get("remote", "").lower() == want
            ]

        if language:

            filtered = [
                r
                for r in filtered
                if (
                    not r.get("languages")
                    or any(
                        language.lower() in lang.lower()
                        for lang in r.get("languages", [])
                    )
                )
            ]

        if category:

            filtered = [
                r
                for r in filtered
                if any(
                    category.lower() in key.lower()
                    for key in r.get("keys", [])
                )
            ]

        return filtered