"""Hibrit retriever: FAISS yogun (dense, OpenAI embedding API) benzerlik + BM25 seyrek
(sparse) skorunun agirlikli birlesimi.

Oneri formu 2.4: "Retriever (FAISS/BM25)". Bilgi tabani kucuk olcekli oldugundan (dokuman
basina birkaç chunk), her sorguda tum chunk'lar uzerinde tam tarama yapmak (brute-force)
performans sorunu yaratmaz; bu yuzden FAISS IndexFlatIP (exact search) + rank_bm25 BM25Okapi
(tam corpus) tercih edilmistir - ANN (approximate) index'e gecmek bu olcekte gereksizdir.

Dense embedding (OpenAI API, yerel model YOK - bkz. src/rag/embeddings.py), es anlamli/parafraz
sorgularda (ornegin "tuz fazla mi?" ~ "sodyum yuksek mi?") guclu; BM25 ise Turkce esik degeri
gibi tam kelime/sayi eslesmelerinde (ornegin "22.5" veya "who_salt_daily_intake_g") daha guclu
oldugundan ikisi birlikte kullanilir.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from src.rag.chunking import Chunk
from src.rag.embeddings import DEFAULT_EMBEDDING_MODEL, embed_texts
from src.rag.index_builder import DEFAULT_INDEX_DIR, load_index

DEFAULT_DENSE_WEIGHT = 0.6
DEFAULT_BM25_WEIGHT = 0.4


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float
    dense_score: float
    bm25_score: float


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _min_max_normalize(scores: np.ndarray) -> np.ndarray:
    lo, hi = scores.min(), scores.max()
    if hi - lo < 1e-9:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


class Retriever:
    def __init__(
        self,
        chunks: list[Chunk],
        index,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        embedding_client=None,
        dense_weight: float = DEFAULT_DENSE_WEIGHT,
        bm25_weight: float = DEFAULT_BM25_WEIGHT,
    ) -> None:
        self.chunks = chunks
        self.index = index
        self.embedding_model = embedding_model
        self.embedding_client = embedding_client
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self._bm25 = BM25Okapi([_tokenize(c.embedding_text) for c in chunks])

    @classmethod
    def load(
        cls,
        index_dir: Path = DEFAULT_INDEX_DIR,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        **kwargs,
    ) -> "Retriever":
        index, chunks = load_index(index_dir)
        return cls(chunks, index, embedding_model=embedding_model, **kwargs)

    def _dense_scores(self, query: str) -> np.ndarray:
        query_vec = embed_texts([query], model=self.embedding_model, client=self.embedding_client)
        n = len(self.chunks)
        similarities, indices = self.index.search(query_vec, n)
        scores = np.zeros(n, dtype=np.float32)
        for score, idx in zip(similarities[0], indices[0]):
            if idx != -1:
                scores[idx] = score
        return scores

    def _bm25_scores(self, query: str) -> np.ndarray:
        return np.asarray(self._bm25.get_scores(_tokenize(query)), dtype=np.float32)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float | None = None,
    ) -> list[RetrievalResult]:
        """Sorgu icin en alakali top_k chunk'i, hibrit skora gore azalan sirada doner.

        min_score verilirse (0-1 araliginda, birlesik skor uzerinde), bu esigin altindaki
        sonuclar elenir - alakasiz/dusuk guvenli chunk'larin generation'a sizmasini onler.
        """
        dense_raw = self._dense_scores(query)
        bm25_raw = self._bm25_scores(query)

        dense_norm = _min_max_normalize(dense_raw)
        bm25_norm = _min_max_normalize(bm25_raw)
        combined = self.dense_weight * dense_norm + self.bm25_weight * bm25_norm

        order = np.argsort(-combined)
        results = []
        for idx in order[:top_k]:
            if min_score is not None and combined[idx] < min_score:
                continue
            results.append(
                RetrievalResult(
                    chunk=self.chunks[idx],
                    score=float(combined[idx]),
                    dense_score=float(dense_raw[idx]),
                    bm25_score=float(bm25_raw[idx]),
                )
            )
        return results
