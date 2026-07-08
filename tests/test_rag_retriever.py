import hashlib
import re

import numpy as np
import pytest

from src.rag.chunking import Chunk
from src.rag.index_builder import build_faiss_index
from src.rag.retriever import Retriever

VOCAB_SIZE = 128


def _stable_hash(word: str, mod: int) -> int:
    return int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16) % mod


def _fake_embed_texts(texts: list[str], model: str = "", client=None) -> np.ndarray:
    """OpenAI embedding API'sini taklit eden, deterministik bag-of-words tabanli sahte embedding
    (bkz. tests/test_rag_index_builder.py - ayni yaklasim, network/API key gerektirmez)."""
    vectors = []
    for text in texts:
        vec = np.zeros(VOCAB_SIZE, dtype=np.float32)
        for word in re.findall(r"\w+", text.lower()):
            vec[_stable_hash(word, VOCAB_SIZE)] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        vectors.append(vec)
    return np.array(vectors, dtype=np.float32)


@pytest.fixture(autouse=True)
def _patch_embeddings(monkeypatch):
    monkeypatch.setattr("src.rag.index_builder.embed_texts", _fake_embed_texts)
    monkeypatch.setattr("src.rag.retriever.embed_texts", _fake_embed_texts)


def make_chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id="salt::0",
            doc_id="salt",
            title="Tuz ve Sodyum",
            section="Ozet",
            text="WHO yetiskinler icin gunluk 5 gramdan az tuz ve sodyum onerir.",
            verified=True,
            source="WHO",
            source_url=None,
        ),
        Chunk(
            chunk_id="sugar::0",
            doc_id="sugar",
            title="Serbest Seker",
            section="Ozet",
            text="Serbest sekerin gunluk enerjinin yuzde 10'undan az olmasi onerilir.",
            verified=True,
            source="WHO",
            source_url=None,
        ),
        Chunk(
            chunk_id="allergen::0",
            doc_id="allergen",
            title="Alerjenler",
            section="Liste",
            text="Bu projede laktoz, gluten, findik, soya, yumurta, balik alerjenleri izlenir.",
            verified=False,
            source=None,
            source_url=None,
        ),
    ]


@pytest.fixture
def retriever():
    chunks = make_chunks()
    index = build_faiss_index(chunks)
    return Retriever(chunks, index)


def test_retrieve_returns_top_k_results(retriever):
    results = retriever.retrieve("tuz alimi ne kadar olmali?", top_k=2)
    assert len(results) == 2


def test_retrieve_ranks_most_relevant_chunk_first_for_salt_query(retriever):
    results = retriever.retrieve("gunluk sodyum ve tuz miktari nedir?", top_k=1)
    assert results[0].chunk.chunk_id == "salt::0"


def test_retrieve_ranks_most_relevant_chunk_first_for_sugar_query(retriever):
    results = retriever.retrieve("serbest seker enerji yuzdesi kac olmali?", top_k=1)
    assert results[0].chunk.chunk_id == "sugar::0"


def test_retrieve_bm25_helps_exact_keyword_match(retriever):
    # "findik" tam kelime eslesmesi BM25'te guclu olmali
    results = retriever.retrieve("findik alerjeni hangi urunlerde bulunur?", top_k=1)
    assert results[0].chunk.chunk_id == "allergen::0"


def test_retrieve_respects_min_score_threshold(retriever):
    results = retriever.retrieve("tamamen alakasiz bir konu: uzay arastirmalari", top_k=3, min_score=0.99)
    assert results == []


def test_retrieve_scores_are_descending(retriever):
    results = retriever.retrieve("beslenme onerileri nelerdir?", top_k=3)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
