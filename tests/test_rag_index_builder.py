import hashlib
import re
from pathlib import Path

import numpy as np
import pytest

from src.rag.chunking import Chunk
from src.rag.index_builder import build_faiss_index, load_index, save_index

VOCAB_SIZE = 128


def _stable_hash(word: str, mod: int) -> int:
    return int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16) % mod


def _fake_embed_texts(texts: list[str], provider: str = "", model: str = "", client=None) -> np.ndarray:
    """OpenAI embedding API'sini taklit eden, deterministik bag-of-words tabanli sahte embedding.

    Gercek API cagrisi yapmadan (network/API key gerektirmeden) retrieval siralama mantigini
    test etmeyi saglar - kelime ortusmesi arttikca cosine benzerligi de artar.
    """
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


def make_chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id="doc_a::0",
            doc_id="doc_a",
            title="Tuz ve Sodyum",
            section="Ozet",
            text="WHO gunluk 5 gramdan az tuz onerir.",
            verified=True,
            source="WHO",
            source_url=None,
        ),
        Chunk(
            chunk_id="doc_b::0",
            doc_id="doc_b",
            title="Seker Alimi",
            section="Ozet",
            text="Serbest sekerin enerjinin yuzde 10'undan az olmasi onerilir.",
            verified=True,
            source="WHO",
            source_url=None,
        ),
    ]


def test_build_faiss_index_returns_index_with_correct_count():
    chunks = make_chunks()
    index = build_faiss_index(chunks)
    assert index.ntotal == len(chunks)


def test_build_faiss_index_rejects_empty_chunk_list():
    with pytest.raises(ValueError):
        build_faiss_index([])


def test_save_and_load_index_roundtrip(tmp_path: Path):
    chunks = make_chunks()
    index = build_faiss_index(chunks)
    save_index(index, chunks, out_dir=tmp_path)

    loaded_index, loaded_chunks = load_index(tmp_path)

    assert loaded_index.ntotal == len(chunks)
    assert [c.chunk_id for c in loaded_chunks] == [c.chunk_id for c in chunks]
    assert loaded_chunks[0].text == chunks[0].text
    assert loaded_chunks[0].verified == chunks[0].verified


def test_search_returns_most_similar_chunk_first(tmp_path: Path):
    chunks = make_chunks()
    index = build_faiss_index(chunks)
    save_index(index, chunks, out_dir=tmp_path)
    loaded_index, loaded_chunks = load_index(tmp_path)

    from src.rag.index_builder import embed_texts

    query_vec = embed_texts(["WHO gunluk tuz onerisi ne kadardir?"])
    similarities, indices = loaded_index.search(query_vec, 1)

    best_chunk = loaded_chunks[indices[0][0]]
    assert best_chunk.chunk_id == "doc_a::0"
