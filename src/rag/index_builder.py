"""Bilgi tabani chunk'larindan FAISS yogun (dense) embedding index'i olusturur.

Oneri formu 2.4: "Retriever (FAISS/BM25)". Burada FAISS tarafi + kalici saklama (index +
chunk metadatasi) implemente edilir; BM25 tarafi src/rag/retriever.py icinde hibrit skor
icin ayrica calisir (bu modulden bagimsizdir, kendi rank_bm25 corpus'unu kurar).

Embedding OpenAI API'sinden alinir (bkz. src/rag/embeddings.py) - yerel bir model diske
inmez. Cosine benzerligi icin embedding vektorleri L2-normalize edilip IndexFlatIP kullanilir
(normalize edilmis vektorlerde ic carpim = cosine benzerligi).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import faiss

from src.rag.chunking import Chunk, chunk_knowledge_base
from src.rag.embeddings import DEFAULT_EMBEDDING_MODEL, embed_texts

DEFAULT_INDEX_DIR = Path(__file__).resolve().parents[2] / "data" / "knowledge_base" / "index"


def build_faiss_index(
    chunks: list[Chunk], model: str = DEFAULT_EMBEDDING_MODEL, client=None
) -> faiss.Index:
    """Chunk listesinden IndexFlatIP (cosine benzerligi) FAISS index'i olusturur."""
    if not chunks:
        raise ValueError("Bos chunk listesinden index olusturulamaz")

    embeddings = embed_texts([c.embedding_text for c in chunks], model=model, client=client)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index


def save_index(index: faiss.Index, chunks: list[Chunk], out_dir: Path = DEFAULT_INDEX_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / "index.faiss"))
    chunks_data = [asdict(c) for c in chunks]
    (out_dir / "chunks.json").write_text(
        json.dumps(chunks_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_index(out_dir: Path = DEFAULT_INDEX_DIR) -> tuple[faiss.Index, list[Chunk]]:
    index = faiss.read_index(str(out_dir / "index.faiss"))
    chunks_data = json.loads((out_dir / "chunks.json").read_text(encoding="utf-8"))
    chunks = [Chunk(**d) for d in chunks_data]
    return index, chunks


def build_and_save_index(
    docs_dir: Path | None = None,
    out_dir: Path = DEFAULT_INDEX_DIR,
    model: str = DEFAULT_EMBEDDING_MODEL,
) -> tuple[faiss.Index, list[Chunk]]:
    """Uctan uca: dokumanlari chunk'la -> embed et (OpenAI API) -> FAISS index kur -> diske kaydet."""
    chunks = chunk_knowledge_base(docs_dir) if docs_dir else chunk_knowledge_base()
    index = build_faiss_index(chunks, model=model)
    save_index(index, chunks, out_dir)
    return index, chunks


def main() -> None:
    chunks = chunk_knowledge_base()
    print(f"{len(chunks)} chunk bulundu, OpenAI embedding API'si ile vektorlestiriliyor ({DEFAULT_EMBEDDING_MODEL})...")
    index = build_faiss_index(chunks)
    save_index(index, chunks)
    print(f"FAISS index kaydedildi: {DEFAULT_INDEX_DIR} ({index.ntotal} vektor)")


if __name__ == "__main__":
    main()
