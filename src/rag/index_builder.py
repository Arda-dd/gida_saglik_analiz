"""Bilgi tabani chunk'larindan FAISS yogun (dense) embedding index'i olusturur.

Oneri formu 2.4: "Retriever (FAISS/BM25)". Burada FAISS tarafi + kalici saklama (index +
chunk metadatasi) implemente edilir; BM25 tarafi src/rag/retriever.py icinde hibrit skor
icin ayrica calisir (bu modulden bagimsizdir, kendi rank_bm25 corpus'unu kurar).

Embedding API uzerinden alinir (bkz. src/rag/embeddings.py, varsayilan: HuggingFace Inference
API, ucretsiz) - yerel bir model diske inmez. Cosine benzerligi icin embedding vektorleri
L2-normalize edilip IndexFlatIP kullanilir (normalize edilmis vektorlerde ic carpim = cosine
benzerligi).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import faiss

from src.rag.chunking import Chunk, chunk_knowledge_base
from src.rag.embeddings import DEFAULT_PROVIDER, embed_texts, get_embedding_settings

DEFAULT_INDEX_DIR = Path(__file__).resolve().parents[2] / "data" / "knowledge_base" / "index"


def build_faiss_index(
    chunks: list[Chunk],
    provider: str = DEFAULT_PROVIDER,
    model: str | None = None,
    client=None,
) -> faiss.Index:
    """Chunk listesinden IndexFlatIP (cosine benzerligi) FAISS index'i olusturur."""
    if not chunks:
        raise ValueError("Bos chunk listesinden index olusturulamaz")

    embeddings = embed_texts(
        [c.embedding_text for c in chunks], provider=provider, model=model, client=client
    )
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index


def save_index(index: faiss.Index, chunks: list[Chunk], out_dir: Path = DEFAULT_INDEX_DIR) -> None:
    import shutil
    import uuid
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Windows unicode uyumlulugu icin index dosyasini gecici olarak ASCII bir dizinde (C:\\Users\\Public) yazip
    # ardindan python shutil kullanarak hedef dizine kopyaliyoruz.
    temp_name = f"temp_faiss_{uuid.uuid4().hex}.index"
    temp_path = Path("C:/Users/Public") / temp_name
    
    try:
        faiss.write_index(index, str(temp_path))
        shutil.copy(str(temp_path), str(out_dir / "index.faiss"))
    finally:
        if temp_path.exists():
            temp_path.unlink()

    chunks_data = [asdict(c) for c in chunks]
    (out_dir / "chunks.json").write_text(
        json.dumps(chunks_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_index(out_dir: Path = DEFAULT_INDEX_DIR) -> tuple[faiss.Index, list[Chunk]]:
    import shutil
    import uuid
    
    # Windows unicode uyumlulugu icin index dosyasini gecici olarak ASCII bir dizine kopyalayip yukluyoruz.
    temp_name = f"temp_faiss_{uuid.uuid4().hex}.index"
    temp_path = Path("C:/Users/Public") / temp_name
    
    shutil.copy(str(out_dir / "index.faiss"), str(temp_path))
    try:
        index = faiss.read_index(str(temp_path))
    finally:
        if temp_path.exists():
            temp_path.unlink()

    chunks_data = json.loads((out_dir / "chunks.json").read_text(encoding="utf-8"))
    chunks = [Chunk(**d) for d in chunks_data]
    return index, chunks


def build_and_save_index(
    docs_dir: Path | None = None,
    out_dir: Path = DEFAULT_INDEX_DIR,
    provider: str | None = None,
    model: str | None = None,
) -> tuple[faiss.Index, list[Chunk]]:
    """Uctan uca: dokumanlari chunk'la -> embed et (API) -> FAISS index kur -> diske kaydet."""
    if provider is None or model is None:
        cfg_provider, cfg_model = get_embedding_settings()
        provider = provider or cfg_provider
        model = model or cfg_model

    chunks = chunk_knowledge_base(docs_dir) if docs_dir else chunk_knowledge_base()
    index = build_faiss_index(chunks, provider=provider, model=model)
    save_index(index, chunks, out_dir)
    return index, chunks


def main() -> None:
    provider, model = get_embedding_settings()
    chunks = chunk_knowledge_base()
    print(f"{len(chunks)} chunk bulundu, {provider} embedding API'si ile vektorlestiriliyor ({model})...")
    index = build_faiss_index(chunks, provider=provider, model=model)
    save_index(index, chunks)
    print(f"FAISS index kaydedildi: {DEFAULT_INDEX_DIR} ({index.ntotal} vektor)")


if __name__ == "__main__":
    main()
