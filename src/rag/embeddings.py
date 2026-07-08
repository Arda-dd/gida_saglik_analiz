"""OpenAI embedding API sarmalayicisi (Faz 4 - RAG).

Kullanici karari (2026-07-08): embedding de LLM gibi API uzerinden alinir, yerel bir
sentence-transformers modeli KULLANILMAZ - diske yuzlerce MB'lik model dosyasi inmesin diye
(disk alani kisitli oldugu tespit edildi). Bu, projenin zaten benimsedigi "API tabanli, yerel
kaynagi zorlama" ilkesiyle tutarlidir (bkz. src/rag/llm_provider.py, ayni karar LLM icin de
gecerliydi).
"""

from __future__ import annotations

import os

import numpy as np
from dotenv import load_dotenv

load_dotenv()

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


def get_embedding_client():
    import openai

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY bulunamadi (.env dosyasini kontrol edin) - embedding icin gerekli"
        )
    return openai.OpenAI(api_key=api_key)


def embed_texts(
    texts: list[str], model: str = DEFAULT_EMBEDDING_MODEL, client=None
) -> np.ndarray:
    """Metinleri OpenAI embedding API'siyle vektorlere cevirir, L2-normalize eder.

    Normalize edilmis vektorlerde ic carpim (dot product) = cosine benzerligi olur,
    bu yuzden FAISS IndexFlatIP ile dogrudan kullanilabilir.
    """
    if client is None:
        client = get_embedding_client()

    response = client.embeddings.create(model=model, input=texts)
    vectors = np.array([item.embedding for item in response.data], dtype=np.float32)

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms
