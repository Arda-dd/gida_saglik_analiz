"""Embedding API sarmalayicisi (Faz 4 - RAG).

Kullanici karari (2026-07-08): embedding LLM gibi API uzerinden alinir, yerel bir model
KULLANILMAZ - diske yuzlerce MB'lik model dosyasi inmesin diye (disk alani kisitli oldugu
tespit edildi). Ilk tercih OpenAI embedding API'siydi; kullanici sonrasinda (2026-07-09) tamamen
ucretsiz, kart gerektirmeyen bir secenek istedi. Bu yuzden HuggingFace Inference API (ucretsiz
token) varsayilan saglayici yapildi - ayni cok-dilli model
(sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) yerel indirilmek yerine HF'nin
sunucusunda calisir, hicbir dosya diske inmez. OpenAI secenegi de (odemeli) alternatif olarak
korunur - bkz. src/rag/llm_provider.py'deki ayni saglayici-secimi felsefesi.
"""

from __future__ import annotations

import os

import numpy as np
from dotenv import load_dotenv

load_dotenv()

DEFAULT_PROVIDER = "huggingface"
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_HF_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize eder (cosine benzerligi = ic carpim icin, FAISS IndexFlatIP ile uyumlu)."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def _embed_openai(texts: list[str], model: str, client=None) -> np.ndarray:
    if client is None:
        import openai

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY bulunamadi (.env dosyasini kontrol edin)")
        client = openai.OpenAI(api_key=api_key)

    response = client.embeddings.create(model=model, input=texts)
    vectors = np.array([item.embedding for item in response.data], dtype=np.float32)
    return _normalize(vectors)


def _embed_huggingface(texts: list[str], model: str, client=None) -> np.ndarray:
    if client is None:
        from huggingface_hub import InferenceClient

        api_key = os.environ.get("HUGGINGFACE_API_KEY")
        if not api_key:
            raise ValueError("HUGGINGFACE_API_KEY bulunamadi (.env dosyasini kontrol edin)")
        client = InferenceClient(token=api_key)

    vectors = []
    for text in texts:
        raw = np.array(client.feature_extraction(text, model=model), dtype=np.float32)
        # Bazi modeller token-bazinda (seq_len, dim) cikti doner - mean pooling ile
        # cumle-seviyesi tek vektore indirilir; (dim,) donen modellerde bu adim atlanir.
        if raw.ndim == 2:
            raw = raw.mean(axis=0)
        vectors.append(raw)
    return _normalize(np.array(vectors, dtype=np.float32))


_EMBEDDING_BACKENDS = {
    "openai": (_embed_openai, DEFAULT_OPENAI_EMBEDDING_MODEL),
    "huggingface": (_embed_huggingface, DEFAULT_HF_EMBEDDING_MODEL),
}


def get_embedding_settings(config: dict | None = None) -> tuple[str, str]:
    """config.yaml -> rag.embedding_provider/embedding_model degerlerini (provider, model) olarak doner."""
    if config is None:
        from src.common.config import get_config

        config = get_config()

    rag_cfg = config.get("rag", {})
    provider = rag_cfg.get("embedding_provider", DEFAULT_PROVIDER)
    if provider not in _EMBEDDING_BACKENDS:
        raise ValueError(f"Bilinmeyen embedding saglayici: {provider}")
    model = rag_cfg.get("embedding_model") or _EMBEDDING_BACKENDS[provider][1]
    return provider, model


def embed_texts(
    texts: list[str],
    provider: str = DEFAULT_PROVIDER,
    model: str | None = None,
    client=None,
) -> np.ndarray:
    """Metinleri secilen saglayicinin embedding API'siyle vektorlere cevirir (L2-normalize)."""
    if provider not in _EMBEDDING_BACKENDS:
        raise ValueError(f"Bilinmeyen embedding saglayici: {provider}")
    backend_fn, default_model = _EMBEDDING_BACKENDS[provider]
    return backend_fn(texts, model or default_model, client=client)
