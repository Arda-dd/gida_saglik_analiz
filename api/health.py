"""/health uc noktasinin kontrolleri.

Neden var: 2026-10-14'te HuggingFace'in ucretsiz kredisi bitince (402) canli RAG aciklamasi
haftalarca SESSIZCE devre disi kaldi - pipeline opsiyonel katman hatasini bilerek yutuyordu ve
eski /health her zaman {"status": "ok"} donuyordu. Bu modul sistemin calismak icin ihtiyac
duydugu parcalari tek tek kontrol eder:

- hizli kontroller (varsayilan, ag cagrisi yok): API anahtarlari, OCR motor sirasi, RAG
  indeksinin varligi ve vektor boyutunun yapilandirilan embedding modeliyle uyumu, CNN modeli;
- derin kontroller (?deep=true): embedding ve LLM API'lerine gercek, kucuk birer cagri -
  kredi bitmesi / gecersiz anahtar / 402 gibi hatalari ancak bunlar yakalar (maliyeti ihmal
  edilebilir, ama her istekte degil, izleme amacli calistirilmali).

Herhangi bir kontrol basarisizsa status "degraded" olur; uygulama yine de calisir (opsiyonel
katmanlar devre disi kalir), bu yuzden HTTP 200 doner.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from src.common.config import get_config

# Anahtar kontrolu, .env'i yukleyen modullerden (ocr.extract, rag.embeddings) once calisabilir.
load_dotenv()

# Bilinen embedding modellerinin vektor boyutu - indeks bu boyutta degilse baska bir modelle
# kurulmustur ve sorgular calismaz (python -m src.rag.index_builder ile yeniden kurulmali).
KNOWN_EMBEDDING_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 384,
}
REQUIRED_KEYS_BY_PROVIDER = {
    "openai": "OPENAI_API_KEY",
    "huggingface": "HUGGINGFACE_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


def _ok(**details) -> dict:
    return {"ok": True, **details}


def _fail(error: str, **details) -> dict:
    return {"ok": False, "error": error, **details}


def check_api_keys(config: dict) -> dict:
    needed = {
        "llm": REQUIRED_KEYS_BY_PROVIDER.get(config["llm"]["provider"]),
        "embedding": REQUIRED_KEYS_BY_PROVIDER.get(config["rag"]["embedding_provider"]),
    }
    missing = sorted({key for key in needed.values() if key and not os.environ.get(key)})
    if missing:
        return _fail(f"eksik anahtar: {', '.join(missing)}", required=needed)
    return _ok(required=needed)


def check_ocr_engines(config: dict) -> dict:
    from src.ocr.extract import REQUIRED_API_KEYS

    engines = config.get("ocr", {}).get("live_engines", [])
    usable = [e for e in engines if not REQUIRED_API_KEYS.get(e) or os.environ.get(REQUIRED_API_KEYS[e])]
    if not usable:
        return _fail("kullanilabilir OCR motoru yok", configured=engines)
    details = {"configured": engines, "usable": usable}
    if engines and usable[0] != engines[0]:
        return _fail(f"birincil motor ({engines[0]}) anahtarsiz, {usable[0]}'a dusulecek", **details)
    return _ok(**details)


def check_rag_index(config: dict) -> dict:
    from src.rag.index_builder import DEFAULT_INDEX_DIR, load_index

    if not (DEFAULT_INDEX_DIR / "index.faiss").exists():
        return _fail("RAG indeksi yok (python -m src.rag.index_builder)")
    index, chunks = load_index()
    model = config["rag"].get("embedding_model")
    expected = KNOWN_EMBEDDING_DIMS.get(model)
    details = {"vectors": index.ntotal, "dim": index.d, "embedding_model": model}
    if expected is not None and index.d != expected:
        return _fail(
            f"indeks {index.d} boyutlu ama {model} {expected} boyutlu - indeks yeniden kurulmali "
            "(python -m src.rag.index_builder)",
            **details,
        )
    return _ok(**details)


def check_vision_model() -> dict:
    from src.vision.infer import DEFAULT_CHECKPOINT

    if not DEFAULT_CHECKPOINT.exists():
        return _fail(f"CNN agirlik dosyasi yok: {DEFAULT_CHECKPOINT.name}")
    return _ok(checkpoint=DEFAULT_CHECKPOINT.name)


def check_embedding_api(config: dict) -> dict:
    from src.rag.embeddings import embed_texts, get_embedding_settings

    provider, model = get_embedding_settings(config)
    try:
        vector = embed_texts(["saglik kontrolu"], provider=provider, model=model)
    except Exception as exc:
        return _fail(f"{type(exc).__name__}: {str(exc)[:200]}", provider=provider)
    return _ok(provider=provider, dim=int(vector.shape[1]))


def check_llm_api(config: dict) -> dict:
    from src.rag.llm_provider import get_llm_provider

    try:
        reply = get_llm_provider(config).generate("Sadece 'tamam' yaz.")
    except Exception as exc:
        return _fail(f"{type(exc).__name__}: {str(exc)[:200]}", provider=config["llm"]["provider"])
    return _ok(provider=config["llm"]["provider"], reply=reply.strip()[:20])


def run_health_checks(deep: bool = False) -> dict:
    config = get_config()
    checks = {
        "api_keys": check_api_keys(config),
        "ocr_engines": check_ocr_engines(config),
        "rag_index": check_rag_index(config),
        "vision_model": check_vision_model(),
    }
    if deep:
        checks["embedding_api"] = check_embedding_api(config)
        checks["llm_api"] = check_llm_api(config)
    status = "ok" if all(c["ok"] for c in checks.values()) else "degraded"
    return {"status": status, "checks": checks}
