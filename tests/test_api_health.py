from types import SimpleNamespace
from unittest.mock import patch

import api.health as health

CONFIG = {
    "llm": {"provider": "openai"},
    "rag": {"embedding_provider": "openai", "embedding_model": "text-embedding-3-small"},
    "ocr": {"live_engines": ["openai_vision", "ocrspace", "easyocr"]},
}


def test_api_keys_check_reports_missing_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = health.check_api_keys(CONFIG)
    assert result["ok"] is False
    assert "OPENAI_API_KEY" in result["error"]


def test_api_keys_check_passes_when_present(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert health.check_api_keys(CONFIG)["ok"] is True


def test_ocr_check_flags_when_primary_engine_has_no_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OCR_SPACE_API_KEY", "k")
    result = health.check_ocr_engines(CONFIG)
    assert result["ok"] is False
    assert result["usable"][0] == "ocrspace"


def test_rag_index_check_detects_dimension_mismatch(tmp_path):
    # HuggingFace (384) ile kurulmus indeks + OpenAI (1536) embedding = sorgular calismaz.
    fake_index = SimpleNamespace(ntotal=19, d=384)
    with patch("src.rag.index_builder.load_index", return_value=(fake_index, [])), patch(
        "src.rag.index_builder.DEFAULT_INDEX_DIR", tmp_path
    ):
        (tmp_path / "index.faiss").write_bytes(b"x")
        result = health.check_rag_index(CONFIG)
    assert result["ok"] is False
    assert "yeniden kurulmali" in result["error"]


def test_embedding_api_check_surfaces_credit_errors():
    with patch("src.rag.embeddings.embed_texts", side_effect=RuntimeError("402 Payment Required")):
        result = health.check_embedding_api(CONFIG)
    assert result["ok"] is False
    assert "402" in result["error"]


def test_run_health_checks_is_degraded_when_any_check_fails(monkeypatch):
    monkeypatch.setattr(health, "check_vision_model", lambda: {"ok": False, "error": "yok"})
    monkeypatch.setattr(health, "check_rag_index", lambda config: {"ok": True})
    result = health.run_health_checks()
    assert result["status"] == "degraded"
    assert "embedding_api" not in result["checks"]  # derin kontroller varsayilan olarak kapali
