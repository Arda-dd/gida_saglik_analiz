import pytest

from src.ocr.evaluate import (
    engine_unavailable_reason,
    estimate_cost_usd,
    failed_entry_result,
    summarize_engine,
)

ENTRY = {
    "product_id": "off_1",
    "category": "atistirmalik",
    "nutrition_image": "x.jpg",
    "ground_truth_nutriments": {"sugars_100g": 32, "fat_100g": 17, "salt_100g": None},
}


def _ok(correct, evaluated, seconds, cost=0.0):
    return {
        "n_fields_evaluated": evaluated, "n_fields_correct": correct, "seconds": seconds,
        "cost_usd": cost, "ocr_confidence": 100.0, "error": None,
    }


def test_failed_entry_counts_all_truth_fields_as_wrong_instead_of_dropping_them():
    result = failed_entry_result(ENTRY, "ocrspace", RuntimeError("kota doldu"))

    assert result["n_fields_evaluated"] == 2  # salt None oldugu icin sayilmaz
    assert result["n_fields_correct"] == 0
    assert result["field_results"] == {"sugar_g": False, "fat_g": False}
    assert "kota doldu" in result["error"]


def test_summarize_engine_reports_strict_and_successful_only_accuracy():
    failed = failed_entry_result(ENTRY, "ocrspace", RuntimeError("429"))
    summary = summarize_engine([_ok(2, 2, 1.0), _ok(1, 2, 3.0), failed])

    assert summary["field_accuracy_pct"] == pytest.approx(3 / 6 * 100)
    assert summary["field_accuracy_on_successful_images_pct"] == pytest.approx(3 / 4 * 100)
    assert summary["n_failed_images"] == 1
    assert summary["median_seconds_per_image"] == pytest.approx(2.0)


def test_summarize_engine_sums_cost():
    summary = summarize_engine([_ok(1, 1, 1.0, cost=0.001), _ok(1, 1, 1.0, cost=0.002)])
    assert summary["total_cost_usd"] == pytest.approx(0.003)


def test_estimate_cost_usd_uses_token_prices():
    usage = {"model": "gpt-4.1-mini", "input_tokens": 1_000_000, "output_tokens": 1_000_000}
    assert estimate_cost_usd(usage) == pytest.approx(0.40 + 1.60)


def test_estimate_cost_usd_is_zero_for_free_engines_or_unknown_model():
    assert estimate_cost_usd(None) == 0.0
    assert estimate_cost_usd({"model": "bilinmeyen", "input_tokens": 10, "output_tokens": 10}) == 0.0


def test_engine_unavailable_reason_requires_api_keys(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OCR_SPACE_API_KEY", raising=False)

    assert "OPENAI_API_KEY" in engine_unavailable_reason("openai_vision")
    assert "OCR_SPACE_API_KEY" in engine_unavailable_reason("ocrspace")
    assert engine_unavailable_reason("easyocr") is None
