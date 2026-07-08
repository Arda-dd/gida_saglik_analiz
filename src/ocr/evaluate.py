"""Gercek besin tablosu gorselleri uzerinde OCR + normalize pipeline'inin degerlendirmesi.

Oneri formu 2.6: OCR bileseninin performansi "cikarilan besin iceriklerinin etiket
gercegi (ground truth) ile uyumu uzerinden analiz edilecektir." Bu script, Faz 1'de
toplanan OFF ground-truth nutriments verisiyle (data/raw/openfoodfacts_ocr_samples/manifest.json)
gercek OCR ciktilarini karsilastirir.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.ocr.extract import extract_text_easyocr, extract_text_tesseract
from src.ocr.normalize import extract_and_normalize

MANIFEST_PATH = Path("data/raw/openfoodfacts_ocr_samples/manifest.json")
REPORT_PATH = Path("docs/ocr_evaluation_report.json")

# OFF ground-truth alan adi -> bizim schema alan adi (off_client.py ile ayni esleme mantigi)
GT_FIELD_MAP = {
    "energy-kcal_100g": "energy_kcal",
    "energy-kj_100g": "energy_kj",
    "fat_100g": "fat_g",
    "saturated-fat_100g": "saturated_fat_g",
    "carbohydrates_100g": "carbohydrate_g",
    "sugars_100g": "sugar_g",
    "fiber_100g": "fiber_g",
    "proteins_100g": "protein_g",
    "salt_100g": "salt_g",
}


def _values_match(extracted: float | None, truth: float, rel_tol: float = 0.15, abs_tol: float = 1.0) -> bool:
    if extracted is None:
        return False
    tolerance = max(abs_tol, abs(truth) * rel_tol)
    return abs(extracted - truth) <= tolerance


def evaluate_entry(entry: dict, engine: str) -> dict:
    image_path = entry["nutrition_image"]

    if engine == "tesseract":
        ocr_result = extract_text_tesseract(Path(image_path))
    else:
        ocr_result = extract_text_easyocr(Path(image_path))

    extracted_facts, _basis = extract_and_normalize(ocr_result.text)

    truth = entry["ground_truth_nutriments"]
    field_results = {}
    for gt_key, schema_field in GT_FIELD_MAP.items():
        truth_value = truth.get(gt_key)
        if truth_value is None:
            continue
        extracted_value = getattr(extracted_facts, schema_field)
        field_results[schema_field] = _values_match(extracted_value, float(truth_value))

    return {
        "product_id": entry["product_id"],
        "category": entry["category"],
        "engine": engine,
        "ocr_confidence": ocr_result.mean_confidence,
        "field_results": field_results,
        "n_fields_evaluated": len(field_results),
        "n_fields_correct": sum(field_results.values()),
    }


def main() -> None:
    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    print(f"{len(manifest)} gercek besin tablosu gorseli degerlendiriliyor (2 motor)...\n")

    all_results = []
    for engine in ["tesseract", "easyocr"]:
        print(f"=== {engine} ===")
        total_fields = 0
        correct_fields = 0
        confidences = []

        for entry in manifest:
            result = evaluate_entry(entry, engine)
            all_results.append(result)
            total_fields += result["n_fields_evaluated"]
            correct_fields += result["n_fields_correct"]
            confidences.append(result["ocr_confidence"])
            print(
                f"  {result['product_id']}: {result['n_fields_correct']}/{result['n_fields_evaluated']} "
                f"alan dogru, guven={result['ocr_confidence']:.1f}"
            )

        field_accuracy = correct_fields / total_fields if total_fields else 0.0
        mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        print(
            f"  --> {engine} TOPLAM: {correct_fields}/{total_fields} alan dogru "
            f"(%{field_accuracy * 100:.1f}), ortalama guven={mean_confidence:.1f}\n"
        )

    report = {"per_image_results": all_results}

    # Motor bazinda ozet
    summary = {}
    for engine in ["tesseract", "easyocr"]:
        engine_results = [r for r in all_results if r["engine"] == engine]
        total = sum(r["n_fields_evaluated"] for r in engine_results)
        correct = sum(r["n_fields_correct"] for r in engine_results)
        summary[engine] = {
            "total_fields_evaluated": total,
            "total_fields_correct": correct,
            "field_accuracy_pct": (correct / total * 100) if total else 0.0,
            "mean_ocr_confidence": sum(r["ocr_confidence"] for r in engine_results) / len(engine_results)
            if engine_results
            else 0.0,
        }
    report["summary"] = summary

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Rapor kaydedildi: {REPORT_PATH}")


if __name__ == "__main__":
    main()
