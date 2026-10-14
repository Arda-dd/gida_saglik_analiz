"""Gercek besin tablosu gorselleri uzerinde OCR + normalize pipeline'inin degerlendirmesi.

Oneri formu 2.6: OCR bileseninin performansi "cikarilan besin iceriklerinin etiket
gercegi (ground truth) ile uyumu uzerinden analiz edilecektir." Bu script, Faz 1'de
toplanan OFF ground-truth nutriments verisiyle (data/raw/openfoodfacts_ocr_samples/manifest.json)
gercek OCR ciktilarini karsilastirir.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import time
from pathlib import Path
from typing import Callable

from src.ocr.extract import (
    OCRResult,
    extract_text_easyocr,
    extract_text_ocrspace,
    extract_text_openai_vision,
    extract_text_tesseract,
    extract_text_windows,
)
from src.ocr.normalize import extract_and_normalize

MANIFEST_PATH = Path("data/raw/openfoodfacts_ocr_samples/manifest.json")
REPORT_PATH = Path("docs/ocr_evaluation_report.json")

EXTRACTORS: dict[str, Callable[[Path], OCRResult]] = {
    "tesseract": extract_text_tesseract,
    "easyocr": extract_text_easyocr,
    "windows": extract_text_windows,
    "ocrspace": extract_text_ocrspace,
    "openai_vision": extract_text_openai_vision,
}
ENGINES = list(EXTRACTORS)

# Ucretli API'lerin 1M token basina USD fiyati (maliyet tahmini icin, 2026-10 liste fiyati).
TOKEN_PRICES_USD_PER_1M = {"gpt-4.1-mini": {"input": 0.40, "output": 1.60}}


def engine_unavailable_reason(engine: str) -> str | None:
    """Motor bu ortamda calistirilamiyorsa nedenini, calistirilabiliyorsa None doner.

    ocrspace/openai_vision agdan gecen ve anahtar gerektiren API'lerdir; windows sadece
    Windows'ta vardir - eksik olan motor degerlendirmeden atlanir, tum kosu durmaz.
    """
    if engine == "ocrspace" and not os.environ.get("OCR_SPACE_API_KEY"):
        return "OCR_SPACE_API_KEY ayarlanmamis"
    if engine == "openai_vision" and not os.environ.get("OPENAI_API_KEY"):
        return "OPENAI_API_KEY ayarlanmamis"
    if engine == "windows" and os.name != "nt":
        return "Windows OCR sadece Windows'ta calisir"
    return None


def estimate_cost_usd(usage: dict | None) -> float:
    if not usage or usage.get("model") not in TOKEN_PRICES_USD_PER_1M:
        return 0.0
    prices = TOKEN_PRICES_USD_PER_1M[usage["model"]]
    return (usage["input_tokens"] * prices["input"] + usage["output_tokens"] * prices["output"]) / 1_000_000

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


NUMBER_IN_TEXT = re.compile(r"\d+(?:[.,]\d+)?")


def number_recall_fields(ocr_text: str, truth_fields: dict[str, float]) -> dict[str, bool]:
    """Ground-truth degeri OCR'in HAM metninde sayi olarak geciyor mu (regex ayristirmasindan
    bagimsiz, saf "motor bu sayiyi okuyabildi mi" olcusu).

    Alan dogrulugu (field_results) OCR + normalize regex'ini BIRLIKTE olcer: bir motor sayiyi
    kusursuz okusa bile normalize o etiket formatini tanimiyorsa (ornegin birimin sayidan once
    geldigi "Proteines g 8,0") alan yanlis sayilir. Bu metrik iki katmani ayirir.
    Degeri 0 olan alanlar haric tutulur: "0" metinde tesadufen cok sik gecer ve bu alanlar
    etiketlerde genelde hic yazilmaz. Kucuk sayilar (1, 2, 3) baska bir satirda tesadufen
    gecebildigi icin bu metrik okuma basarisinin hafif iyimser bir ust sinirdir.
    """
    numbers = {float(m.replace(",", ".")) for m in NUMBER_IN_TEXT.findall(ocr_text or "")}
    return {
        field: any(abs(number - truth_value) < 0.051 for number in numbers)
        for field, truth_value in truth_fields.items()
        if truth_value != 0
    }


def _truth_fields(entry: dict) -> dict[str, float]:
    truth = entry["ground_truth_nutriments"]
    return {
        schema_field: float(truth[gt_key])
        for gt_key, schema_field in GT_FIELD_MAP.items()
        if truth.get(gt_key) is not None
    }


def evaluate_entry(entry: dict, engine: str) -> dict:
    image_path = Path(entry["nutrition_image"])

    started = time.perf_counter()
    ocr_result = EXTRACTORS[engine](image_path)
    elapsed = time.perf_counter() - started

    extracted_facts, _basis = extract_and_normalize(ocr_result.text)

    truth_fields = _truth_fields(entry)
    field_results = {
        field: _values_match(getattr(extracted_facts, field), truth_value)
        for field, truth_value in truth_fields.items()
    }

    return {
        "product_id": entry["product_id"],
        "category": entry["category"],
        "engine": engine,
        "ocr_confidence": ocr_result.mean_confidence,
        "field_results": field_results,
        "n_fields_evaluated": len(field_results),
        "n_fields_correct": sum(field_results.values()),
        "number_recall": number_recall_fields(ocr_result.text, truth_fields),
        "seconds": elapsed,
        "cost_usd": estimate_cost_usd(ocr_result.usage),
        "error": None,
        "ocr_text": ocr_result.text,
    }


def failed_entry_result(entry: dict, engine: str, error: Exception) -> dict:
    """Hata alan gorselin alanlari YANLIS sayilir (paydadan dusmez).

    Eski surumde hata alan gorsel 0/0 kaydediliyordu; bu, kotaya takilan bir motorun
    dogrulugunu sadece basarili gorseller uzerinden hesaplatip sisiriyordu (OCR.space'in
    %50.6'si 265 degil 176 alan uzerindendi). Basarisiz gorsel sayisi ayrica raporlanir.
    """
    truth_fields = _truth_fields(entry)
    return {
        "product_id": entry["product_id"],
        "category": entry["category"],
        "engine": engine,
        "ocr_confidence": 0.0,
        "field_results": {field: False for field in truth_fields},
        "n_fields_evaluated": len(truth_fields),
        "n_fields_correct": 0,
        "number_recall": number_recall_fields("", truth_fields),
        "seconds": None,
        "cost_usd": 0.0,
        "error": str(error)[:300],
        "ocr_text": None,
    }


def summarize_engine(engine_results: list[dict]) -> dict:
    total = sum(r["n_fields_evaluated"] for r in engine_results)
    correct = sum(r["n_fields_correct"] for r in engine_results)
    ok_results = [r for r in engine_results if r["error"] is None]
    ok_total = sum(r["n_fields_evaluated"] for r in ok_results)
    ok_correct = sum(r["n_fields_correct"] for r in ok_results)
    timings = [r["seconds"] for r in ok_results if r["seconds"] is not None]
    recall_values = [found for r in engine_results for found in r.get("number_recall", {}).values()]
    return {
        "total_fields_evaluated": total,
        "total_fields_correct": correct,
        "field_accuracy_pct": (correct / total * 100) if total else 0.0,
        "number_recall_pct": (sum(recall_values) / len(recall_values) * 100) if recall_values else 0.0,
        "number_recall_fields": len(recall_values),
        "n_images": len(engine_results),
        "n_failed_images": len(engine_results) - len(ok_results),
        "field_accuracy_on_successful_images_pct": (ok_correct / ok_total * 100) if ok_total else 0.0,
        "median_seconds_per_image": statistics.median(timings) if timings else None,
        "mean_seconds_per_image": statistics.mean(timings) if timings else None,
        "total_cost_usd": sum(r["cost_usd"] for r in engine_results),
        "mean_ocr_confidence": sum(r["ocr_confidence"] for r in engine_results) / len(engine_results)
        if engine_results
        else 0.0,
    }


def run_engine(engine: str, manifest: list[dict]) -> list[dict]:
    print(f"=== {engine} ===")
    results = []
    for entry in manifest:
        if engine == "ocrspace":
            # Ucretsiz katmanin istek hizi siniri var (gercek calistirmada 429 Too Many
            # Requests goruldu, 2026-08-15) - istekler arasi kisa bir bekleme ile buna uyulur.
            time.sleep(3)
        try:
            try:
                result = evaluate_entry(entry, engine)
            except Exception:
                if engine != "ocrspace":
                    raise
                # Gecici kota/ag hatasina karsi bir kez, beklemeli tekrar dene.
                time.sleep(20)
                result = evaluate_entry(entry, engine)
        except Exception as exc:
            # Tek bir gorseldeki hata tum degerlendirmeyi durdurmamali - alanlari yanlis sayilir.
            print(f"  {entry['product_id']}: HATA ({exc})")
            result = failed_entry_result(entry, engine, exc)
        results.append(result)
        if result["error"] is None:
            print(
                f"  {result['product_id']}: {result['n_fields_correct']}/{result['n_fields_evaluated']} "
                f"alan dogru, {result['seconds']:.1f} sn"
            )

    summary = summarize_engine(results)
    print(
        f"  --> {engine} TOPLAM: {summary['total_fields_correct']}/{summary['total_fields_evaluated']} "
        f"alan dogru (%{summary['field_accuracy_pct']:.1f}), sayi yakalama %{summary['number_recall_pct']:.1f}, basarisiz gorsel: "
        f"{summary['n_failed_images']}, medyan {summary['median_seconds_per_image'] or 0:.2f} sn/gorsel, "
        f"maliyet ${summary['total_cost_usd']:.4f}\n"
    )
    return results


def merge_results(existing: list[dict], new: list[dict]) -> list[dict]:
    """Ayni (motor, urun) satirlarini yenileriyle degistirir, digerlerini korur.

    Bir motoru (ornegin prompt'u degisen openai_vision) ya da sadece kotaya takilan gorselleri
    yeniden olcerken diger motorlari bastan calistirmamak icin (OCR.space ucretsiz kotasi gun
    icinde ikinci tam kosuda 30 gorselin 10'unda 429 verdi).
    """
    new_by_key = {(r["engine"], r["product_id"]): r for r in new}
    merged = [new_by_key.pop((r["engine"], r["product_id"]), r) for r in existing]
    return merged + list(new_by_key.values())


def build_report(rows: list[dict]) -> dict:
    engines = [engine for engine in ENGINES if any(r["engine"] == engine for r in rows)]
    return {
        "per_image_results": rows,
        "summary": {engine: summarize_engine([r for r in rows if r["engine"] == engine]) for engine in engines},
    }


def reparse_rows(rows: list[dict], manifest: list[dict]) -> list[dict]:
    """Kayitli ham OCR metinlerini GUNCEL ayristiriciyla yeniden puanlar (OCR/API cagrisi yok).

    Ayristirici (normalize) degistiginde motorlari yeniden calistirmadan etkisini olcmek icin:
    sayi yakalama (number_recall) OCR'in kendisine bagli oldugundan degismez, sadece alan
    dogrulugu yeniden hesaplanir. Hata almis (ocr_text'i olmayan) satirlar oldugu gibi kalir.
    """
    entries = {e["product_id"]: e for e in manifest}
    reparsed = []
    for row in rows:
        if row["error"] is not None or row.get("ocr_text") is None:
            reparsed.append(row)
            continue
        facts, _basis = extract_and_normalize(row["ocr_text"])
        field_results = {
            field: _values_match(getattr(facts, field), truth_value)
            for field, truth_value in _truth_fields(entries[row["product_id"]]).items()
        }
        reparsed.append(
            {**row, "field_results": field_results, "n_fields_correct": sum(field_results.values())}
        )
    return reparsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Gercek besin tablosu gorsellerinde OCR motor kiyasi")
    parser.add_argument(
        "--engines", nargs="+", choices=ENGINES, default=ENGINES,
        help="degerlendirilecek motorlar (varsayilan: hepsi)",
    )
    parser.add_argument(
        "--merge", action="store_true",
        help="mevcut raporu silme, sadece secilen motorlarin satirlarini guncelle",
    )
    parser.add_argument(
        "--retry-failed", action="store_true",
        help="mevcut raporda HATA almis gorselleri (secilen motorlar icin) yeniden dene ve birlestir",
    )
    parser.add_argument(
        "--reparse", action="store_true",
        help="OCR calistirmadan, kayitli ham metinleri guncel ayristiriciyla yeniden puanla",
    )
    args = parser.parse_args()

    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    if args.reparse:
        with REPORT_PATH.open("r", encoding="utf-8") as f:
            old_report = json.load(f)
        report = build_report(reparse_rows(old_report["per_image_results"], manifest))
        for engine, summary in report["summary"].items():
            before = old_report["summary"][engine]["field_accuracy_pct"]
            print(
                f"[yeniden ayristirma] {engine}: alan dogrulugu %{before:.1f} -> "
                f"%{summary['field_accuracy_pct']:.1f} ({summary['total_fields_correct']}/"
                f"{summary['total_fields_evaluated']})"
            )
        with REPORT_PATH.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"Rapor kaydedildi: {REPORT_PATH}")
        return

    existing_rows: list[dict] = []
    if args.merge or args.retry_failed:
        with REPORT_PATH.open("r", encoding="utf-8") as f:
            existing_rows = json.load(f)["per_image_results"]

    engines = []
    for engine in args.engines:
        reason = engine_unavailable_reason(engine)
        if reason:
            print(f"'{engine}' atlaniyor: {reason}")
        else:
            engines.append(engine)

    new_rows = []
    for engine in engines:
        entries = manifest
        if args.retry_failed:
            failed_ids = {r["product_id"] for r in existing_rows if r["engine"] == engine and r["error"]}
            entries = [e for e in manifest if e["product_id"] in failed_ids]
            print(f"{engine}: {len(entries)} hatali gorsel yeniden deneniyor")
            if not entries:
                continue
        new_rows.extend(run_engine(engine, entries))

    report = build_report(merge_results(existing_rows, new_rows))
    for engine, summary in report["summary"].items():
        print(
            f"[rapor] {engine}: alan dogrulugu %{summary['field_accuracy_pct']:.1f}, sayi yakalama "
            f"%{summary['number_recall_pct']:.1f}, basarisiz gorsel {summary['n_failed_images']}"
        )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Rapor kaydedildi: {REPORT_PATH}")


if __name__ == "__main__":
    main()
