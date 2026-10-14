"""Alerjen tespitinin gercek icindekiler gorsellerinde degerlendirmesi.

Saglik acisindan en kritik hata YANLIS NEGATIF'tir (findik alerjisi olan birine "alerjen yok"
demek), bu yuzden birincil olcut "icerir" alerjenlerinde duyarlilik (recall).

Ground truth: data/eval/allergen_ground_truth.json - Open Food Facts'in allergens_tags
("icerir") ve traces_tags ("eser miktarda icerebilir") alanlari, 2026-10-14'te cekildi.
Onemli sinirlilik: OFF'ta icindekiler metni girilmemis urunlerde (ozellikle Fas sut urunleri)
alerjen etiketi de bostur - bu urunlerde "alerjen yok" bilgisi guvenilmez oldugundan yanlis
pozitifler SADECE icindekiler metni olan urunlerde sayilir.

Kullanim:
    python -m src.ocr.evaluate_allergens                     # OCR + puanlama (API cagrisi)
    python -m src.ocr.evaluate_allergens --reparse           # kayitli metinleri yeniden puanla
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.common.schema import Allergen
from src.data.off_client import OFF_ALLERGEN_MAP
from src.ocr.allergens import detect_allergens
from src.ocr.evaluate import MANIFEST_PATH
from src.ocr.extract import extract_text_easyocr, extract_text_openai_vision

GROUND_TRUTH_PATH = Path("data/eval/allergen_ground_truth.json")
REPORT_PATH = Path("docs/allergen_evaluation_report.json")
ENGINES = {"openai_vision": extract_text_openai_vision, "easyocr": extract_text_easyocr}


def map_tags(tags: list[str]) -> set[Allergen]:
    return {OFF_ALLERGEN_MAP[tag] for tag in tags if tag in OFF_ALLERGEN_MAP}


def score(rows: list[dict], ground_truth: dict) -> dict:
    """Motor bazinda: icerir-recall, eser-recall, yanlis pozitif sayisi (guvenilir urunlerde)."""
    summary: dict[str, dict] = {}
    for engine in sorted({r["engine"] for r in rows}):
        tp = fn = fp = trace_hit = trace_total = 0
        missed, false_pos = [], []
        for row in (r for r in rows if r["engine"] == engine):
            gt = ground_truth[row["product_id"]]
            contains = map_tags(gt["allergens_tags"])
            traces = map_tags(gt["traces_tags"]) - contains
            predicted = set(detect_allergens(row["ocr_text"] or ""))
            tp += len(predicted & contains)
            fn += len(contains - predicted)
            missed += [f"{row['product_id']}:{a.value}" for a in contains - predicted]
            trace_total += len(traces)
            trace_hit += len(predicted & traces)
            if gt["has_ingredients_text"]:
                extra = predicted - contains - traces
                fp += len(extra)
                false_pos += [f"{row['product_id']}:{a.value}" for a in extra]
        summary[engine] = {
            "contains_recall_pct": tp / (tp + fn) * 100 if tp + fn else 0.0,
            "contains_tp": tp,
            "contains_fn": fn,
            "traces_recall_pct": trace_hit / trace_total * 100 if trace_total else 0.0,
            "traces_found": trace_hit,
            "traces_total": trace_total,
            "false_positives_on_reliable_products": fp,
            "missed": missed,
            "false_positive_examples": false_pos,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Alerjen tespiti degerlendirmesi")
    parser.add_argument("--reparse", action="store_true", help="OCR calistirmadan yeniden puanla")
    args = parser.parse_args()

    ground_truth = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))["urunler"]
    if args.reparse:
        rows = json.loads(REPORT_PATH.read_text(encoding="utf-8"))["per_image_results"]
    else:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        rows = []
        for engine, extractor in ENGINES.items():
            for entry in manifest:
                try:
                    text = extractor(Path(entry["ingredients_image"])).text
                except Exception as exc:
                    print(f"  {engine} {entry['product_id']}: HATA {exc}")
                    text = None
                rows.append({"engine": engine, "product_id": entry["product_id"], "ocr_text": text})

    summary = score(rows, ground_truth)
    for engine, s in summary.items():
        print(
            f"{engine}: icerir recall %{s['contains_recall_pct']:.1f} ({s['contains_tp']}/"
            f"{s['contains_tp'] + s['contains_fn']}), eser %{s['traces_recall_pct']:.1f} "
            f"({s['traces_found']}/{s['traces_total']}), yanlis pozitif {s['false_positives_on_reliable_products']}"
        )
        print(f"   kacirilan: {s['missed']}")
        print(f"   yanlis pozitif: {s['false_positive_examples']}")
    REPORT_PATH.write_text(
        json.dumps({"summary": summary, "per_image_results": rows}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Rapor kaydedildi: {REPORT_PATH}")


if __name__ == "__main__":
    main()
