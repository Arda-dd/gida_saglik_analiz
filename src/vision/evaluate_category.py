"""Kategori tahmini: CNN, OpenAI gorsel modeli ve birlesimin ayni test bolmesinde kiyasi.

CNN'in egitimde kullanilan stratified test bolmesi (src/vision/dataset.py,
stratified_train_test_split, random_state=42) yeniden uretilir - boylece CNN'in egitimde
gormedigi 60 gorsel uzerinde karsilastirilir.

Kullanim: python -m src.vision.evaluate_category   (60 OpenAI cagrisi, ~$0.01)
"""

from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv

from src.vision.dataset import load_vision_dataframe, stratified_train_test_split
from src.vision.infer import load_best_model_from_report, predict_category
from src.vision.vlm_classify import classify_with_openai, combine_category_predictions

DATASET_CSV = Path("data/processed/dataset.csv")
REPORT_PATH = Path("docs/category_ensemble_report.json")


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    agree = [r for r in rows if r["cnn"] == r["vlm"]]
    disagree = [r for r in rows if r["cnn"] != r["vlm"]]
    return {
        "n_test": n,
        "cnn_accuracy": sum(r["cnn"] == r["true"] for r in rows) / n,
        "vlm_accuracy": sum(r["vlm"] == r["true"] for r in rows) / n,
        "combined_accuracy": sum(r["combined"] == r["true"] for r in rows) / n,
        "n_agree": len(agree),
        "accuracy_when_agree": sum(r["cnn"] == r["true"] for r in agree) / len(agree) if agree else None,
        "n_disagree": len(disagree),
        "disagree_cnn_correct": sum(r["cnn"] == r["true"] for r in disagree),
        "disagree_vlm_correct": sum(r["vlm"] == r["true"] for r in disagree),
    }


def main() -> None:
    load_dotenv()
    _train, test = stratified_train_test_split(load_vision_dataframe(DATASET_CSV))
    model, categories = load_best_model_from_report()
    rows = []
    for _, item in test.iterrows():
        image_path = Path(item["image_path"])
        cnn_category, cnn_confidence = predict_category(model, image_path, categories)
        try:
            vlm_category = classify_with_openai(image_path)
        except Exception as exc:
            print(f"  {item['product_id']}: OpenAI hatasi {exc}")
            vlm_category = None
        combined, _confidence, source = combine_category_predictions(cnn_category, cnn_confidence, vlm_category)
        rows.append(
            {
                "product_id": item["product_id"], "true": item["category"], "cnn": cnn_category,
                "cnn_confidence": float(cnn_confidence), "vlm": vlm_category,
                "combined": combined, "source": source,
            }
        )
    summary = summarize(rows)
    print(json.dumps(summary, indent=1))
    REPORT_PATH.write_text(json.dumps({"summary": summary, "predictions": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Rapor kaydedildi: {REPORT_PATH}")


if __name__ == "__main__":
    main()
