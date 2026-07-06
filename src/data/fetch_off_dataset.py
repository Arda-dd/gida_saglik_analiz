"""Faz 1 icin gercek Open Food Facts verisini ceken orkestrasyon script'i.

5 kategori icin temsili OFF categories_tags kullanarak urun cekilir, gorselleri
data/raw/openfoodfacts/images/'a indirilir, ham JSON'lar kategori bazinda kaydedilir,
validation.py ile dogrulanip dataset_builder.py ile data/processed/dataset.csv uretilir.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from src.common.schema import ProductRecord
from src.data.dataset_builder import build_dataset, save_dataset
from src.data.off_client import (
    download_product_image,
    fetch_products_by_category,
    off_record_to_product_record,
)
from src.data.validation import filter_valid_records

# Her kategori icin temsili OFF categories_tags degeri (en genis/yaygin olani secildi)
CATEGORY_SEARCH_TAGS = {
    "konserve": "en:canned-vegetables",
    "sut_urunu": "en:dairies",
    "icecek": "en:beverages",
    "atistirmalik": "en:snacks",
    "hazir_gida": "en:meals",
}

RAW_DIR = Path("data/raw/openfoodfacts")
IMAGES_DIR = RAW_DIR / "images"
DATASET_OUT = Path("data/processed/dataset.csv")

TARGET_PER_CATEGORY = 90


def main() -> None:
    all_records: list[ProductRecord] = []
    summary: dict[str, dict] = {}

    for category_name, tag in CATEGORY_SEARCH_TAGS.items():
        cache_path = RAW_DIR / f"{category_name}.json"
        if cache_path.exists():
            print(f"[{category_name}] onbellekten okunuyor ({cache_path})...")
            with cache_path.open("r", encoding="utf-8") as f:
                raw_products = json.load(f)
        else:
            print(f"[{category_name}] '{tag}' icin OFF'tan urunler cekiliyor...")
            raw_products = fetch_products_by_category(
                tag, page_size=TARGET_PER_CATEGORY, max_pages=1, sleep_between_pages=0
            )
            print(f"  {len(raw_products)} ham urun alindi")

            RAW_DIR.mkdir(parents=True, exist_ok=True)
            with cache_path.open("w", encoding="utf-8") as f:
                json.dump(raw_products, f, ensure_ascii=False, indent=2)

            time.sleep(5)  # kategoriler arasi OFF sunucusuna karsi kibarlik

        category_records = []
        images_downloaded = 0
        for raw in raw_products:
            record = off_record_to_product_record(raw)
            if record is None:
                continue
            category_records.append(record)

            image_url = raw.get("image_front_url")
            if image_url:
                dest = IMAGES_DIR / f"{record.product_id}.jpg"
                if dest.exists():
                    images_downloaded += 1
                elif download_product_image(image_url, dest):
                    images_downloaded += 1

        all_records.extend(category_records)
        summary[category_name] = {
            "raw_fetched": len(raw_products),
            "converted": len(category_records),
            "images_downloaded": images_downloaded,
        }
        print(f"  {len(category_records)} kayida cevrildi, {images_downloaded} gorsel indirildi")

    valid_records, issues = filter_valid_records(all_records)
    print(f"\nToplam ham kayit: {len(all_records)}")
    print(f"Gecerli (dogrulanmis) kayit: {len(valid_records)}")
    print(f"Reddedilen/flaglenen sorun sayisi: {len(issues)}")

    rule_counts: dict[str, int] = {}
    for issue in issues:
        rule_counts[issue.rule] = rule_counts.get(issue.rule, 0) + 1
    print(f"Sorun turleri: {rule_counts}")

    import pandas as pd

    empty_local_manifest = pd.DataFrame(
        columns=["product_id", "category", "image_path", "width", "height", "blur_score", "is_valid", "reasons"]
    )
    dataset_df = build_dataset(valid_records, empty_local_manifest)
    save_dataset(dataset_df, DATASET_OUT)
    print(f"\ndataset.csv kaydedildi: {DATASET_OUT} ({len(dataset_df)} satir)")

    print("\nKategori dagilimi:")
    print(dataset_df["category"].value_counts())

    print("\nOzet:", json.dumps(summary, ensure_ascii=False, indent=2))

    rejection_rate_pct = (len(all_records) - len(valid_records)) / len(all_records) * 100
    print(f"\nRet orani: %{rejection_rate_pct:.1f} (hedef: <%10)")

    issues_df = pd.DataFrame(issues, columns=["product_id", "rule", "message"])
    quality_report_path = Path("data/processed/data_quality_report.csv")
    quality_report_path.parent.mkdir(parents=True, exist_ok=True)
    issues_df.to_csv(quality_report_path, index=False)
    print(f"data_quality_report.csv kaydedildi: {quality_report_path} ({len(issues_df)} sorun kaydi)")


if __name__ == "__main__":
    main()
