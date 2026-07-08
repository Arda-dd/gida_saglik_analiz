"""Faz 3 (OCR) gelistirme/dogrulama icin gercek besin tablosu ve icindekiler gorselleri ceker.

Faz 1/2'de indirilen gorseller `image_front_url` (urunun on yuzu) idi - kategori
siniflandirmasi icin dogruydu ama OCR ile besin degeri okumak icin YANLIS gorsel turu.

Onemli bulgu: OFF'un `/api/v2/search` endpoint'i 'image_nutrition_url'/'image_ingredients_url'
alanlarini DONDURMEZ (arama indeksinde yok); bu alanlar sadece tekil urun endpoint'inde
(`/api/v2/product/{code}.json`) mevcuttur. Bu yuzden bu script, Faz 1'de zaten cekilmis olan
394 urunun kodlarini kullanarak tek tek urun sorgusu yapar.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from src.data.off_client import download_product_image, fetch_single_product

DATASET_CSV = Path("data/processed/dataset.csv")
OUT_DIR = Path("data/raw/openfoodfacts_ocr_samples")
NUTRITION_IMAGES_DIR = OUT_DIR / "nutrition_images"
INGREDIENTS_IMAGES_DIR = OUT_DIR / "ingredients_images"
MANIFEST_PATH = OUT_DIR / "manifest.json"

FIELDS = "code,image_nutrition_url,image_ingredients_url,nutriments"
TARGET_PER_CATEGORY = 6
MAX_LOOKUPS_PER_CATEGORY = 40  # havuzun tamamini taramadan makul bir sinir


def main() -> None:
    df = pd.read_csv(DATASET_CSV)
    off_df = df[df["source"] == "open_food_facts"].copy()
    off_df["code"] = off_df["product_id"].str.replace("off_", "", regex=False)

    manifest = []
    NUTRITION_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    INGREDIENTS_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    for category, group in off_df.groupby("category"):
        print(f"[{category}] taraniyor (havuz: {len(group)})...")
        found = 0
        checked = 0

        for _, row in group.iterrows():
            if found >= TARGET_PER_CATEGORY or checked >= MAX_LOOKUPS_PER_CATEGORY:
                break
            checked += 1

            product = fetch_single_product(row["code"], fields=FIELDS)
            time.sleep(0.5)  # OFF API'sine karsi kibarlik
            if product is None:
                continue

            nutrition_url = product.get("image_nutrition_url")
            if not nutrition_url:
                continue

            dest = NUTRITION_IMAGES_DIR / f"{category}_{row['code']}.jpg"
            if not dest.exists():
                if download_product_image(nutrition_url, dest) is None:
                    continue

            ingredients_path = None
            ingredients_url = product.get("image_ingredients_url")
            if ingredients_url:
                ing_dest = INGREDIENTS_IMAGES_DIR / f"{category}_{row['code']}.jpg"
                if ing_dest.exists() or download_product_image(ingredients_url, ing_dest):
                    ingredients_path = str(ing_dest)

            manifest.append(
                {
                    "product_id": row["product_id"],
                    "category": category,
                    "nutrition_image": str(dest),
                    "ingredients_image": ingredients_path,
                    "ground_truth_nutriments": product.get("nutriments", {}),
                }
            )
            found += 1

        print(f"  {checked} urun kontrol edildi, {found} besin tablosu gorseli bulundu")

    with MANIFEST_PATH.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"\nToplam {len(manifest)} besin tablosu gorseli + ground-truth nutriments -> {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
