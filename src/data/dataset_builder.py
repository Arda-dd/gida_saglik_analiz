"""OFF ve yerel market verilerini tek bir `data/processed/dataset.csv`'de birlestirir.

Yerel goruntuler icin besin degerleri henuz bilinmez (OCR Faz 3'te yapilacaktir) -
bu satirlarda beslenme kolonlari NaN birakilir, sadece kategori + goruntu yolu doldurulur.
OFF kayitlari, dataset'e girmeden once src.data.validation.filter_valid_records ile
otomatik olarak dogrulanir/elenir.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common.schema import ProductRecord
from src.data.validation import filter_valid_records

DATASET_COLUMNS = [
    "product_id",
    "category",
    "source",
    "image_path",
    "energy_kcal",
    "energy_kj",
    "fat_g",
    "saturated_fat_g",
    "carbohydrate_g",
    "sugar_g",
    "fiber_g",
    "protein_g",
    "salt_g",
    "sodium_mg",
    "allergens",
]


def _product_record_to_row(record: ProductRecord, image_path: str | None = None) -> dict:
    n = record.nutrition
    return {
        "product_id": record.product_id,
        "category": record.category.value,
        "source": record.source,
        "image_path": image_path,
        "energy_kcal": n.energy_kcal,
        "energy_kj": n.energy_kj,
        "fat_g": n.fat_g,
        "saturated_fat_g": n.saturated_fat_g,
        "carbohydrate_g": n.carbohydrate_g,
        "sugar_g": n.sugar_g,
        "fiber_g": n.fiber_g,
        "protein_g": n.protein_g,
        "salt_g": n.salt_g,
        "sodium_mg": n.sodium_mg,
        "allergens": ";".join(a.value for a in record.allergens),
    }


def _local_manifest_row(row: pd.Series) -> dict:
    return {
        "product_id": row["product_id"],
        "category": row["category"],
        "source": "local_market",
        "image_path": row["image_path"],
        "energy_kcal": None,
        "energy_kj": None,
        "fat_g": None,
        "saturated_fat_g": None,
        "carbohydrate_g": None,
        "sugar_g": None,
        "fiber_g": None,
        "protein_g": None,
        "salt_g": None,
        "sodium_mg": None,
        "allergens": "",
    }


def build_dataset(
    off_records: list[ProductRecord], local_manifest_df: pd.DataFrame
) -> pd.DataFrame:
    """OFF kayitlarini dogrulayip yerel manifestle birlestirerek birlesik veri setini uretir.

    Sadece kalite kontrolunden gecmis (is_valid=True) yerel goruntuler dahil edilir.
    """
    valid_off_records, _ = filter_valid_records(off_records)
    off_rows = [_product_record_to_row(r) for r in valid_off_records]

    local_rows = []
    if not local_manifest_df.empty:
        valid_local = local_manifest_df[local_manifest_df["is_valid"]]
        local_rows = [_local_manifest_row(row) for _, row in valid_local.iterrows()]

    all_rows = off_rows + local_rows
    if not all_rows:
        return pd.DataFrame(columns=DATASET_COLUMNS)

    return pd.DataFrame(all_rows, columns=DATASET_COLUMNS)


def save_dataset(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def load_dataset(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)
