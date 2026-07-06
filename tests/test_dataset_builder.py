import pandas as pd

from src.common.schema import NutritionFacts, ProductCategory, ProductRecord
from src.data.dataset_builder import DATASET_COLUMNS, build_dataset, load_dataset, save_dataset


def _off_record(product_id="off_1", **nutrition_kwargs) -> ProductRecord:
    return ProductRecord(
        product_id=product_id,
        category=ProductCategory.ICECEK,
        nutrition=NutritionFacts(**nutrition_kwargs),
        source="open_food_facts",
    )


def _local_manifest(rows: list[dict]) -> pd.DataFrame:
    columns = ["product_id", "category", "image_path", "width", "height", "blur_score", "is_valid", "reasons"]
    return pd.DataFrame(rows, columns=columns)


def test_build_dataset_combines_off_and_local():
    off_records = [_off_record("off_1", energy_kcal=100, fat_g=1, carbohydrate_g=1, protein_g=1)]
    local_df = _local_manifest(
        [
            {
                "product_id": "local_icecek_0001",
                "category": "icecek",
                "image_path": "data/raw/local/icecek/local_icecek_0001.jpg",
                "width": 1800,
                "height": 1800,
                "blur_score": 200.0,
                "is_valid": True,
                "reasons": "",
            }
        ]
    )

    df = build_dataset(off_records, local_df)

    assert len(df) == 2
    assert set(df["product_id"]) == {"off_1", "local_icecek_0001"}
    assert list(df.columns) == DATASET_COLUMNS


def test_build_dataset_excludes_invalid_local_images():
    off_records = []
    local_df = _local_manifest(
        [
            {
                "product_id": "local_icecek_0001",
                "category": "icecek",
                "image_path": "x.jpg",
                "width": 500,
                "height": 500,
                "blur_score": 50.0,
                "is_valid": False,
                "reasons": "cozunurluk_dusuk",
            }
        ]
    )

    df = build_dataset(off_records, local_df)
    assert df.empty


def test_build_dataset_excludes_invalid_off_records():
    # enerji ve tum makrolar eksik -> validation.py tarafindan reddedilir
    off_records = [_off_record("off_bad")]
    local_df = _local_manifest([])

    df = build_dataset(off_records, local_df)
    assert df.empty


def test_build_dataset_local_rows_have_null_nutrition():
    off_records = []
    local_df = _local_manifest(
        [
            {
                "product_id": "local_icecek_0001",
                "category": "icecek",
                "image_path": "x.jpg",
                "width": 1800,
                "height": 1800,
                "blur_score": 200.0,
                "is_valid": True,
                "reasons": "",
            }
        ]
    )

    df = build_dataset(off_records, local_df)
    assert df.iloc[0]["energy_kcal"] is None


def test_build_dataset_empty_inputs_returns_empty_with_correct_columns():
    df = build_dataset([], _local_manifest([]))
    assert df.empty
    assert list(df.columns) == DATASET_COLUMNS


def test_save_and_load_dataset_roundtrip(tmp_path):
    off_records = [_off_record("off_1", energy_kcal=100, fat_g=1, carbohydrate_g=1, protein_g=1)]
    df = build_dataset(off_records, _local_manifest([]))

    out_path = tmp_path / "dataset.csv"
    save_dataset(df, out_path)
    loaded = load_dataset(out_path)

    assert len(loaded) == len(df)
    assert loaded.iloc[0]["product_id"] == "off_1"
