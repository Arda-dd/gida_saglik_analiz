from unittest.mock import MagicMock

from src.common.schema import Allergen, ProductCategory
from src.data.off_client import (
    fetch_products_by_category,
    off_record_to_product_record,
)


def _mock_response(json_data: dict) -> MagicMock:
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = json_data
    return response


def test_fetch_products_by_category_paginates_until_empty():
    session = MagicMock()
    session.get.side_effect = [
        _mock_response({"products": [{"code": "1"}, {"code": "2"}]}),
        _mock_response({"products": []}),
    ]

    products = fetch_products_by_category(
        "en:dairies", max_pages=5, session=session, sleep_between_pages=0
    )

    assert len(products) == 2
    assert session.get.call_count == 2


def test_fetch_products_by_category_respects_max_pages():
    session = MagicMock()
    session.get.return_value = _mock_response({"products": [{"code": "1"}]})

    products = fetch_products_by_category(
        "en:dairies", max_pages=3, session=session, sleep_between_pages=0
    )

    assert session.get.call_count == 3
    assert len(products) == 3


def test_off_record_to_product_record_maps_real_shaped_json():
    off_json = {
        "code": "6111246721261",
        "categories_tags": ["en:dairies", "en:cheeses"],
        "nutriments": {
            "energy-kcal_100g": 159,
            "energy-kj_100g": 643,
            "fat_100g": 11,
            "saturated-fat_100g": 7.15,
            "carbohydrates_100g": 10,
            "sugars_100g": 4,
            "fiber_100g": 0,
            "proteins_100g": 5,
            "salt_100g": 0.1,
            "sodium_100g": 0.04,
        },
        "allergens_tags": ["en:milk"],
    }

    record = off_record_to_product_record(off_json)

    assert record is not None
    assert record.product_id == "off_6111246721261"
    assert record.category == ProductCategory.SUT_URUNU
    assert record.nutrition.energy_kcal == 159
    assert record.nutrition.sodium_mg == 40  # 0.04g * 1000
    assert record.allergens == [Allergen.LAKTOZ]
    assert record.source == "open_food_facts"


def test_off_record_to_product_record_missing_code_returns_none():
    assert off_record_to_product_record({"categories_tags": []}) is None


def test_off_record_to_product_record_missing_nutriments_are_none():
    off_json = {"code": "123", "categories_tags": []}
    record = off_record_to_product_record(off_json)

    assert record is not None
    assert record.nutrition.energy_kcal is None
    assert record.nutrition.sodium_mg is None
    assert record.category == ProductCategory.BILINMIYOR


def test_off_record_to_product_record_unmapped_allergens_ignored():
    off_json = {"code": "123", "categories_tags": [], "allergens_tags": ["en:celery", "en:milk"]}
    record = off_record_to_product_record(off_json)

    assert record.allergens == [Allergen.LAKTOZ]
