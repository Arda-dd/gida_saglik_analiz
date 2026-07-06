from src.common.schema import ProductCategory
from src.data.off_category_map import map_categories_tags


def test_maps_known_dairy_tag():
    assert map_categories_tags(["en:dairies"]) == ProductCategory.SUT_URUNU


def test_maps_known_beverage_tag():
    assert map_categories_tags(["en:sodas"]) == ProductCategory.ICECEK


def test_maps_known_snack_tag():
    assert map_categories_tags(["en:chips-and-fries"]) == ProductCategory.ATISTIRMALIK


def test_maps_known_canned_tag():
    assert map_categories_tags(["en:canned-fish"]) == ProductCategory.KONSERVE


def test_maps_known_meal_tag():
    assert map_categories_tags(["en:ready-meals"]) == ProductCategory.HAZIR_GIDA


def test_unknown_tag_maps_to_bilinmiyor():
    assert map_categories_tags(["en:not-a-real-category"]) == ProductCategory.BILINMIYOR


def test_empty_list_maps_to_bilinmiyor():
    assert map_categories_tags([]) == ProductCategory.BILINMIYOR


def test_priority_order_canned_wins_over_meal():
    # Bir urun hem konserve hem hazir_gida tag'ine sahipse, konserve (daha spesifik/oncelikli) kazanmali.
    tags = ["en:meals", "en:canned-vegetables"]
    assert map_categories_tags(tags) == ProductCategory.KONSERVE


def test_multiple_unrelated_tags_finds_correct_match():
    tags = ["en:organic", "en:beverages", "en:some-other-tag"]
    assert map_categories_tags(tags) == ProductCategory.ICECEK
