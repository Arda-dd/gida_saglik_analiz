"""Open Food Facts 'categories_tags' taksonomisinden bizim 5 kategorimize esleme.

Siralama onemlidir: bir urun birden fazla tag'e sahip olabilir (ornegin hem
"en:meals" hem "en:canned-foods"), bu yuzden en spesifik/oncelikli kategoriler
listenin basinda, en genel olan (hazir_gida) en sonda yer alir.
"""

from __future__ import annotations

from src.common.schema import ProductCategory

OFF_CATEGORY_PRIORITY: list[tuple[ProductCategory, set[str]]] = [
    (
        ProductCategory.KONSERVE,
        {
            "en:canned-foods",
            "en:canned-vegetables",
            "en:canned-fish",
            "en:preserves",
            "en:jams",
            "en:pickles",
        },
    ),
    (
        ProductCategory.SUT_URUNU,
        {
            "en:dairies",
            "en:milks",
            "en:yogurts",
            "en:cheeses",
            "en:fermented-milk-products",
            "en:dairy-desserts",
        },
    ),
    (
        ProductCategory.ICECEK,
        {
            "en:beverages",
            "en:sodas",
            "en:fruit-juices",
            "en:waters",
            "en:energy-drinks",
            "en:teas",
            "en:plant-based-beverages",
        },
    ),
    (
        ProductCategory.ATISTIRMALIK,
        {
            "en:snacks",
            "en:salty-snacks",
            "en:sweet-snacks",
            "en:biscuits-and-cakes",
            "en:chips-and-fries",
            "en:chocolates",
            "en:cereal-bars",
        },
    ),
    (
        ProductCategory.HAZIR_GIDA,
        {
            "en:meals",
            "en:ready-meals",
            "en:frozen-foods",
            "en:pizzas",
            "en:instant-soups",
            "en:pastas",
            "en:sandwiches",
        },
    ),
]


def map_categories_tags(categories_tags: list[str]) -> ProductCategory:
    """OFF categories_tags listesini oncelik sirasina gore bizim kategorimize cevirir.

    Ilk eslesen kategori dondurulur; hicbiri eslesmezse BILINMIYOR.
    """
    if not categories_tags:
        return ProductCategory.BILINMIYOR

    tags_set = set(categories_tags)
    for category, off_tags in OFF_CATEGORY_PRIORITY:
        if tags_set & off_tags:
            return category

    return ProductCategory.BILINMIYOR
