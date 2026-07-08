"""Open Food Facts (OFF) REST API v2 istemcisi.

Tam CSV/MongoDB dump'i (GB'larca) yerine, OFF'un v2 arama API'sini
(world.openfoodfacts.org/api/v2/search) sayfalayarak kullanir.

Onemli bulgu (bkz. Faz 1 arastirmasi): OFF, nutriments alanlarini sunucu tarafinda
zaten 100g/100mL bazina normalize eder - '_100g' son ekli alanlar `nutrition_data_per`
degeri (100g veya porsiyon) fark etmeksizin daima 100g bazindadir. Bu yuzden bu
istemci ayrica bir porsiyon->100g donusumu yapmaz (src/common/units.py burada
kullanilmaz - OFF verisi zaten normalize gelir).

Not: OFF'ta 'sodium_100g' GRAM cinsindendir; bizim NutritionFacts.sodium_mg alanimiz
miligram bekledigi icin x1000 carpimi uygulanir.
"""

from __future__ import annotations

import time
from pathlib import Path

import requests

from src.common.schema import Allergen, NutritionFacts, ProductRecord
from src.data.off_category_map import map_categories_tags

BASE_URL = "https://world.openfoodfacts.org/api/v2/search"
PRODUCT_URL = "https://world.openfoodfacts.org/api/v2/product/{code}.json"
USER_AGENT = (
    "GidaSaglikAsistani-TUBITAK2209A/0.1 "
    "(egitim amacli arastirma projesi; iletisim: ardatnmzoglu@gmail.com)"
)
DEFAULT_FIELDS = "code,product_name,categories_tags,nutriments,image_front_url,allergens_tags"

# OFF'un 'allergens_tags' degerlerinden bizim daralt&ilmis Allergen enum'imize esleme.
# Not: OFF'un tam alerjen listesi bizimkinden genistir (kereviz, hardal, susam, sulfit,
# lupin, yumusakca, kabuklu deniz urunu, yer fistigi vb. burada yoktur) - bkz.
# data/knowledge_base/docs/allergen_labeling_overview.md.
OFF_ALLERGEN_MAP: dict[str, Allergen] = {
    "en:milk": Allergen.LAKTOZ,
    "en:gluten": Allergen.GLUTEN,
    "en:nuts": Allergen.FINDIK,
    "en:hazelnuts": Allergen.FINDIK,
    "en:soybeans": Allergen.SOYA,
    "en:eggs": Allergen.YUMURTA,
    "en:fish": Allergen.BALIK,
}


RETRYABLE_STATUS_CODES = {503, 429}


def _get_with_retry(
    session: requests.Session, url: str, params: dict, headers: dict, timeout: int, max_retries: int = 3
) -> requests.Response:
    """Gecici sunucu hatalarinda (503/429) ustel bekleme ile yeniden dener."""
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        response = session.get(url, params=params, headers=headers, timeout=timeout)
        if response.status_code not in RETRYABLE_STATUS_CODES:
            return response
        last_exc = requests.exceptions.HTTPError(
            f"{response.status_code} gecici hata (deneme {attempt + 1}/{max_retries})", response=response
        )
        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)  # 1s, 2s, 4s...

    assert last_exc is not None
    raise last_exc


def fetch_products_by_category(
    category_tag: str,
    page_size: int = 100,
    max_pages: int = 3,
    fields: str = DEFAULT_FIELDS,
    countries_tags: str | None = None,
    session: requests.Session | None = None,
    sleep_between_pages: float = 1.0,
) -> list[dict]:
    """OFF v2 arama API'sinden bir kategoriye ait urun listesini sayfalayarak ceker.

    Gecici (503/429) sunucu hatalarinda otomatik olarak yeniden dener.
    """
    session = session or requests.Session()
    headers = {"User-Agent": USER_AGENT}
    all_products: list[dict] = []

    for page in range(1, max_pages + 1):
        params: dict = {
            "categories_tags": category_tag,
            "page": page,
            "page_size": page_size,
            "fields": fields,
        }
        if countries_tags:
            params["countries_tags"] = countries_tags

        response = _get_with_retry(session, BASE_URL, params, headers, timeout=20)
        response.raise_for_status()
        data = response.json()
        products = data.get("products", [])
        if not products:
            break

        all_products.extend(products)

        if page < max_pages and sleep_between_pages > 0:
            time.sleep(sleep_between_pages)  # OFF API'sine karsi kibarlik (rate limiting)

    return all_products


def fetch_single_product(
    code: str, fields: str, session: requests.Session | None = None
) -> dict | None:
    """OFF v2 tekil urun API'sinden (api/v2/product/{code}.json) belirli alanlari ceker.

    Onemli bulgu: /api/v2/search endpoint'i 'image_nutrition_url' / 'image_ingredients_url'
    gibi secili-goruntu alanlarini DONDURMEZ (arama indeksinde yok), fakat bu tekil urun
    endpoint'i dondurur. Besin tablosu/icindekiler gorseli gerektiginde bu fonksiyon kullanilir.
    """
    session = session or requests.Session()
    headers = {"User-Agent": USER_AGENT}
    url = PRODUCT_URL.format(code=code)

    response = _get_with_retry(session, url, {"fields": fields}, headers, timeout=20)
    if response.status_code != 200:
        return None

    data = response.json()
    if data.get("status") != 1:
        return None

    return data.get("product")


def download_product_image(
    image_url: str, dest_path: Path, timeout: int = 15
) -> Path | None:
    """Urun gorselini indirir; agdan kaynakli hatalarda None doner (kayit yine islenir)."""
    try:
        headers = {"User-Agent": USER_AGENT}
        response = requests.get(image_url, headers=headers, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException:
        return None

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_bytes(response.content)
    return dest_path


def _map_off_allergens(allergens_tags: list[str] | None) -> list[Allergen]:
    if not allergens_tags:
        return []
    mapped = {OFF_ALLERGEN_MAP[tag] for tag in allergens_tags if tag in OFF_ALLERGEN_MAP}
    return sorted(mapped, key=lambda a: a.value)


def off_record_to_product_record(off_json: dict) -> ProductRecord | None:
    """Tek bir OFF urun JSON'unu ortak ProductRecord semasina cevirir.

    Barkod (code) yoksa None doner. Kategori, categories_tags uzerinden
    off_category_map.map_categories_tags ile belirlenir (eslesme yoksa BILINMIYOR).
    """
    code = off_json.get("code")
    if not code:
        return None

    nutr = off_json.get("nutriments") or {}

    def _get(key: str) -> float | None:
        value = nutr.get(f"{key}_100g")
        return float(value) if value is not None else None

    sodium_g = _get("sodium")

    nutrition = NutritionFacts(
        energy_kcal=_get("energy-kcal"),
        energy_kj=_get("energy-kj"),
        fat_g=_get("fat"),
        saturated_fat_g=_get("saturated-fat"),
        carbohydrate_g=_get("carbohydrates"),
        sugar_g=_get("sugars"),
        fiber_g=_get("fiber"),
        protein_g=_get("proteins"),
        salt_g=_get("salt"),
        sodium_mg=(sodium_g * 1000) if sodium_g is not None else None,
    )

    category = map_categories_tags(off_json.get("categories_tags") or [])

    return ProductRecord(
        product_id=f"off_{code}",
        category=category,
        nutrition=nutrition,
        allergens=_map_off_allergens(off_json.get("allergens_tags")),
        source="open_food_facts",
    )
