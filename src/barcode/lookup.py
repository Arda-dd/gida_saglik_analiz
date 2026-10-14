"""Barkoddan Open Food Facts (OFF) urun sorgusu.

Barkod bulunursa urunun OFF'ta kayitli besin degerleri, kategorisi ve alerjenleri kullanilir -
OCR'in okuma hatalarindan bagimsiz, kategori CNN'inin zayifligini (%75) da telafi eden bir kaynak.
Urun OFF'ta yoksa ya da besin verisi bossa pipeline OCR sonucuna devam eder (api/pipeline.py).

Canli istek icin kisa zaman asimi ve TEKRARSIZ istek kullanilir: toplu veri indirme icin yazilan
src/data/off_client.py'deki 3 x 20 sn'lik tekrar mantigi OFF yavasken kullaniciyi bir dakika
bekletirdi. Ayni barkod ayni surecte tekrar sorgulanmaz (lru_cache).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import requests

from src.common.schema import Allergen, ProductRecord
from src.data.off_client import PRODUCT_URL, USER_AGENT, _map_off_allergens, off_record_to_product_record

LOOKUP_FIELDS = "code,product_name,nutriments,categories_tags,allergens_tags,traces_tags"
LOOKUP_TIMEOUT_S = 8
# OFF kaydinin "kullanilabilir" sayilmasi icin en az bu kadar besin alani dolu olmali - tek bir
# alan (ornegin sadece enerji) girilmis eksik kayitlar OCR sonucunun yerini almamali.
MIN_NUTRITION_FIELDS = 3


@dataclass
class BarcodeProduct:
    barcode: str
    record: ProductRecord
    product_name: str | None = None
    traces: list[Allergen] = field(default_factory=list)  # "eser miktarda icerebilir"

    @property
    def nutrition_field_count(self) -> int:
        return len(self.record.nutrition.model_dump(exclude_none=True))

    @property
    def has_usable_nutrition(self) -> bool:
        return self.nutrition_field_count >= MIN_NUTRITION_FIELDS


def fetch_off_product(barcode: str, session: requests.Session | None = None) -> dict | None:
    """OFF tekil urun API'sinden urunu ceker; urun yoksa None, ag hatasinda istisna firlatir."""
    http = session or requests
    response = http.get(
        PRODUCT_URL.format(code=barcode),
        params={"fields": LOOKUP_FIELDS},
        headers={"User-Agent": USER_AGENT},
        timeout=LOOKUP_TIMEOUT_S,
    )
    if response.status_code == 404:
        return None
    response.raise_for_status()
    data = response.json()
    if data.get("status") != 1:
        return None
    return data.get("product")


@lru_cache(maxsize=512)
def lookup_barcode(barcode: str) -> BarcodeProduct | None:
    """Barkodu OFF'ta arar. Urun yoksa None; ag/API hatasi cagirana firlatilir (pipeline
    bunu yakalayip OCR'a devam eder ve uyari ekler)."""
    product = fetch_off_product(barcode)
    if product is None:
        return None
    product.setdefault("code", barcode)
    record = off_record_to_product_record(product)
    if record is None:
        return None
    return BarcodeProduct(
        barcode=barcode,
        record=record,
        product_name=product.get("product_name") or None,
        traces=_map_off_allergens(product.get("traces_tags")),
    )
