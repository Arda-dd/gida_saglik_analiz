"""Tum modullerin (OCR, vision, RAG, health) paylastigi ortak veri sozlesmesi.

Besin degerleri her zaman 100g veya 100mL bazinda normalize edilmis olarak
tasinir (bkz. src/common/units.py -> normalize_to_per_100).
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ProductCategory(str, Enum):
    SUT_URUNU = "sut_urunu"
    ATISTIRMALIK = "atistirmalik"
    ICECEK = "icecek"
    HAZIR_GIDA = "hazir_gida"
    KONSERVE = "konserve"
    BILINMIYOR = "bilinmiyor"


class Allergen(str, Enum):
    LAKTOZ = "laktoz"
    GLUTEN = "gluten"
    FINDIK = "findik"
    SOYA = "soya"
    YUMURTA = "yumurta"
    BALIK = "balik"


class UserObjective(str, Enum):
    KILO_VERME = "kilo_verme"
    PROTEIN_AGIRLIKLI = "protein_agirlikli"
    ALERJI_TAKIBI = "alerji_takibi"



class NutritionBasis(str, Enum):
    """Etikette degerlerin hangi baza gore verildigi (normalizasyon oncesi kaynak bilgisi)."""

    PER_100G = "per_100g"
    PER_100ML = "per_100ml"
    PER_SERVING = "per_serving"


class NutritionFacts(BaseModel):
    """100g veya 100mL bazina normalize edilmis besin degerleri.

    Tum alanlar opsiyoneldir: OCR/kaynak veride eksik olabilecek degerleri
    None birakmak, onlari 0 varsaymaktan (yanlis saglik degerlendirmesi
    riski) daha guvenlidir.
    """

    energy_kcal: float | None = Field(default=None, ge=0)
    energy_kj: float | None = Field(default=None, ge=0)
    fat_g: float | None = Field(default=None, ge=0)
    saturated_fat_g: float | None = Field(default=None, ge=0)
    carbohydrate_g: float | None = Field(default=None, ge=0)
    sugar_g: float | None = Field(default=None, ge=0)
    fiber_g: float | None = Field(default=None, ge=0)
    protein_g: float | None = Field(default=None, ge=0)
    salt_g: float | None = Field(default=None, ge=0)
    sodium_mg: float | None = Field(default=None, ge=0)


class ProductRecord(BaseModel):
    """Bir gida etiketinden uretilen tek bir kayit (pipeline'in ortak cikti birimi)."""

    product_id: str
    category: ProductCategory = ProductCategory.BILINMIYOR
    nutrition: NutritionFacts
    nutrition_basis: NutritionBasis = NutritionBasis.PER_100G
    allergens: list[Allergen] = Field(default_factory=list)
    source: str | None = None  # ör. "open_food_facts", "local_market"
