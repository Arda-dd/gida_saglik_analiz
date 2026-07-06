"""Etiket verilerinin birim donusumleri ve normalizasyonu.

Referanslar (bkz. oneri formu, 2.1 - Veri Toplama ve On Isleme):
- Enerji: 1 kcal = 4.184 kJ
- Sodyum -> Tuz: Tuz (g) = Sodyum (g) x 2.5  (Turk Gida Kodeksi)
- "porsiyon basina" degerler, "100 g/100 mL basina" formuna cevrilir.
"""

from __future__ import annotations

from src.common.schema import NutritionFacts

KCAL_TO_KJ = 4.184
SODIUM_TO_SALT_FACTOR = 2.5


def kcal_to_kj(kcal: float) -> float:
    return kcal * KCAL_TO_KJ


def kj_to_kcal(kj: float) -> float:
    return kj / KCAL_TO_KJ


def salt_g_from_sodium_mg(sodium_mg: float) -> float:
    """Sodyum (mg) -> Tuz (g). Tuz(g) = Sodyum(g) x 2.5"""
    sodium_g = sodium_mg / 1000
    return sodium_g * SODIUM_TO_SALT_FACTOR


def sodium_mg_from_salt_g(salt_g: float) -> float:
    """Tuz (g) -> Sodyum (mg)."""
    sodium_g = salt_g / SODIUM_TO_SALT_FACTOR
    return sodium_g * 1000


def per_serving_to_per_100(value_per_serving: float, serving_size_g: float) -> float:
    """Porsiyon basina bir degeri 100g/100mL basina cevirir.

    Ornek: 1 porsiyon (30g) icin 5g seker -> 100g'da 16.67g seker.
    """
    if serving_size_g <= 0:
        raise ValueError("serving_size_g pozitif olmalidir")
    return value_per_serving * 100 / serving_size_g


def normalize_nutrition_to_per_100(
    facts: NutritionFacts, serving_size_g: float
) -> NutritionFacts:
    """Porsiyon bazli bir NutritionFacts nesnesini 100g/100mL bazina cevirir.

    Sadece dolu (None olmayan) alanlar donusturulur.
    """
    data = facts.model_dump()
    converted = {
        key: (per_serving_to_per_100(value, serving_size_g) if value is not None else None)
        for key, value in data.items()
    }
    return NutritionFacts(**converted)


def fill_missing_energy_and_salt(facts: NutritionFacts) -> NutritionFacts:
    """Enerji (kcal/kJ) ve tuz/sodyum alanlarindan biri eksikse digerinden turetir."""
    data = facts.model_dump()

    if data["energy_kcal"] is None and data["energy_kj"] is not None:
        data["energy_kcal"] = kj_to_kcal(data["energy_kj"])
    elif data["energy_kj"] is None and data["energy_kcal"] is not None:
        data["energy_kj"] = kcal_to_kj(data["energy_kcal"])

    if data["salt_g"] is None and data["sodium_mg"] is not None:
        data["salt_g"] = salt_g_from_sodium_mg(data["sodium_mg"])
    elif data["sodium_mg"] is None and data["salt_g"] is not None:
        data["sodium_mg"] = sodium_mg_from_salt_g(data["salt_g"])

    return NutritionFacts(**data)
