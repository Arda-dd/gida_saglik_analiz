import pytest

from src.common.schema import NutritionFacts
from src.common.units import (
    fill_missing_energy_and_salt,
    kcal_to_kj,
    kj_to_kcal,
    normalize_nutrition_to_per_100,
    per_serving_to_per_100,
    salt_g_from_sodium_mg,
    sodium_mg_from_salt_g,
)


def test_kcal_to_kj():
    assert kcal_to_kj(100) == pytest.approx(418.4)


def test_kj_to_kcal_roundtrip():
    kcal = 250.0
    assert kj_to_kcal(kcal_to_kj(kcal)) == pytest.approx(kcal)


def test_salt_g_from_sodium_mg():
    # 400 mg sodyum -> 0.4 g sodyum -> 1.0 g tuz (x2.5)
    assert salt_g_from_sodium_mg(400) == pytest.approx(1.0)


def test_sodium_mg_from_salt_g_roundtrip():
    salt = 2.1
    assert salt_g_from_sodium_mg(sodium_mg_from_salt_g(salt)) == pytest.approx(salt)


def test_per_serving_to_per_100():
    # 30 g porsiyonda 5 g seker -> 100 g'da 16.67 g
    assert per_serving_to_per_100(5, 30) == pytest.approx(16.6667, rel=1e-4)


def test_per_serving_to_per_100_invalid_serving_size():
    with pytest.raises(ValueError):
        per_serving_to_per_100(5, 0)


def test_normalize_nutrition_to_per_100_scales_all_fields():
    facts = NutritionFacts(energy_kcal=60, sugar_g=5, salt_g=0.3)
    normalized = normalize_nutrition_to_per_100(facts, serving_size_g=30)
    assert normalized.energy_kcal == pytest.approx(200)
    assert normalized.sugar_g == pytest.approx(16.6667, rel=1e-4)
    assert normalized.salt_g == pytest.approx(1.0)


def test_normalize_nutrition_to_per_100_keeps_none_fields():
    facts = NutritionFacts(energy_kcal=60)
    normalized = normalize_nutrition_to_per_100(facts, serving_size_g=30)
    assert normalized.protein_g is None


def test_fill_missing_energy_derives_kj_from_kcal():
    facts = NutritionFacts(energy_kcal=100)
    filled = fill_missing_energy_and_salt(facts)
    assert filled.energy_kj == pytest.approx(418.4)


def test_fill_missing_salt_derives_from_sodium():
    facts = NutritionFacts(sodium_mg=400)
    filled = fill_missing_energy_and_salt(facts)
    assert filled.salt_g == pytest.approx(1.0)


def test_fill_missing_does_not_override_existing_values():
    facts = NutritionFacts(energy_kcal=100, energy_kj=999)
    filled = fill_missing_energy_and_salt(facts)
    assert filled.energy_kj == 999
