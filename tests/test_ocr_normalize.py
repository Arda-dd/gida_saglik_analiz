import pytest

from src.common.schema import NutritionBasis
from src.ocr.normalize import (
    detect_nutrition_basis,
    extract_and_normalize,
    extract_energy,
    extract_fat,
    extract_nutrition_facts,
    extract_salt,
    extract_saturated_fat,
    extract_sodium,
)


def test_extract_energy_reads_kcal():
    kcal, kj = extract_energy("Enerji: 450 kcal")
    assert kcal == 450
    assert kj is None


def test_extract_energy_reads_both_kcal_and_kj():
    kcal, kj = extract_energy("Enerji 450 kcal / 1883 kJ")
    assert kcal == 450
    assert kj == 1883


def test_extract_energy_english_label():
    kcal, kj = extract_energy("Energy: 200 kcal")
    assert kcal == 200


def test_extract_fat_does_not_confuse_with_saturated_fat():
    text = "Doymus yag: 5 g Yag: 12 g"
    assert extract_fat(text) == 12
    assert extract_saturated_fat(text) == 5


def test_extract_fat_only_saturated_present_returns_none_for_total():
    text = "Doymus yag: 5 g"
    assert extract_fat(text) is None
    assert extract_saturated_fat(text) == 5


def test_extract_salt_comma_decimal():
    assert extract_salt("Tuz: 1,2 g") == pytest.approx(1.2)


def test_extract_sodium_in_mg():
    assert extract_sodium("Sodyum: 400 mg") == 400


def test_extract_sodium_in_grams_converted_to_mg():
    assert extract_sodium("Sodium: 0.4 g") == pytest.approx(400)


def test_extract_nutrition_facts_full_label():
    text = (
        "Enerji: 450 kcal Yag: 12 g Doymus yag: 5 g Karbonhidrat: 30 g "
        "Seker: 10 g Lif: 2 g Protein: 8 g Tuz: 1.2 g"
    )
    facts = extract_nutrition_facts(text)

    assert facts.energy_kcal == 450
    assert facts.fat_g == 12
    assert facts.saturated_fat_g == 5
    assert facts.carbohydrate_g == 30
    assert facts.sugar_g == 10
    assert facts.fiber_g == 2
    assert facts.protein_g == 8
    assert facts.salt_g == pytest.approx(1.2)
    # sodyum metinde yoktu ama tuzdan turetilmis olmali (fill_missing_energy_and_salt)
    assert facts.sodium_mg == pytest.approx(480)  # 1.2 / 2.5 * 1000


def test_extract_nutrition_facts_missing_values_are_none():
    facts = extract_nutrition_facts("Enerji: 450 kcal")
    assert facts.protein_g is None
    assert facts.fiber_g is None


def test_detect_nutrition_basis_100g():
    basis, serving = detect_nutrition_basis("Besin degerleri 100 g icin")
    assert basis == NutritionBasis.PER_100G
    assert serving is None


def test_detect_nutrition_basis_100ml():
    basis, serving = detect_nutrition_basis("100 ml basina degerler")
    assert basis == NutritionBasis.PER_100ML


def test_detect_nutrition_basis_serving_with_size():
    basis, serving = detect_nutrition_basis("Porsiyon (30 g) basina degerler")
    assert basis == NutritionBasis.PER_SERVING
    assert serving == 30


def test_detect_nutrition_basis_serving_without_size():
    basis, serving = detect_nutrition_basis("Bir porsiyonda bulunan degerler")
    assert basis == NutritionBasis.PER_SERVING
    assert serving is None


def test_detect_nutrition_basis_defaults_to_100g_when_unclear():
    basis, serving = detect_nutrition_basis("Enerji 450 kcal")
    assert basis == NutritionBasis.PER_100G


def test_extract_and_normalize_converts_serving_to_100g():
    text = "Porsiyon (30 g) icin: Enerji: 60 kcal Seker: 5 g"
    facts, basis = extract_and_normalize(text)

    assert basis == NutritionBasis.PER_100G  # normalize edildi
    assert facts.energy_kcal == pytest.approx(200)  # 60 * 100/30
    assert facts.sugar_g == pytest.approx(16.6667, rel=1e-4)


def test_extract_and_normalize_leaves_per_100g_untouched():
    text = "100 g icin: Enerji: 450 kcal"
    facts, basis = extract_and_normalize(text)

    assert basis == NutritionBasis.PER_100G
    assert facts.energy_kcal == 450
