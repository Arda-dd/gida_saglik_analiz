from src.common.schema import NutritionFacts
from src.ocr.risk_engine import (
    assess_risks,
    assess_salt_risk,
    assess_saturated_fat_risk,
    assess_sodium_risk,
    assess_sugar_risk,
    describe_risks,
)


def test_assess_sugar_risk_above_threshold():
    # esik 22.5 g/100g (config.yaml)
    assert assess_sugar_risk(NutritionFacts(sugar_g=30)) is True


def test_assess_sugar_risk_below_threshold():
    assert assess_sugar_risk(NutritionFacts(sugar_g=5)) is False


def test_assess_sugar_risk_missing_value_is_false():
    assert assess_sugar_risk(NutritionFacts()) is False


def test_assess_salt_risk_above_threshold():
    # esik 1.5 g/100g
    assert assess_salt_risk(NutritionFacts(salt_g=2.0)) is True


def test_assess_salt_risk_below_threshold():
    assert assess_salt_risk(NutritionFacts(salt_g=0.5)) is False


def test_assess_saturated_fat_risk_above_threshold():
    # esik 5.0 g/100g
    assert assess_saturated_fat_risk(NutritionFacts(saturated_fat_g=8)) is True


def test_assess_saturated_fat_risk_below_threshold():
    assert assess_saturated_fat_risk(NutritionFacts(saturated_fat_g=2)) is False


def test_assess_sodium_risk_above_threshold():
    # esik 600 mg/100g
    assert assess_sodium_risk(NutritionFacts(sodium_mg=700)) is True


def test_assess_sodium_risk_below_threshold():
    assert assess_sodium_risk(NutritionFacts(sodium_mg=100)) is False


def test_assess_risks_returns_all_triggered_flags():
    nutrition = NutritionFacts(sugar_g=30, salt_g=2.0, saturated_fat_g=8, sodium_mg=700)
    flags = assess_risks(nutrition)

    assert set(flags) == {"yuksek_seker", "yuksek_tuz", "yuksek_doymus_yag", "yuksek_sodyum"}


def test_assess_risks_returns_empty_when_all_healthy():
    nutrition = NutritionFacts(sugar_g=1, salt_g=0.1, saturated_fat_g=0.5, sodium_mg=40)
    assert assess_risks(nutrition) == []


def test_assess_risks_partial_data_only_flags_available_fields():
    nutrition = NutritionFacts(sugar_g=30)  # digerleri None
    assert assess_risks(nutrition) == ["yuksek_seker"]


def test_describe_risks_returns_turkish_descriptions():
    descriptions = describe_risks(["yuksek_seker", "yuksek_tuz"])
    assert len(descriptions) == 2
    assert "seker" in descriptions[0].lower()
    assert "tuz" in descriptions[1].lower()
