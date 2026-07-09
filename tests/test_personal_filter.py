import pytest

from src.common.schema import Allergen, NutritionFacts
from src.health.personal_filter import (
    check_allergen_conflict,
    compute_diet_compliance_score,
    condition_based_warnings,
)
from src.health.profile import ChronicCondition, HealthProfile


def make_profile(**kwargs) -> HealthProfile:
    defaults = {"profile_id": "anon_test"}
    defaults.update(kwargs)
    return HealthProfile(**defaults)


class TestConditionBasedWarnings:
    def test_diabetes_warns_on_high_sugar_flag(self):
        warnings = condition_based_warnings(["yuksek_seker"], [ChronicCondition.DIYABET])
        assert len(warnings) == 1
        assert "Diyabet" in warnings[0]

    def test_diabetes_does_not_warn_on_unrelated_flag(self):
        # Yanlis esik/yanlis kosul eslesmesi senaryosu: diyabet kosulu, tuz bayragiyla
        # tetiklenmemeli (sadece seker ile iliskilidir).
        warnings = condition_based_warnings(["yuksek_tuz"], [ChronicCondition.DIYABET])
        assert warnings == []

    def test_hypertension_warns_on_high_salt_and_sodium(self):
        warnings = condition_based_warnings(
            ["yuksek_tuz", "yuksek_sodyum"], [ChronicCondition.HIPERTANSIYON]
        )
        assert len(warnings) == 2

    def test_no_conditions_produces_no_warnings(self):
        warnings = condition_based_warnings(["yuksek_seker", "yuksek_tuz"], [])
        assert warnings == []

    def test_no_risk_flags_produces_no_warnings_even_with_conditions(self):
        warnings = condition_based_warnings([], [ChronicCondition.DIYABET, ChronicCondition.HIPERTANSIYON])
        assert warnings == []

    def test_multiple_conditions_do_not_cross_contaminate(self):
        # Diyabet kosulu icin sadece seker, hipertansiyon icin sadece tuz/sodyum uyarisi gelmeli.
        warnings = condition_based_warnings(
            ["yuksek_seker"], [ChronicCondition.DIYABET, ChronicCondition.HIPERTANSIYON]
        )
        assert len(warnings) == 1
        assert "Diyabet" in warnings[0]

    def test_kidney_disease_warns_on_high_sodium(self):
        warnings = condition_based_warnings(["yuksek_sodyum"], [ChronicCondition.BOBREK_HASTALIGI])
        assert len(warnings) == 1

    def test_heart_disease_warns_on_high_saturated_fat(self):
        warnings = condition_based_warnings(
            ["yuksek_doymus_yag"], [ChronicCondition.KALP_HASTALIGI]
        )
        assert len(warnings) == 1


class TestAllergenConflict:
    def test_detects_matching_allergen(self):
        conflicts = check_allergen_conflict([Allergen.FINDIK, Allergen.GLUTEN], [Allergen.FINDIK])
        assert conflicts == [Allergen.FINDIK]

    def test_no_conflict_when_allergens_differ(self):
        # Yanlis alerjen senaryosu: urun findik icerir ama kullanicinin alerjisi soyadir - celisme YOK.
        conflicts = check_allergen_conflict([Allergen.FINDIK], [Allergen.SOYA])
        assert conflicts == []

    def test_no_conflict_when_product_has_no_allergens(self):
        conflicts = check_allergen_conflict([], [Allergen.FINDIK, Allergen.GLUTEN])
        assert conflicts == []

    def test_no_conflict_when_profile_has_no_allergens(self):
        conflicts = check_allergen_conflict([Allergen.FINDIK], [])
        assert conflicts == []

    def test_multiple_matching_allergens(self):
        conflicts = check_allergen_conflict(
            [Allergen.FINDIK, Allergen.LAKTOZ, Allergen.SOYA], [Allergen.FINDIK, Allergen.SOYA]
        )
        assert set(conflicts) == {Allergen.FINDIK, Allergen.SOYA}


class TestDietComplianceScore:
    def test_no_targets_and_no_risks_gives_perfect_score(self):
        profile = make_profile()
        nutrition = NutritionFacts(energy_kcal=200)
        score = compute_diet_compliance_score(nutrition, profile, risk_flags=[])
        assert score == 100.0

    def test_high_calorie_share_reduces_score(self):
        profile = make_profile(daily_calorie_target_kcal=2000)
        # 100g'da 900 kcal -> gunluk hedefin %45'i, makul pay (%25) asiliyor.
        nutrition = NutritionFacts(energy_kcal=900)
        score = compute_diet_compliance_score(nutrition, profile, risk_flags=[])
        assert score < 100.0

    def test_calorie_share_within_reasonable_bound_does_not_penalize(self):
        profile = make_profile(daily_calorie_target_kcal=2000)
        # 100g'da 400 kcal -> gunluk hedefin %20'si, makul payin (%25) altinda.
        nutrition = NutritionFacts(energy_kcal=400)
        score = compute_diet_compliance_score(nutrition, profile, risk_flags=[])
        assert score == 100.0

    def test_condition_warning_reduces_score(self):
        profile = make_profile(chronic_conditions=[ChronicCondition.DIYABET])
        nutrition = NutritionFacts(energy_kcal=200)
        score = compute_diet_compliance_score(nutrition, profile, risk_flags=["yuksek_seker"])
        assert score == 75.0

    def test_score_never_goes_below_zero(self):
        profile = make_profile(
            daily_calorie_target_kcal=100,
            chronic_conditions=[
                ChronicCondition.DIYABET,
                ChronicCondition.HIPERTANSIYON,
                ChronicCondition.KALP_HASTALIGI,
            ],
        )
        nutrition = NutritionFacts(energy_kcal=900)
        score = compute_diet_compliance_score(
            nutrition,
            profile,
            risk_flags=["yuksek_seker", "yuksek_tuz", "yuksek_sodyum", "yuksek_doymus_yag"],
        )
        assert score == 0.0

    def test_missing_calorie_target_skips_calorie_penalty(self):
        profile = make_profile()  # daily_calorie_target_kcal=None
        nutrition = NutritionFacts(energy_kcal=900)
        score = compute_diet_compliance_score(nutrition, profile, risk_flags=[])
        assert score == 100.0
