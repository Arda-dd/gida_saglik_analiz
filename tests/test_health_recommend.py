from src.common.schema import Allergen, NutritionFacts, ProductCategory, ProductRecord
from src.health.profile import ChronicCondition, HealthProfile
from src.health.recommend import build_health_assessment, recommend_alternatives


def make_product(product_id, category, allergens=None, **nutrition_kwargs) -> ProductRecord:
    return ProductRecord(
        product_id=product_id,
        category=category,
        nutrition=NutritionFacts(**nutrition_kwargs),
        allergens=allergens or [],
        source="test",
    )


class TestBuildHealthAssessment:
    def test_diabetic_gets_extra_warning_for_high_sugar_product(self):
        profile = HealthProfile(profile_id="p1", chronic_conditions=[ChronicCondition.DIYABET])
        nutrition = NutritionFacts(sugar_g=35.0)

        assessment = build_health_assessment(nutrition, [], profile)

        assert any("Diyabet" in msg for msg in assessment.health_risk_messages)
        assert assessment.allergen_warning is False

    def test_non_diabetic_does_not_get_diabetes_warning(self):
        profile = HealthProfile(profile_id="p2")
        nutrition = NutritionFacts(sugar_g=35.0)

        assessment = build_health_assessment(nutrition, [], profile)

        assert not any("Diyabet" in msg for msg in assessment.health_risk_messages)
        # Genel risk motoru uyarisi (Faz 3) yine de gelmeli.
        assert len(assessment.health_risk_messages) >= 1

    def test_allergen_conflict_is_flagged(self):
        profile = HealthProfile(profile_id="p3", allergens=[Allergen.FINDIK])
        nutrition = NutritionFacts(sugar_g=2.0)

        assessment = build_health_assessment(nutrition, [Allergen.FINDIK], profile)

        assert assessment.allergen_warning is True
        assert assessment.allergen_conflicts == [Allergen.FINDIK]

    def test_same_product_different_profiles_yield_different_assessments(self):
        # Profile Consistency: ayni urun, farkli profiller icin farkli (dogru) sonuc uretmeli.
        nutrition = NutritionFacts(sugar_g=35.0, salt_g=2.0)
        diabetic_profile = HealthProfile(profile_id="p4", chronic_conditions=[ChronicCondition.DIYABET])
        healthy_profile = HealthProfile(profile_id="p5")

        diabetic_assessment = build_health_assessment(nutrition, [], diabetic_profile)
        healthy_assessment = build_health_assessment(nutrition, [], healthy_profile)

        assert diabetic_assessment.diet_compliance_score < healthy_assessment.diet_compliance_score
        assert diabetic_assessment.health_risk_messages != healthy_assessment.health_risk_messages


class TestRecommendAlternatives:
    def test_recommends_lower_risk_same_category_product(self):
        current = make_product("cur", ProductCategory.ATISTIRMALIK, sugar_g=40.0, salt_g=2.0)
        better = make_product("alt1", ProductCategory.ATISTIRMALIK, sugar_g=3.0, salt_g=0.2)
        candidates = [better]
        profile = HealthProfile(profile_id="p1")

        result = recommend_alternatives(current, candidates, profile)

        assert result == [better]

    def test_excludes_different_category(self):
        current = make_product("cur", ProductCategory.ATISTIRMALIK, sugar_g=40.0)
        wrong_category = make_product("alt1", ProductCategory.ICECEK, sugar_g=1.0)
        profile = HealthProfile(profile_id="p1")

        result = recommend_alternatives(current, [wrong_category], profile)

        assert result == []

    def test_excludes_candidate_with_allergen_conflict(self):
        current = make_product("cur", ProductCategory.ATISTIRMALIK, sugar_g=40.0)
        allergenic = make_product(
            "alt1", ProductCategory.ATISTIRMALIK, allergens=[Allergen.FINDIK], sugar_g=1.0
        )
        profile = HealthProfile(profile_id="p1", allergens=[Allergen.FINDIK])

        result = recommend_alternatives(current, [allergenic], profile)

        assert result == []

    def test_excludes_candidate_with_equal_or_higher_risk(self):
        current = make_product("cur", ProductCategory.ATISTIRMALIK, sugar_g=40.0)
        equally_risky = make_product("alt1", ProductCategory.ATISTIRMALIK, sugar_g=30.0)
        profile = HealthProfile(profile_id="p1")

        result = recommend_alternatives(current, [equally_risky], profile)

        assert result == []

    def test_excludes_self(self):
        current = make_product("cur", ProductCategory.ATISTIRMALIK, sugar_g=40.0)
        profile = HealthProfile(profile_id="p1")

        result = recommend_alternatives(current, [current], profile)

        assert result == []

    def test_returns_empty_list_when_no_better_alternative_exists(self):
        current = make_product("cur", ProductCategory.ATISTIRMALIK, sugar_g=2.0)
        profile = HealthProfile(profile_id="p1")

        result = recommend_alternatives(current, [], profile)

        assert result == []

    def test_ranks_by_ascending_risk_and_respects_max_results(self):
        current = make_product(
            "cur", ProductCategory.ATISTIRMALIK, sugar_g=40.0, salt_g=3.0, saturated_fat_g=10.0
        )
        best = make_product("alt_best", ProductCategory.ATISTIRMALIK, sugar_g=1.0)
        medium = make_product("alt_medium", ProductCategory.ATISTIRMALIK, sugar_g=1.0, salt_g=2.0)
        profile = HealthProfile(profile_id="p1")

        result = recommend_alternatives(
            current, [medium, best], profile, max_results=1
        )

        assert result == [best]
