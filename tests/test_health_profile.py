import pytest
from pydantic import ValidationError

from src.common.schema import Allergen
from src.health.profile import ChronicCondition, HealthProfile


def test_health_profile_defaults_to_empty_lists():
    profile = HealthProfile(profile_id="anon_0001")

    assert profile.chronic_conditions == []
    assert profile.allergens == []
    assert profile.daily_calorie_target_kcal is None


def test_health_profile_accepts_conditions_and_allergens():
    profile = HealthProfile(
        profile_id="anon_0002",
        chronic_conditions=[ChronicCondition.DIYABET, ChronicCondition.HIPERTANSIYON],
        allergens=[Allergen.FINDIK],
        daily_calorie_target_kcal=2000,
    )

    assert ChronicCondition.DIYABET in profile.chronic_conditions
    assert Allergen.FINDIK in profile.allergens
    assert profile.daily_calorie_target_kcal == 2000


def test_health_profile_rejects_non_positive_calorie_target():
    with pytest.raises(ValidationError):
        HealthProfile(profile_id="anon_0003", daily_calorie_target_kcal=0)


def test_health_profile_rejects_negative_macro_target():
    with pytest.raises(ValidationError):
        HealthProfile(profile_id="anon_0004", daily_protein_target_g=-10)


def test_health_profile_does_not_contain_pii_fields():
    # Profil sadece anonim nitelikler icerir - isim/e-posta/telefon gibi alanlar YOK.
    fields = HealthProfile.model_fields.keys()
    assert "name" not in fields
    assert "email" not in fields
    assert "phone" not in fields
