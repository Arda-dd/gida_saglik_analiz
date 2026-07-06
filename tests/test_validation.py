from src.common.schema import NutritionFacts, ProductRecord
from src.data.validation import (
    filter_valid_records,
    find_duplicate_product_ids,
    safe_build_record,
    validate_record,
)


def _record(product_id="p1", **nutrition_kwargs) -> ProductRecord:
    return ProductRecord(product_id=product_id, nutrition=NutritionFacts(**nutrition_kwargs))


def test_missing_energy_and_all_macros_is_rejected():
    record = _record()  # her sey None
    issues = validate_record(record)
    assert any(i.rule == "missing_energy_and_macros" for i in issues)


def test_record_with_only_energy_is_not_missing():
    record = _record(energy_kcal=100)
    issues = validate_record(record)
    assert not any(i.rule == "missing_energy_and_macros" for i in issues)


def test_implausible_energy_over_900_is_flagged():
    record = _record(energy_kcal=950)
    issues = validate_record(record)
    assert any(i.rule == "implausible_energy" for i in issues)


def test_energy_at_900_is_not_flagged():
    record = _record(energy_kcal=900, fat_g=0, carbohydrate_g=0, protein_g=0)
    issues = validate_record(record)
    assert not any(i.rule == "implausible_energy" for i in issues)


def test_macro_energy_mismatch_flagged_but_not_rejecting():
    # fat=0,carb=0,protein=0 -> hesaplanan enerji 0, ama beyan 500kcal -> buyuk sapma
    record = _record(energy_kcal=500, fat_g=0, carbohydrate_g=0, protein_g=0)
    issues = validate_record(record)
    assert any(i.rule == "macro_energy_mismatch" for i in issues)


def test_macro_energy_consistent_not_flagged():
    # fat*9 + carb*4 + protein*4 = 10*9 + 20*4 + 5*4 = 90+80+20 = 190
    record = _record(energy_kcal=190, fat_g=10, carbohydrate_g=20, protein_g=5)
    issues = validate_record(record)
    assert not any(i.rule == "macro_energy_mismatch" for i in issues)


def test_salt_sodium_inconsistent_is_flagged():
    # salt=0.1 ama sodium=100mg -> beklenen salt = 0.1*2.5 = 0.25, %15'ten fazla sapma
    record = _record(salt_g=0.1, sodium_mg=100)
    issues = validate_record(record)
    assert any(i.rule == "salt_sodium_inconsistent" for i in issues)


def test_salt_sodium_consistent_not_flagged():
    # sodium=400mg -> beklenen salt = 0.4*2.5 = 1.0
    record = _record(salt_g=1.0, sodium_mg=400)
    issues = validate_record(record)
    assert not any(i.rule == "salt_sodium_inconsistent" for i in issues)


def test_find_duplicate_product_ids():
    records = [_record("a", energy_kcal=1), _record("b", energy_kcal=1), _record("a", energy_kcal=2)]
    issues = find_duplicate_product_ids(records)
    assert len(issues) == 1
    assert issues[0].product_id == "a"


def test_filter_valid_records_rejects_missing_and_duplicates():
    good = _record("good", energy_kcal=100, fat_g=1, carbohydrate_g=1, protein_g=1)
    missing = _record("missing")  # rejected: missing energy+macros
    duplicate = _record("good", energy_kcal=100, fat_g=1, carbohydrate_g=1, protein_g=1)

    valid, issues = filter_valid_records([good, missing, duplicate])

    assert len(valid) == 1
    assert valid[0].product_id == "good"
    rejected_rules = {i.rule for i in issues}
    assert "missing_energy_and_macros" in rejected_rules
    assert "duplicate_product_id" in rejected_rules


def test_filter_valid_records_keeps_flagged_but_not_rejecting():
    # salt/sodium tutarsiz ama enerji/makro tamam -> reddedilmemeli, sadece flag
    record = _record("flagged", energy_kcal=100, fat_g=1, carbohydrate_g=1, protein_g=1, salt_g=0.1, sodium_mg=100)
    valid, issues = filter_valid_records([record])

    assert len(valid) == 1
    assert any(i.rule == "salt_sodium_inconsistent" for i in issues)


def test_safe_build_record_returns_schema_error_on_negative_value():
    record, issue = safe_build_record(
        {"product_id": "bad", "nutrition": {"energy_kcal": -5}}
    )
    assert record is None
    assert issue is not None
    assert issue.rule == "schema_error"


def test_safe_build_record_returns_valid_record():
    record, issue = safe_build_record({"product_id": "ok", "nutrition": {"energy_kcal": 100}})
    assert issue is None
    assert record is not None
    assert record.product_id == "ok"
