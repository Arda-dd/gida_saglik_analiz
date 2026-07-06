"""Toplanan ProductRecord kayitlarinin veri butunlugu dogrulamasi.

Formdaki taahhut: "Veri butunlugu, eksik veya uc degerlerin belirlenmesi amaciyla
dogrulama kurallariyla denetlenecek ve hatali kayitlar model egitimine dahil
edilmeyecektir." (oneri formu, 2.1)
"""

from __future__ import annotations

from typing import NamedTuple

from pydantic import ValidationError

from src.common.config import get_config
from src.common.schema import ProductRecord


class ValidationIssue(NamedTuple):
    product_id: str
    rule: str
    message: str


def _energy_and_macros_missing(nutrition) -> bool:
    energy_missing = nutrition.energy_kcal is None and nutrition.energy_kj is None
    macros_missing = (
        nutrition.fat_g is None
        and nutrition.carbohydrate_g is None
        and nutrition.protein_g is None
    )
    return energy_missing and macros_missing


def _energy_implausible(nutrition) -> bool:
    max_kcal = get_config()["thresholds"]["energy_kcal_max_plausible_per_100g"]
    return nutrition.energy_kcal is not None and nutrition.energy_kcal > max_kcal


def _macro_energy_mismatch(nutrition, tolerance_pct: float = 20.0) -> bool:
    """fat*9 + carb*4 + protein*4 ile energy_kcal arasinda buyuk sapma var mi (flag amacli)."""
    if nutrition.energy_kcal is None:
        return False
    if nutrition.fat_g is None or nutrition.carbohydrate_g is None or nutrition.protein_g is None:
        return False

    estimated = nutrition.fat_g * 9 + nutrition.carbohydrate_g * 4 + nutrition.protein_g * 4
    if estimated == 0:
        return nutrition.energy_kcal > 0

    deviation_pct = abs(estimated - nutrition.energy_kcal) / estimated * 100
    return deviation_pct > tolerance_pct


def _salt_sodium_inconsistent(nutrition, tolerance_pct: float = 15.0) -> bool:
    """salt_g ve sodium_mg birlikte doluysa, Tuz(g) = Sodyum(g) x 2.5 tutarliligini kontrol eder."""
    if nutrition.salt_g is None or nutrition.sodium_mg is None:
        return False

    expected_salt_g = (nutrition.sodium_mg / 1000) * 2.5
    if expected_salt_g == 0:
        return nutrition.salt_g != 0

    deviation_pct = abs(expected_salt_g - nutrition.salt_g) / expected_salt_g * 100
    return deviation_pct > tolerance_pct


def validate_record(record: ProductRecord) -> list[ValidationIssue]:
    """Tek bir kaydi kontrol eder; reddi gerektiren ve sadece flag'lenen sorunlari birlikte doner.

    Cagiran kod, 'rule' alanina bakarak hangi sorunlarin reddi gerektirdigine karar verir
    (bkz. filter_valid_records - REJECT_RULES kumesini kullanir).
    """
    issues: list[ValidationIssue] = []
    n = record.nutrition

    if _energy_and_macros_missing(n):
        issues.append(
            ValidationIssue(record.product_id, "missing_energy_and_macros", "Enerji ve tum makrolar eksik")
        )

    if _energy_implausible(n):
        issues.append(
            ValidationIssue(
                record.product_id,
                "implausible_energy",
                f"energy_kcal={n.energy_kcal} teorik maksimumu asiyor",
            )
        )

    if _macro_energy_mismatch(n):
        issues.append(
            ValidationIssue(
                record.product_id,
                "macro_energy_mismatch",
                "Makrolardan hesaplanan enerji ile beyan edilen enerji arasinda >%20 sapma",
            )
        )

    if _salt_sodium_inconsistent(n):
        issues.append(
            ValidationIssue(
                record.product_id,
                "salt_sodium_inconsistent",
                "salt_g ve sodium_mg, Tuz=Sodyum*2.5 formuluyle tutarsiz (>%15 sapma)",
            )
        )

    return issues


# Bu kurallardan biri varsa kayit REDDEDILIR (model egitimine dahil edilmez).
# Digerleri (macro_energy_mismatch, salt_sodium_inconsistent) sadece FLAG'lenir, elenmez.
REJECT_RULES = {"missing_energy_and_macros", "implausible_energy", "schema_error", "duplicate_product_id"}


def find_duplicate_product_ids(records: list[ProductRecord]) -> list[ValidationIssue]:
    seen: set[str] = set()
    issues: list[ValidationIssue] = []
    for record in records:
        if record.product_id in seen:
            issues.append(
                ValidationIssue(record.product_id, "duplicate_product_id", "Tekrarlanan product_id")
            )
        else:
            seen.add(record.product_id)
    return issues


def filter_valid_records(
    records: list[ProductRecord],
) -> tuple[list[ProductRecord], list[ValidationIssue]]:
    """Kayitlari dogrular; sadece REJECT_RULES'a giren sorunlu kayitlari eler.

    Duplicate product_id'lerde ilk gorulen kayit tutulur, sonrakiler elenir.
    """
    all_issues: list[ValidationIssue] = []
    seen_ids: set[str] = set()
    valid_records: list[ProductRecord] = []

    for record in records:
        record_issues = validate_record(record)

        if record.product_id in seen_ids:
            record_issues.append(
                ValidationIssue(record.product_id, "duplicate_product_id", "Tekrarlanan product_id")
            )

        all_issues.extend(record_issues)

        has_rejecting_issue = any(issue.rule in REJECT_RULES for issue in record_issues)
        if not has_rejecting_issue:
            valid_records.append(record)
            seen_ids.add(record.product_id)

    return valid_records, all_issues


def safe_build_record(record_kwargs: dict) -> tuple[ProductRecord | None, ValidationIssue | None]:
    """Pydantic ValidationError'i yakalayip 'schema_error' olarak raporlayan yardimci sarici."""
    try:
        return ProductRecord(**record_kwargs), None
    except ValidationError as e:
        product_id = record_kwargs.get("product_id", "unknown")
        return None, ValidationIssue(product_id, "schema_error", str(e))
