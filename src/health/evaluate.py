"""Faz 5 degerlendirme scripti: Profile Consistency Score + Recommendation Relevance.

Oneri formu 2.5 metrikleri (form takvimi hedefi: ikisi de >=%90). Bu katman tamamen kural
tabanli oldugundan (LLM'e bagimli DEGIL - bkz. src/health/personal_filter.py, src/health/
recommend.py), degerlendirme API cagrisi gerektirmez: elle etiketlenmis sentetik senaryolarla
(her senaryo icin BEKLENEN davranis onceden tanimlanir) gercek davranis kiyaslanir.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.common.schema import Allergen, NutritionFacts, ProductCategory, ProductRecord
from src.health.profile import ChronicCondition, HealthProfile
from src.health.recommend import build_health_assessment, recommend_alternatives

REPORT_PATH = Path(__file__).resolve().parents[2] / "docs" / "health_evaluation_report.json"

# Bilinen tum kosul-uyarisi anahtar kelimeleri - "beklenmeyen uyari" (yanlis kosula yanlis
# bayrak eslenmesi) kontrolu icin kullanilir.
_ALL_CONDITION_KEYWORDS = ["Diyabet", "Hipertansiyon", "Bobrek", "Kalp"]


@dataclass
class ProfileConsistencyCase:
    name: str
    nutrition: NutritionFacts
    product_allergens: list[Allergen]
    profile: HealthProfile
    expect_allergen_warning: bool
    expected_condition_keywords: list[str]


PROFILE_CONSISTENCY_CASES: list[ProfileConsistencyCase] = [
    ProfileConsistencyCase(
        name="diyabetik_yuksek_seker_urunu",
        nutrition=NutritionFacts(sugar_g=35.0),
        product_allergens=[],
        profile=HealthProfile(profile_id="c1", chronic_conditions=[ChronicCondition.DIYABET]),
        expect_allergen_warning=False,
        expected_condition_keywords=["Diyabet"],
    ),
    ProfileConsistencyCase(
        name="saglikli_profil_ayni_urun_diyabet_uyarisi_olmamali",
        nutrition=NutritionFacts(sugar_g=35.0),
        product_allergens=[],
        profile=HealthProfile(profile_id="c2"),
        expect_allergen_warning=False,
        expected_condition_keywords=[],
    ),
    ProfileConsistencyCase(
        name="hipertansif_yuksek_tuz_urunu",
        nutrition=NutritionFacts(salt_g=2.5),
        product_allergens=[],
        profile=HealthProfile(profile_id="c3", chronic_conditions=[ChronicCondition.HIPERTANSIYON]),
        expect_allergen_warning=False,
        expected_condition_keywords=["Hipertansiyon"],
    ),
    ProfileConsistencyCase(
        name="bobrek_hastasi_yuksek_sodyum_urunu",
        nutrition=NutritionFacts(sodium_mg=1200),
        product_allergens=[],
        profile=HealthProfile(profile_id="c4", chronic_conditions=[ChronicCondition.BOBREK_HASTALIGI]),
        expect_allergen_warning=False,
        expected_condition_keywords=["Bobrek"],
    ),
    ProfileConsistencyCase(
        name="kalp_hastasi_yuksek_doymus_yag_urunu",
        nutrition=NutritionFacts(saturated_fat_g=12.0),
        product_allergens=[],
        profile=HealthProfile(profile_id="c5", chronic_conditions=[ChronicCondition.KALP_HASTALIGI]),
        expect_allergen_warning=False,
        expected_condition_keywords=["Kalp"],
    ),
    ProfileConsistencyCase(
        name="findik_alerjisi_findik_iceren_urun",
        nutrition=NutritionFacts(sugar_g=2.0),
        product_allergens=[Allergen.FINDIK],
        profile=HealthProfile(profile_id="c6", allergens=[Allergen.FINDIK]),
        expect_allergen_warning=True,
        expected_condition_keywords=[],
    ),
    ProfileConsistencyCase(
        name="findik_alerjisi_findik_icermeyen_urun_uyari_olmamali",
        nutrition=NutritionFacts(sugar_g=2.0),
        product_allergens=[],
        profile=HealthProfile(profile_id="c7", allergens=[Allergen.FINDIK]),
        expect_allergen_warning=False,
        expected_condition_keywords=[],
    ),
    ProfileConsistencyCase(
        name="coklu_kosul_dogru_ve_ayrik_uyarilar",
        nutrition=NutritionFacts(sugar_g=30.0, salt_g=2.0),
        product_allergens=[],
        profile=HealthProfile(
            profile_id="c8",
            chronic_conditions=[ChronicCondition.DIYABET, ChronicCondition.HIPERTANSIYON],
        ),
        expect_allergen_warning=False,
        expected_condition_keywords=["Diyabet", "Hipertansiyon"],
    ),
    ProfileConsistencyCase(
        name="dusuk_riskli_urun_hicbir_kosul_uyarisi_uretmemeli",
        nutrition=NutritionFacts(sugar_g=2.0, salt_g=0.2, saturated_fat_g=1.0),
        product_allergens=[],
        profile=HealthProfile(
            profile_id="c9",
            chronic_conditions=[ChronicCondition.DIYABET, ChronicCondition.KALP_HASTALIGI],
        ),
        expect_allergen_warning=False,
        expected_condition_keywords=[],
    ),
]


def evaluate_profile_consistency(
    cases: list[ProfileConsistencyCase] = PROFILE_CONSISTENCY_CASES,
) -> tuple[list[dict], float]:
    """Ayni degerlendirme mantiginin FARKLI profiller icin DOGRU sekilde farklilastigini olcer."""
    rows = []
    for case in cases:
        assessment = build_health_assessment(case.nutrition, case.product_allergens, case.profile)
        combined_messages = " ".join(assessment.health_risk_messages)

        keywords_ok = all(kw in combined_messages for kw in case.expected_condition_keywords)
        no_unexpected_keywords = all(
            known_kw in case.expected_condition_keywords or known_kw not in combined_messages
            for known_kw in _ALL_CONDITION_KEYWORDS
        )
        allergen_ok = assessment.allergen_warning == case.expect_allergen_warning
        passed = keywords_ok and no_unexpected_keywords and allergen_ok

        rows.append(
            {
                "name": case.name,
                "passed": passed,
                "health_risk_messages": assessment.health_risk_messages,
                "allergen_warning": assessment.allergen_warning,
            }
        )

    score = sum(1 for r in rows if r["passed"]) / len(rows) if rows else 0.0
    return rows, score


def _make_product(product_id: str, category: ProductCategory, allergens=None, **nutrition_kwargs) -> ProductRecord:
    return ProductRecord(
        product_id=product_id,
        category=category,
        nutrition=NutritionFacts(**nutrition_kwargs),
        allergens=allergens or [],
        source="synthetic_eval",
    )


@dataclass
class RecommendationCase:
    name: str
    current: ProductRecord
    candidates: list[ProductRecord]
    profile: HealthProfile
    expected_recommended_ids: set[str]


RECOMMENDATION_CASES: list[RecommendationCase] = [
    RecommendationCase(
        name="findik_alerjisi_guvenli_alternatif_bulunmali_digerleri_elenmeli",
        current=_make_product("cur1", ProductCategory.ATISTIRMALIK, sugar_g=40.0, salt_g=2.0),
        candidates=[
            _make_product("good1", ProductCategory.ATISTIRMALIK, sugar_g=3.0),
            _make_product(
                "allergenic1", ProductCategory.ATISTIRMALIK, allergens=[Allergen.FINDIK], sugar_g=1.0
            ),
            _make_product("wrong_cat1", ProductCategory.ICECEK, sugar_g=1.0),
            # current ile AYNI iki risk bayragini tasir (yuksek_seker + yuksek_tuz) - risk
            # SAYISI esit oldugundan "daha iyi" sayilmamali, elenmeli.
            _make_product("risky1", ProductCategory.ATISTIRMALIK, sugar_g=35.0, salt_g=2.0),
        ],
        profile=HealthProfile(profile_id="r1", allergens=[Allergen.FINDIK]),
        expected_recommended_ids={"good1"},
    ),
    RecommendationCase(
        name="birden_fazla_guvenli_alternatif_risk_sirasina_gore_siralanmali",
        current=_make_product(
            "cur2", ProductCategory.SUT_URUNU, sugar_g=30.0, salt_g=2.0, saturated_fat_g=8.0
        ),
        candidates=[
            _make_product("medium2", ProductCategory.SUT_URUNU, sugar_g=1.0, salt_g=2.0),
            _make_product("best2", ProductCategory.SUT_URUNU, sugar_g=1.0),
        ],
        profile=HealthProfile(profile_id="r2"),
        expected_recommended_ids={"best2", "medium2"},
    ),
    RecommendationCase(
        name="daha_dusuk_riskli_alternatif_yoksa_bos_liste_donmeli",
        current=_make_product("cur3", ProductCategory.ICECEK, sugar_g=2.0),
        candidates=[
            _make_product("risky3", ProductCategory.ICECEK, sugar_g=25.0),
        ],
        profile=HealthProfile(profile_id="r3"),
        expected_recommended_ids=set(),
    ),
    RecommendationCase(
        name="hicbir_aday_yoksa_bos_liste_donmeli",
        current=_make_product("cur4", ProductCategory.KONSERVE, sugar_g=30.0),
        candidates=[],
        profile=HealthProfile(profile_id="r4"),
        expected_recommended_ids=set(),
    ),
]


def evaluate_recommendation_relevance(
    cases: list[RecommendationCase] = RECOMMENDATION_CASES,
) -> tuple[list[dict], float]:
    """Onerilen alternatiflerin gercekten guvenli (alerjen celismesi yok), ayni kategoride
    ve daha dusuk riskli oldugunu - sentetik ama distractor icerikli senaryolarla - dogrular."""
    rows = []
    for case in cases:
        result = recommend_alternatives(case.current, case.candidates, case.profile)
        actual_ids = {p.product_id for p in result}
        passed = actual_ids == case.expected_recommended_ids
        rows.append({"name": case.name, "passed": passed, "actual_ids": sorted(actual_ids)})

    score = sum(1 for r in rows if r["passed"]) / len(rows) if rows else 0.0
    return rows, score


def main() -> None:
    consistency_rows, consistency_score = evaluate_profile_consistency()
    print(f"Profile Consistency Score: %{consistency_score * 100:.1f}")
    for row in consistency_rows:
        status = "OK" if row["passed"] else "FAIL"
        print(f"  [{status}] {row['name']}")

    recommendation_rows, recommendation_score = evaluate_recommendation_relevance()
    print(f"\nRecommendation Relevance: %{recommendation_score * 100:.1f}")
    for row in recommendation_rows:
        status = "OK" if row["passed"] else "FAIL"
        print(f"  [{status}] {row['name']} -> {row['actual_ids']}")

    report = {
        "profile_consistency": {"score": consistency_score, "cases": consistency_rows},
        "recommendation_relevance": {"score": recommendation_score, "cases": recommendation_rows},
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRapor kaydedildi: {REPORT_PATH}")


if __name__ == "__main__":
    main()
