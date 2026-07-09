"""3 katmanli cikti + alternatif urun onerisi (Faz 5).

Oneri formu 2.5: her degerlendirme (1) Saglik Riski, (2) Diyet Uyum Skoru, (3) Alerjen Uyarisi
seklinde 3 katmanda sunulur. Ayrica ayni kategoride, kullanicinin profiliyle celismeyen ve daha
dusuk riskli alternatif urunler onerilir (Faz 2 kategori etiketi + bu katmanin kendi risk/alerjen
filtresi kullanilir - Faz 4 retriever'a bagimli DEGILDIR, saf kural tabanli siralama).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.common.schema import Allergen, NutritionFacts, ProductRecord
from src.health.personal_filter import (
    check_allergen_conflict,
    compute_diet_compliance_score,
    condition_based_warnings,
)
from src.health.profile import HealthProfile
from src.ocr.risk_engine import assess_risks, describe_risks


@dataclass
class HealthAssessment:
    health_risk_messages: list[str]
    diet_compliance_score: float
    allergen_warning: bool
    allergen_conflicts: list[Allergen] = field(default_factory=list)


def build_health_assessment(
    nutrition: NutritionFacts,
    product_allergens: list[Allergen],
    profile: HealthProfile,
) -> HealthAssessment:
    """Genel risk motoru (Faz 3) + kisisel profil filtresini (Faz 5) birlestirip 3 katmanli
    degerlendirme uretir."""
    risk_flags = assess_risks(nutrition)
    general_messages = describe_risks(risk_flags)
    condition_messages = condition_based_warnings(risk_flags, profile.chronic_conditions)

    allergen_conflicts = check_allergen_conflict(product_allergens, profile.allergens)

    return HealthAssessment(
        health_risk_messages=general_messages + condition_messages,
        diet_compliance_score=compute_diet_compliance_score(nutrition, profile, risk_flags),
        allergen_warning=bool(allergen_conflicts),
        allergen_conflicts=allergen_conflicts,
    )


def recommend_alternatives(
    current_product: ProductRecord,
    candidates: list[ProductRecord],
    profile: HealthProfile,
    max_results: int = 3,
) -> list[ProductRecord]:
    """Ayni kategoride, profil alerjenleriyle celismeyen ve mevcut urunden daha az risk
    bayragi olan alternatifleri, risk sayisina gore artan sirada doner.

    Hicbir aday daha dusuk riskli degilse bos liste doner - kullaniciyi yanlis/anlamsiz bir
    "alternatif" ile yanlis yonlendirmemek icin (form: kaynak referansli, guvenilir cikti ilkesi).
    """
    same_category = [
        p
        for p in candidates
        if p.category == current_product.category and p.product_id != current_product.product_id
    ]

    safe_candidates = [
        p for p in same_category if not check_allergen_conflict(p.allergens, profile.allergens)
    ]

    def risk_count(product: ProductRecord) -> int:
        return len(assess_risks(product.nutrition))

    current_risk_count = risk_count(current_product)
    better_candidates = [p for p in safe_candidates if risk_count(p) < current_risk_count]
    ranked = sorted(better_candidates, key=risk_count)

    return ranked[:max_results]
