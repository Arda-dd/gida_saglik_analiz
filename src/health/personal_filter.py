"""Kisisel Profil Filtreleme Katmani (Faz 5).

Oneri formu 2.5: retrieval ile generation arasinda, genel risk degerlendirmesini kullanicinin
kronik durumu ve alerjenlerine gore ozellestiren katman ("diyabet hastalari icin onerilmez" vb.).

Tum sayisal/mantiksal kararlar kural tabanli Python'dadir (src/ocr/risk_engine.py'deki genel
esiklerle ayni felsefe) - bu katman LLM'e ihtiyac duymaz, sadece risk bayraklarini ve alerjen
listelerini profille kiyaslar.
"""

from __future__ import annotations

from src.common.schema import Allergen, NutritionFacts, UserObjective
from src.health.profile import ChronicCondition, HealthProfile

# Kronik durum -> (ilgili risk bayragi -> kullaniciya ozel uyari) eslemesi.
# Kaynak: oneri formu 2.5 ornegi ("diyabet hastalari icin onerilmez") + genel beslenme bilgisi
# (WHO/TGK esikleriyle iliskili risk bayraklarinin hangi kronik durumlarla ilgili oldugu).
CONDITION_RISK_WARNINGS: dict[ChronicCondition, dict[str, str]] = {
    ChronicCondition.DIYABET: {
        "yuksek_seker": (
            "Diyabet hastalari icin onerilmez (yuksek seker icerigi kan sekerini hizla "
            "yukseltebilir)."
        ),
    },
    ChronicCondition.HIPERTANSIYON: {
        "yuksek_tuz": (
            "Hipertansiyon hastalari icin onerilmez (yuksek tuz kan basincini artirabilir)."
        ),
        "yuksek_sodyum": (
            "Hipertansiyon hastalari icin onerilmez (yuksek sodyum kan basincini artirabilir)."
        ),
    },
    ChronicCondition.BOBREK_HASTALIGI: {
        "yuksek_sodyum": (
            "Bobrek hastalari icin dikkatli tuketilmeli (yuksek sodyum bobrekleri zorlayabilir)."
        ),
        "yuksek_tuz": (
            "Bobrek hastalari icin dikkatli tuketilmeli (yuksek tuz bobrekleri zorlayabilir)."
        ),
    },
    ChronicCondition.KALP_HASTALIGI: {
        "yuksek_doymus_yag": (
            "Kalp hastalari icin onerilmez (yuksek doymus yag kolesterolu olumsuz "
            "etkileyebilir)."
        ),
    },
}

# Bir porsiyonun (100g bazinda normalize edilmis urun icin) gunluk kalori hedefinin makul
# bir payini asmasi durumunda diyet uyum skorundan dusulur.
REASONABLE_SERVING_CALORIE_SHARE = 0.25
CALORIE_PENALTY_CAP = 50.0
CONDITION_PENALTY_PER_WARNING = 25.0
CONDITION_PENALTY_CAP = 50.0


def condition_based_warnings(
    risk_flags: list[str], conditions: list[ChronicCondition]
) -> list[str]:
    """Aktif risk bayraklarindan, kullanicinin kronik durumlariyla iliskili olanlari secip
    kisisellestirilmis uyari metinlerine cevirir."""
    warnings: list[str] = []
    for condition in conditions:
        rules = CONDITION_RISK_WARNINGS.get(condition, {})
        for flag in risk_flags:
            if flag in rules:
                warnings.append(rules[flag])
    return warnings


def check_allergen_conflict(
    product_allergens: list[Allergen], profile_allergens: list[Allergen]
) -> list[Allergen]:
    """Urunun icerdigi alerjenlerden, kullanicinin profilinde de bulunanlari doner (celisme listesi)."""
    profile_set = set(profile_allergens)
    return [a for a in product_allergens if a in profile_set]


def compute_diet_compliance_score(
    nutrition: NutritionFacts, profile: HealthProfile, risk_flags: list[str]
) -> float:
    """0-100 arasi diyet uyum skoru (100 = tam uyumlu).

    Iki bagimsiz ceza bileseninden olusur:
    1. Kalori payi cezasi: urunun (100g bazinda) enerjisi, kullanicinin gunluk kalori hedefinin
       makul bir tek-porsiyon payini (%25) asiyorsa, asim oraniyla orantili ceza (en fazla 50 puan).
       Not: ALERJI_TAKIBI hedefinde kalori cezasi uygulanmaz.
    2. Kronik durum cezasi: kullanicinin durumuyla celisen her aktif risk bayragi icin sabit
       ceza (25 puan/uyari, en fazla 50 puan).
    """
    score = 100.0

    # 1. Kalori Payi Cezasi (Alerji takibi amacinda kalori hedefleri goz ardi edilir)
    if profile.objective != UserObjective.ALERJI_TAKIBI:
        if profile.daily_calorie_target_kcal and nutrition.energy_kcal:
            share = nutrition.energy_kcal / profile.daily_calorie_target_kcal
            if share > REASONABLE_SERVING_CALORIE_SHARE:
                excess_ratio = (share - REASONABLE_SERVING_CALORIE_SHARE) / REASONABLE_SERVING_CALORIE_SHARE
                # Kilo verme hedefinde kalori cezasi 1.5 kat daha agirdir
                multiplier = 1.5 if profile.objective == UserObjective.KILO_VERME else 1.0
                penalty = excess_ratio * CALORIE_PENALTY_CAP * multiplier
                score -= min(CALORIE_PENALTY_CAP * multiplier, penalty)

    # 2. Kronik Durum Cezalari
    condition_warnings = condition_based_warnings(risk_flags, profile.chronic_conditions)
    score -= min(CONDITION_PENALTY_CAP, len(condition_warnings) * CONDITION_PENALTY_PER_WARNING)

    # 3. Hedefe Ozel Ek Puanlama Kurallari
    if profile.objective == UserObjective.KILO_VERME:
        # Seker uyarisi varsa kilo verme skorundan ekstra 15 puan dusulur
        if "yuksek_seker" in risk_flags:
            score -= 15.0
    elif profile.objective == UserObjective.PROTEIN_AGIRLIKLI:
        # Protein miktari >= 10g ise 15 puan diyet uyum bonusu verilir (maks 100)
        if nutrition.protein_g is not None and nutrition.protein_g >= 10.0:
            score += 15.0

    return max(0.0, min(100.0, round(score, 1)))
