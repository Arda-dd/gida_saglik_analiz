"""Esik tabanli kural motoru: besin degerlerini TGK/Nutri-Score esikleriyle kiyaslayip
RAG modulune (Faz 4) aktarilacak baglamsal risk bayraklari uretir.

Oneri formu 2.3: "100 g urundeki toplam seker miktarinin belirli bir gramajin uzerindeki
tespiti, RAG modulune 'yuksek seker riski' baglamsal bilgisini iletecektir."

Esik degerleri data/knowledge_base/thresholds.json'da kaynak/dogrulama notlariyla birlikte
belgelenmistir (bkz. src/data/kb_builder.py) - bu modul config.yaml'daki sayisal degerleri
kullanir (tek dogruluk kaynagi, Faz 0'dan beri gecerli kural).
"""

from __future__ import annotations

from src.common.config import get_config
from src.common.schema import NutritionFacts

# RAG'e iletilecek bayrak -> insan-okunabilir Turkce aciklama (kaynak referansiyla birlikte).
RISK_DESCRIPTIONS: dict[str, str] = {
    "yuksek_seker": "Yuksek seker icerir (100g'da esigin uzerinde).",
    "yuksek_tuz": "Yuksek tuz icerir (100g'da esigin uzerinde).",
    "yuksek_doymus_yag": "Yuksek doymus yag icerir (100g'da esigin uzerinde).",
    "yuksek_sodyum": "Yuksek sodyum icerir (100g'da esigin uzerinde).",
}


def assess_sugar_risk(nutrition: NutritionFacts) -> bool:
    if nutrition.sugar_g is None:
        return False
    threshold = get_config()["thresholds"]["sugar_high_g_per_100g"]
    return nutrition.sugar_g > threshold


def assess_salt_risk(nutrition: NutritionFacts) -> bool:
    if nutrition.salt_g is None:
        return False
    threshold = get_config()["thresholds"]["salt_high_g_per_100g"]
    return nutrition.salt_g > threshold


def assess_saturated_fat_risk(nutrition: NutritionFacts) -> bool:
    if nutrition.saturated_fat_g is None:
        return False
    threshold = get_config()["thresholds"]["saturated_fat_high_g_per_100g"]
    return nutrition.saturated_fat_g > threshold


def assess_sodium_risk(nutrition: NutritionFacts) -> bool:
    if nutrition.sodium_mg is None:
        return False
    threshold = get_config()["thresholds"]["sodium_high_mg_per_100g"]
    return nutrition.sodium_mg > threshold


def assess_risks(nutrition: NutritionFacts) -> list[str]:
    """Tum risk kontrollerini calistirir, tetiklenen bayrak isimlerini (RAG icin) doner."""
    flags = []
    if assess_sugar_risk(nutrition):
        flags.append("yuksek_seker")
    if assess_salt_risk(nutrition):
        flags.append("yuksek_tuz")
    if assess_saturated_fat_risk(nutrition):
        flags.append("yuksek_doymus_yag")
    if assess_sodium_risk(nutrition):
        flags.append("yuksek_sodyum")
    return flags


def describe_risks(flags: list[str]) -> list[str]:
    """Bayrak isimlerini insan-okunabilir Turkce aciklamalara cevirir."""
    return [RISK_DESCRIPTIONS[flag] for flag in flags]
