"""Icindekiler metninden fuzzy/sinonim eslestirme ile alerjen tespiti (rapidfuzz).

Oneri formu 2.1: "Sik rastlanan alerjenler (laktoz, gluten, findik, soya vb.) icin anahtar
kelime ve sinonim sozlukleri olusturulacak, boylece OCR ile cikarilan ham metin icinden bu
bilesenlerin varligi kesin olarak belirlenerek sonraki analiz katmanlarina uyari bayragi
olarak iletilecektir."

Not: Kisa sinonimler (<=3 harf, ornegin "un") icin fuzzy eslestirme YANLIS POZITIF riski
tasir (kisa bir dizi, rastgele metinde bile yuksek benzerlik skoru bulabilir) - bu yuzden
kisa sinonimler tam kelime eslesmesi ister, sadece uzun sinonimler icin OCR hatalarina karsi
toleransli fuzzy eslestirme (rapidfuzz.fuzz.ratio) kullanilir.
"""

from __future__ import annotations

import re

from rapidfuzz import fuzz

from src.common.schema import Allergen

DEFAULT_FUZZY_THRESHOLD = 80  # tek karakterlik OCR hatalarina (~83 benzerlik) tolerans icin
SHORT_SYNONYM_MAX_LEN = 3  # bu uzunluk ve alti icin tam kelime eslesmesi zorunlu

ALLERGEN_SYNONYMS: dict[Allergen, list[str]] = {
    Allergen.LAKTOZ: [
        "süt", "sut", "laktoz", "milk", "lactose", "peynir", "yogurt", "yoğurt", "krema", "tereyağı", "tereyagi",
    ],
    Allergen.GLUTEN: [
        "gluten", "buğday", "bugday", "arpa", "çavdar", "cavdar", "yulaf", "wheat", "barley", "rye", "oat",
    ],
    Allergen.FINDIK: [
        "fındık", "findik", "ceviz", "badem", "kaju", "fıstık", "fistik", "hazelnut", "walnut", "almond", "pistachio",
    ],
    Allergen.SOYA: ["soya", "soy", "soybean"],
    Allergen.YUMURTA: ["yumurta", "egg", "albumin"],
    Allergen.BALIK: ["balık", "balik", "fish", "hamsi", "somon", "uskumru"],
}


def _tokenize(text: str) -> list[str]:
    """Turkce karakterleri koruyarak kelime tokenlerine ayirir."""
    return re.findall(r"[a-zA-ZçÇğĞıİöÖşŞüÜ]+", text.lower())


def detect_allergens(
    ingredients_text: str, threshold: int = DEFAULT_FUZZY_THRESHOLD
) -> list[Allergen]:
    """Icindekiler metninden alerjenleri tespit eder (sinonim listesi + fuzzy tolerans)."""
    if not ingredients_text:
        return []

    tokens = _tokenize(ingredients_text)
    detected: list[Allergen] = []

    for allergen, synonyms in ALLERGEN_SYNONYMS.items():
        if _matches_any_synonym(synonyms, tokens, threshold):
            detected.append(allergen)

    return detected


def _matches_any_synonym(synonyms: list[str], tokens: list[str], threshold: int) -> bool:
    for synonym in synonyms:
        synonym_lower = synonym.lower()
        if len(synonym_lower) <= SHORT_SYNONYM_MAX_LEN:
            if synonym_lower in tokens:
                return True
        else:
            for token in tokens:
                if fuzz.ratio(synonym_lower, token) >= threshold:
                    return True
    return False
