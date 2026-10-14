"""Icindekiler metninden fuzzy/sinonim eslestirme ile alerjen tespiti (rapidfuzz).

Oneri formu 2.1: "Sik rastlanan alerjenler (laktoz, gluten, findik, soya vb.) icin anahtar
kelime ve sinonim sozlukleri olusturulacak, boylece OCR ile cikarilan ham metin icinden bu
bilesenlerin varligi kesin olarak belirlenerek sonraki analiz katmanlarina uyari bayragi
olarak iletilecektir."

Not: Kisa sinonimler (<=4 harf, ornegin "lait", "soja") icin fuzzy eslestirme YANLIS POZITIF
riski tasir ("uova"/"uva", "noix"/"noir") - bu yuzden kisa sinonimler tam kelime eslesmesi
ister, sadece uzun sinonimler icin OCR hatalarina karsi toleransli fuzzy eslestirme
(rapidfuzz.fuzz.ratio) kullanilir.

2026-10-14 genisletmesi (bkz. docs/allergen_evaluation_report.json): ilk olcumde gercek
icindekiler gorsellerinde "icerir" alerjenlerinin yalnizca %25'i yakalaniyordu - sozluk sadece
Turkce/Ingilizceydi ve kelime ayirici "BLÉ" kelimesini "bl", "ŒUF"u "uf" diye boluyordu.
Eklenenler: Unicode kelime ayirma + aksan normalizasyonu (œ->oe, é->e), AB etiketlerinde sik
gorulen diller (FR/DE/NL/IT/ES), cok kelimeli ifadeler ("fruits a coque") ve bilinen yanlis
pozitiflerin disarida birakilmasi (hindistan cevizi, kakao yagi, "sans gluten").
"""

from __future__ import annotations

import re
import unicodedata

from rapidfuzz import fuzz

from src.common.schema import Allergen

DEFAULT_FUZZY_THRESHOLD = 80  # tek karakterlik OCR hatalarina (~83 benzerlik) tolerans icin
SHORT_SYNONYM_MAX_LEN = 4  # bu uzunluk ve alti icin tam kelime eslesmesi zorunlu
# 5 harflilerde tek harf farki %80 benzerlik demek ve gercek yanlis pozitifler uretti
# ("boter"~"boite", "cajou"~"ajout") - bunlarda fuzzy yok, sadece tam/icinde gecme.
FUZZY_MIN_SYNONYM_LEN = 6

ALLERGEN_SYNONYMS: dict[Allergen, list[str]] = {
    Allergen.LAKTOZ: [
        "süt", "sut", "laktoz", "peynir", "yoğurt", "yogurt", "krema", "tereyağı", "tereyagi", "kaymak",
        "peynir altı suyu",
        "milk", "lactose", "whey", "cheese", "cream", "butter", "casein",
        "lait", "lactosérum", "crème", "beurre", "fromage", "babeurre", "caséine",
        "milch", "sahne", "käse", "molke", "melk", "kaas", "boter",
        "latte", "formaggio", "burro", "panna", "leche", "queso", "mantequilla",
    ],
    Allergen.GLUTEN: [
        "gluten", "buğday", "bugday", "arpa", "çavdar", "cavdar", "yulaf",
        "wheat", "barley", "rye", "oat", "oats", "spelt",
        "blé", "froment", "orge", "seigle", "avoine", "épeautre", "kamut",
        "weizen", "gerste", "roggen", "hafer", "dinkel", "tarwe", "gerst", "rogge", "haver",
        "glutine", "frumento", "orzo", "segale", "avena", "farro", "trigo", "cebada", "centeno",
    ],
    Allergen.FINDIK: [
        "fındık", "findik", "ceviz", "badem", "kaju", "fıstık", "fistik",
        "hazelnut", "walnut", "almond", "pistachio", "cashew", "pecan", "macadamia", "nuts", "tree nuts",
        "noisette", "noix", "amande", "pistache", "cajou", "fruits à coque",
        "haselnuss", "walnuss", "mandel", "mandeln", "schalenfrüchte", "hazelnoot", "walnoot", "amandel",
        "nocciola", "nocciole", "mandorla", "mandorle", "noci", "avellana", "almendra", "nueces",
    ],
    Allergen.SOYA: ["soya", "soy", "soybean", "soja", "soia", "sojabohnen"],
    Allergen.YUMURTA: [
        "yumurta", "egg", "eggs", "albumin", "oeuf", "oeufs", "albumine",
        "eier", "eigelb", "vollei", "eieren", "uovo", "uova", "huevo", "huevos",
    ],
    Allergen.BALIK: [
        "balık", "balik", "hamsi", "somon", "uskumru",
        "fish", "tuna", "salmon", "anchovy", "cod", "herring",
        "poisson", "poissons", "anchois", "thon", "saumon", "sardine", "sardines", "cabillaud",
        "maquereau", "colin", "merlu", "hareng",
        "fisch", "lachs", "thunfisch", "pesce", "pescado",
    ],
}

# Alerjen kelimesi icerse de alerjen OLMAYAN ifadeler: eslestirmeden once metinden cikarilir.
EXCLUDED_PHRASES = [
    "hindistan cevizi", "hindistancevizi", "noix de coco", "lait de coco", "crème de coco",
    "coconut", "noix de muscade", "muscade", "nutmeg",
    "beurre de cacao", "cocoa butter", "kakao yağı", "kakaobutter", "burro di cacao",
    "beurre de karité", "shea butter", "peanut butter", "butternut",
    "glutensiz", "gluten free", "gluten-free", "sans gluten", "glutenfrei", "senza glutine",
    "sin gluten", "sans lactose", "lactose free", "laktozsuz",
    # "ferments lactiques" bakteri kulturudur; "froment" (bugday) ile fuzzy karisiyordu.
    "ferments", "ferment",
]


def _normalize(text: str) -> str:
    """Kucuk harf + aksan/ligatur temizligi: "ŒUF" -> "oeuf", "BLÉ" -> "ble", "Süt" -> "sut"."""
    text = text.lower().replace("œ", "oe").replace("æ", "ae").replace("ß", "ss")
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def _tokenize(text: str) -> list[str]:
    """Normalize edilmis metni Unicode harf dizilerine ayirir (rakam/noktalama ayirici)."""
    return re.findall(r"[^\W\d_]+", text)


_NORMALIZED_SYNONYMS = {
    allergen: [_normalize(s) for s in synonyms] for allergen, synonyms in ALLERGEN_SYNONYMS.items()
}
_NORMALIZED_EXCLUSIONS = sorted((_normalize(p) for p in EXCLUDED_PHRASES), key=len, reverse=True)


def detect_allergens(
    ingredients_text: str, threshold: int = DEFAULT_FUZZY_THRESHOLD
) -> list[Allergen]:
    """Icindekiler metninden alerjenleri tespit eder (sinonim listesi + fuzzy tolerans).

    "Icerebilir/peut contenir/may contain" cumlelerindeki alerjenler de dondurulur - alerji
    acisindan guvenli taraf budur (eser miktar da tehlikelidir)."""
    if not ingredients_text:
        return []

    normalized = " ".join(_normalize(ingredients_text).split())
    for phrase in _NORMALIZED_EXCLUSIONS:
        normalized = re.sub(rf"\b{re.escape(phrase)}\b", " ", normalized)
    tokens = _tokenize(normalized)

    detected: list[Allergen] = []
    for allergen, synonyms in _NORMALIZED_SYNONYMS.items():
        if _matches_any_synonym(synonyms, tokens, normalized, threshold):
            detected.append(allergen)
    return detected


def _matches_any_synonym(synonyms: list[str], tokens: list[str], normalized_text: str, threshold: int) -> bool:
    for synonym in synonyms:
        if " " in synonym:
            if re.search(rf"\b{re.escape(synonym)}\b", normalized_text):
                return True
        elif len(synonym) <= SHORT_SYNONYM_MAX_LEN:
            if synonym in tokens:
                return True
        else:
            for token in tokens:
                # Birlesik kelime / Turkce ek: "weizenmehl", "vollmilchpulver", "cevizli",
                # "yumurtali" - fuzzy oran bunlarda esigin altinda kaliyordu. Kisa sinonimlere
                # uygulanmaz ("lait" -> "laitue" (marul) yanlis pozitifi).
                if synonym in token or (
                    len(synonym) >= FUZZY_MIN_SYNONYM_LEN and fuzz.ratio(synonym, token) >= threshold
                ):
                    return True
    return False
