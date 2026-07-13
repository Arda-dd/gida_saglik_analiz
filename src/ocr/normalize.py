"""OCR metninden besin degerlerini cikaran regex tabanli ayristirici + normalizasyon.

Oneri formu 2.3: "Enerji: 450 kcal" ifadesi "enerji" degiskeniyle eslestirilirken, "Tuz: 2.1 g"
degeri sodyum-tuz donusum fonksiyonu araciligiyla saglik degerlendirmesinde kullanilabilir
forma donusturulecektir. "100 g basina" ve "porsiyon basina" ifadeleri otomatik tespit edilip
tum degerler 100g/100mL bazinda normalize edilir.
"""

from __future__ import annotations

import re

from src.common.schema import NutritionBasis, NutritionFacts
from src.common.units import fill_missing_energy_and_salt, normalize_nutrition_to_per_100

NUMBER = r"(\d+[.,]\d+|\d+)"


def _to_float(number_str: str) -> float:
    """OCR'da hem nokta hem virgul ondalik ayirici olarak gorulebilir."""
    return float(number_str.replace(",", "."))


def _search_value(text: str, keyword_pattern: str, unit_pattern: str) -> tuple[float, str] | None:
    """keyword ... SAYI BIRIM seklindeki ilk eslesmeyi (deger, birim) olarak doner.

    Keyword ile sayi arasinda OCR kaynakli kisa artefaktlara (":", bosluk vb.) izin verilir
    ama baska bir besin degerine atlamamak icin bu bosluk 10 karakterle sinirlidir.
    """
    pattern = rf"(?:{keyword_pattern})\D{{0,10}}{NUMBER}\s*({unit_pattern})\b"
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return None
    return _to_float(match.group(1)), match.group(2).lower()


def extract_energy(text: str) -> tuple[float | None, float | None]:
    """Enerjiyi kcal ve/veya kJ olarak cikarir (ikisi de etikette bulunabilir).

    kcal/kJ birimleri beslenme etiketlerinde sadece enerji icin kullanildigindan
    (baska hicbir besin ogesi bu birimlerle olculmez), "enerji" anahtar kelimesine
    yakinlik aranmaz - bu, "Enerji 450 kcal / 1883 kJ" gibi iki degerin arasinda
    baska bir sayi (450) olan durumlarda da kJ'nin dogru yakalanmasini saglar.

    ABD/Kanada tipi etiketlerde birim hic yazilmaz ("Calories 150" - kcal zaten
    varsayilir) - bu yuzden kcal icin once birimli kalibi, bulunamazsa "calories"
    anahtar kelimesini (birimsiz) dener.
    """
    energy_kcal = None
    energy_kj = None

    kcal_match = re.search(rf"{NUMBER}\s*kcal\b", text, re.IGNORECASE)
    if kcal_match:
        energy_kcal = _to_float(kcal_match.group(1))
    else:
        calories_match = re.search(rf"calories?\D{{0,5}}{NUMBER}", text, re.IGNORECASE)
        if calories_match:
            energy_kcal = _to_float(calories_match.group(1))

    kj_match = re.search(rf"{NUMBER}\s*kj\b", text, re.IGNORECASE)
    if kj_match:
        energy_kj = _to_float(kj_match.group(1))

    return energy_kcal, energy_kj


SATURATED_FAT_KEYWORDS = r"doymu[sş]\s*ya[gğ]|saturated\s*fat|acides?\s*gras\s*satur[ée]s?"


def extract_saturated_fat(text: str) -> float | None:
    match = _search_value(text, SATURATED_FAT_KEYWORDS, r"g|gr")
    return match[0] if match else None


def extract_fat(text: str) -> float | None:
    """Toplam yag - 'doymus yag' ile karismamasi icin o ifadeyi devre disi birakir."""
    # Once doymus yag ifadesini metinden gecici olarak cikar, boylece plain 'yag' araması
    # yanlislikla doymus yag degerini yakalamaz.
    cleaned = re.sub(
        rf"(?:{SATURATED_FAT_KEYWORDS})[^0-9]{{0,10}}\d+[.,]?\d*\s*(?:g|gr)\b", "", text, flags=re.IGNORECASE
    )
    match = _search_value(cleaned, r"ya[gğ]|fat|mati[eè]res\s*grasses|lipides", r"g|gr")
    return match[0] if match else None


def extract_carbohydrate(text: str) -> float | None:
    match = _search_value(text, r"karbonhidrat|carbohydrate|glucides", r"g|gr")
    return match[0] if match else None


def extract_sugar(text: str) -> float | None:
    match = _search_value(text, r"[sş]eker|sugars?|sucres?", r"g|gr")
    return match[0] if match else None


def extract_fiber(text: str) -> float | None:
    match = _search_value(text, r"lif|fiber|fibres?(?:\s*alimentaires?)?", r"g|gr")
    return match[0] if match else None


def extract_protein(text: str) -> float | None:
    match = _search_value(text, r"protein|prot[eé]ines?", r"g|gr")
    return match[0] if match else None


def extract_salt(text: str) -> float | None:
    match = _search_value(text, r"tuz|salt|\bsel\b", r"g|gr")
    return match[0] if match else None


def extract_sodium(text: str) -> float | None:
    """Sodyum genelde mg olarak yazilir, bazen g olarak da gorulebilir (g ise mg'a cevrilir)."""
    mg_match = _search_value(text, r"sodyum|sodium", r"mg")
    if mg_match:
        return mg_match[0]

    g_match = _search_value(text, r"sodyum|sodium", r"g|gr")
    if g_match:
        return g_match[0] * 1000

    return None


def extract_nutrition_facts(text: str) -> NutritionFacts:
    """OCR metninden tum besin degerlerini cikarip NutritionFacts nesnesine doldurur."""
    energy_kcal, energy_kj = extract_energy(text)

    facts = NutritionFacts(
        energy_kcal=energy_kcal,
        energy_kj=energy_kj,
        fat_g=extract_fat(text),
        saturated_fat_g=extract_saturated_fat(text),
        carbohydrate_g=extract_carbohydrate(text),
        sugar_g=extract_sugar(text),
        fiber_g=extract_fiber(text),
        protein_g=extract_protein(text),
        salt_g=extract_salt(text),
        sodium_mg=extract_sodium(text),
    )
    return fill_missing_energy_and_salt(facts)


def detect_nutrition_basis(text: str) -> tuple[NutritionBasis, float | None]:
    """'100 g/100 ml basina' veya 'porsiyon basina (X g)' ifadesini tespit eder.

    Porsiyon tespit edilirse ve gram/mL degeri de metinde bulunursa, bu deger
    (serving_size_g, ...) olarak ikinci elemanda dondurulur - normalize_to_per_100 icin kullanilir.
    """
    if re.search(r"100\s*m[il]", text, re.IGNORECASE):
        return NutritionBasis.PER_100ML, None
    if re.search(r"100\s*g", text, re.IGNORECASE):
        return NutritionBasis.PER_100G, None

    # Not: gap kismi (".{0,40}?") sayi icerebilir (\D degil) - ABD tipi etiketlerde
    # "Serving size 1 cup (240mL)" gibi metrik olmayan bir birim (ve sayisi) araya girer;
    # gercek g/mL degeri parantez icinde daha sonra gelir. Non-greedy oldugundan regex
    # gramaj/mL'ye bitisik ILK sayi-birim eslesmesini bulur (once "1 cup" denenir, birim
    # "g/ml" olmadigi icin elenir, sonra "240mL" ile eslesir).
    serving_match = re.search(
        r"porsiyon.{0,40}?(\d+[.,]?\d*)\s*(g|ml)\b", text, re.IGNORECASE | re.DOTALL
    ) or re.search(r"serving.{0,40}?(\d+[.,]?\d*)\s*(g|ml)\b", text, re.IGNORECASE | re.DOTALL)
    if serving_match:
        return NutritionBasis.PER_SERVING, _to_float(serving_match.group(1))

    if re.search(r"porsiyon|serving|portion", text, re.IGNORECASE):
        return NutritionBasis.PER_SERVING, None

    return NutritionBasis.PER_100G, None  # varsayilan: cogu etiket 100g bazlidir


def extract_and_normalize(text: str) -> tuple[NutritionFacts, NutritionBasis]:
    """Tam pipeline: cikarim + eksik alan tamamlama + (mumkunse) 100g/100mL normalizasyonu.

    Porsiyon boyutu tespit edilemezse, ham porsiyon degerleri (normalize edilmeden) donulur
    ve basis PER_SERVING olarak isaretlenir - cagiran kod bu durumu ele almalidir.
    """
    facts = extract_nutrition_facts(text)
    basis, serving_size_g = detect_nutrition_basis(text)

    if basis == NutritionBasis.PER_SERVING and serving_size_g:
        facts = normalize_nutrition_to_per_100(facts, serving_size_g)
        basis = NutritionBasis.PER_100G

    return facts, basis
