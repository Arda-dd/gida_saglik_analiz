"""OCR metninden besin degerlerini cikaran regex tabanli ayristirici + normalizasyon.

Oneri formu 2.3: "Enerji: 450 kcal" ifadesi "enerji" degiskeniyle eslestirilirken, "Tuz: 2.1 g"
degeri sodyum-tuz donusum fonksiyonu araciligiyla saglik degerlendirmesinde kullanilabilir
forma donusturulecektir. "100 g basina" ve "porsiyon basina" ifadeleri otomatik tespit edilip
tum degerler 100g/100mL bazinda normalize edilir.

2026-10-14 genisletmesi (bkz. docs/ocr_comparison_report.md, "darbogaz analizi"): iki bulut OCR
motorunda da DOGRU okunan sayilarin ~%20'si bu katmanda kayboluyordu. Eklenenler:
- anahtar kelime ile sayi arasinda uzun bosluk (cok dilli satirlar, noktali dolgu) ve bir alt
  satira kaymis deger,
- birimin sayidan once yazildigi format ("Proteines | g | 3,0", "Kcal 97", "kJ/kcal 972,5/235"),
- AB etiketlerinde sik gorulen dillerin anahtar kelimeleri ve Arapca gram birimi ("غ"),
- once butun etiketler, sonra butun degerler gelen sutun duzeni (_realign_column_table).
"""

from __future__ import annotations

import re

from src.common.schema import NutritionBasis, NutritionFacts
from src.common.units import fill_missing_energy_and_salt, normalize_nutrition_to_per_100

NUMBER = r"(\d+[.,]\d+|\d+)"
GRAM_UNIT = r"g|gr|غ"  # "غ" = Arapca gram (Fas/Orta Dogu ithal urun etiketleri)

# Anahtar kelime ile sayi arasi: ayni satirda en fazla 80 rakam-olmayan karakter ("matieres
# grasses / vetten : 3,5 g", "Sel ........ 0,71 g", 4-5 dilli satirlar) ve istege bagli olarak
# BIR satir sonu - ama yalnizca deger alt satirin BASINDAYSA ("Fett/Rasvaa/Fedt\n55 g"). Alt
# satirda harf varsa (orn. "Glucides\ndont sucres 5 g") bu baska bir besinin satiridir, atlanmaz.
GAP = r"[^\d\n]{0,80}(?:\n[\s\-–•|:]{0,5})?"

ENERGY_KEYWORDS = r"enerji|[eé]nerg|calor|kalori|الطاقة|القيمة الحرارية"
SATURATED_FAT_KEYWORDS = (
    # Not: "�" (Unicode replacement karakteri) - OCR.space aksanli harfleri (é, è) bu karakterle
    # degistirebiliyor; "acides gras" oneki opsiyonel (etiketlerde kisa "dont saturées" formu var).
    r"doymu[sş]\s*ya[gğ]|saturated\s*fat|saturates|(?:acides?\s*gras\s*)?satur[ée�]s?"
    r"|ges[äa]ttigte\s*fetts[äa]uren|verzadigde\s*vetzuren|acidi\s*grassi\s*saturi"
    r"|grasas\s*saturadas|m[äa]ttat\s*fett|دهنية\s*مشبعة"
)
FAT_KEYWORDS = (
    r"ya[gğ]|\bfat\b|mati[eè�]res?\s*grasses|lipides|graisses|\bfett\b|\bvetten?\b|\bgrassi\b"
    r"|\bgrasas\b|الدهون|الدهنيات"
)
CARBOHYDRATE_KEYWORDS = (
    r"karbonhidrat|carbohydrates?|glucides|kohlenhydrate|koolhydraten|carboidrati"
    r"|hidratos\s*de\s*carbono|kolhydrater|الكربوهيدرات|الكاربوهيدرات"
)
SUGAR_KEYWORDS = r"[sş]eker|sugars?|sucres?|zucker|suikers|zuccheri|az[uú]cares|sockerarter|السكريات"
FIBER_KEYWORDS = r"\blif|fib(?:er|re)s?(?:\s*alimentaires?)?|ballaststoffe|vezels|\bfibra"
PROTEIN_KEYWORDS = r"prote[iíì]n|prot[eé�]ines?|eiwei[ßs]|eiwitten|البروتين"
SALT_KEYWORDS = r"\btuz\b|\bsalt\b|\bsel\b|\bsalz\b|\bzout\b|\bsale\b|\bsal\b|الملح"
SODIUM_KEYWORDS = r"sodyum|sodium"

ALL_NUTRIENT_KEYWORDS = "|".join(
    [
        ENERGY_KEYWORDS, SATURATED_FAT_KEYWORDS, FAT_KEYWORDS, CARBOHYDRATE_KEYWORDS,
        SUGAR_KEYWORDS, FIBER_KEYWORDS, PROTEIN_KEYWORDS, SALT_KEYWORDS, SODIUM_KEYWORDS,
    ]
)
_VALUE_UNIT = rf"(?:{GRAM_UNIT}|mg|kj|kcal)"
VALUE_ONLY_LINE = re.compile(
    rf"^[\s|:\-]*{NUMBER}\s*{_VALUE_UNIT}?(?:\s*[/|]?\s*{NUMBER}\s*{_VALUE_UNIT}?)*[\s|]*$", re.IGNORECASE
)


def _to_float(number_str: str) -> float:
    """OCR'da hem nokta hem virgul ondalik ayirici olarak gorulebilir."""
    return float(number_str.replace(",", "."))


def _search_value(text: str, keyword_pattern: str, unit_pattern: str) -> tuple[float, str] | None:
    """Anahtar kelimeden sonraki ilk "SAYI BIRIM" (ya da "BIRIM SAYI") eslesmesini doner.

    Bosluk (GAP) rakam icermez: anahtar kelimeden sonra gelen ILK sayi alinir, boylece
    "Seker 12 g | 4 g" gibi cok sutunlu satirlarda ilk sutun (100 g basina) secilir ve
    baska bir besinin degerine atlanmaz.
    """
    pattern = rf"(?:{keyword_pattern}){GAP}{NUMBER}\s*({unit_pattern})\b"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return _to_float(match.group(1)), match.group(2).lower()

    # Birim sayidan once: "Proteines | g | 3,0", "Lipides g 3,0", "Fat (g) 3.5" ve iki dilli
    # birim sutunu ("البروتينات | غ | g | 8,0"). Birim tek basina bir kelime olmali (harfle
    # bitisik degil) ve sayidan sonra baska birim gelmemeli.
    unit_token = rf"(?<![^\W\d_])({unit_pattern})(?![^\W\d_])[\s|:)\]]*"
    extra_unit = rf"(?:(?:{unit_pattern})(?![^\W\d_])[\s|:)\]]*)*"
    unit_first = (
        rf"(?:{keyword_pattern})[^\d\n]{{0,40}}?{unit_token}{extra_unit}"
        rf"{NUMBER}(?![\d.,]|\s*(?:{unit_pattern}|mg|%|kcal|kj)\b)"
    )
    match = re.search(unit_first, text, re.IGNORECASE)
    if match:
        return _to_float(match.group(2)), match.group(1).lower()
    return None


def _realign_column_table(text: str) -> str:
    """Once butun etiketler, sonra butun degerler gelen sutun duzenini satir satir eslestirir.

    Bazi OCR motorlari (ozellikle OCR.space) tabloyu sutun sutun okur:
    "Proteines\\nGlucides\\nLipides\\n10 g\\n3,1 g\\n18 g". Bu durumda etiket ile deger arasinda
    satirlar oldugundan normal arama ya hicbir sey bulamaz ya da yanlis degeri alir. En az 3
    rakamsiz etiket satirini, hemen ardindan gelen AYNI SAYIDA yalniz-deger satiriyla sirasiyla
    eslestirir. Sayilar tutmazsa (satir kaymasi, eksik deger) tahmin yurutmez, metni degistirmez.
    """
    lines = text.splitlines()

    def kind(line: str) -> str:
        stripped = line.strip()
        if not stripped:
            return "blank"
        if not re.search(r"\d", stripped) and re.search(ALL_NUTRIENT_KEYWORDS, stripped, re.IGNORECASE):
            return "label"
        if VALUE_ONLY_LINE.match(stripped):
            return "value"
        return "other"

    kinds = [kind(line) for line in lines]
    result: list[str] = []
    i = 0
    while i < len(lines):
        if kinds[i] != "label":
            result.append(lines[i])
            i += 1
            continue

        label_end = i
        while label_end < len(lines) and kinds[label_end] in ("label", "blank"):
            label_end += 1
        value_end = label_end
        while value_end < len(lines) and kinds[value_end] in ("value", "blank"):
            value_end += 1

        labels = [lines[j].strip() for j in range(i, label_end) if kinds[j] == "label"]
        values = [lines[j].strip() for j in range(label_end, value_end) if kinds[j] == "value"]
        if len(labels) >= 3 and len(labels) == len(values):
            result.extend(f"{label} {value}" for label, value in zip(labels, values))
            i = value_end
        else:
            result.extend(lines[i:label_end])
            i = label_end
    return "\n".join(result)


def extract_energy(text: str) -> tuple[float | None, float | None]:
    """Enerjiyi kcal ve/veya kJ olarak cikarir (ikisi de etikette bulunabilir).

    kcal/kJ birimleri beslenme etiketlerinde sadece enerji icin kullanildigindan
    (baska hicbir besin ogesi bu birimlerle olculmez), "enerji" anahtar kelimesine
    yakinlik aranmaz - bu, "Enerji 450 kcal / 1883 kJ" gibi iki degerin arasinda
    baska bir sayi (450) olan durumlarda da kJ'nin dogru yakalanmasini saglar.

    Siralama: (1) birimlerin once, degerlerin sonra geldigi cift ("kJ/kcal 972,5/235"),
    (2) "450 kcal", (3) birim once tek deger ("Kcal | 97"), (4) ABD tipi birimsiz
    "Calories 150" (kcal varsayilir).
    """
    energy_kcal = None
    energy_kj = None

    pair = re.search(rf"\bkj\s*/\s*kcal\b[^\d\n]{{0,15}}{NUMBER}\s*/\s*{NUMBER}", text, re.IGNORECASE)
    if pair:
        energy_kj, energy_kcal = _to_float(pair.group(1)), _to_float(pair.group(2))
    pair = re.search(rf"\bkcal\s*/\s*kj\b[^\d\n]{{0,15}}{NUMBER}\s*/\s*{NUMBER}", text, re.IGNORECASE)
    if pair and energy_kcal is None:
        energy_kcal, energy_kj = _to_float(pair.group(1)), _to_float(pair.group(2))

    if energy_kcal is None:
        kcal_match = re.search(rf"{NUMBER}\s*kcal\b", text, re.IGNORECASE) or re.search(
            rf"\bkcal\b[\s|:]*{NUMBER}(?![\d.,]|\s*(?:kcal|kj|%))", text, re.IGNORECASE
        )
        if kcal_match:
            energy_kcal = _to_float(kcal_match.group(1))
        else:
            calories_match = re.search(rf"calories?\D{{0,5}}{NUMBER}", text, re.IGNORECASE)
            if calories_match:
                energy_kcal = _to_float(calories_match.group(1))

    if energy_kj is None:
        kj_match = re.search(rf"{NUMBER}\s*kj\b", text, re.IGNORECASE) or re.search(
            rf"\bkj\b[\s|:]*{NUMBER}(?![\d.,]|\s*(?:kcal|kj|%))", text, re.IGNORECASE
        )
        if kj_match:
            energy_kj = _to_float(kj_match.group(1))

    return energy_kcal, energy_kj


def extract_saturated_fat(text: str) -> float | None:
    match = _search_value(text, SATURATED_FAT_KEYWORDS, GRAM_UNIT)
    return match[0] if match else None


def extract_fat(text: str) -> float | None:
    """Toplam yag - 'doymus yag' ile karismamasi icin o ifadeyi devre disi birakir."""
    # Once doymus yag ifadesini (degeriyle birlikte) metinden cikar, boylece plain 'yag'/'fett'
    # aramasi yanlislikla doymus yag degerini ("davon gesattigte Fettsauren 0,6 g") yakalamaz.
    cleaned = re.sub(
        rf"(?:{SATURATED_FAT_KEYWORDS}){GAP}\d+[.,]?\d*\s*(?:{GRAM_UNIT})\b", "", text, flags=re.IGNORECASE
    )
    match = _search_value(cleaned, FAT_KEYWORDS, GRAM_UNIT)
    return match[0] if match else None


def extract_carbohydrate(text: str) -> float | None:
    match = _search_value(text, CARBOHYDRATE_KEYWORDS, GRAM_UNIT)
    return match[0] if match else None


def extract_sugar(text: str) -> float | None:
    match = _search_value(text, SUGAR_KEYWORDS, GRAM_UNIT)
    return match[0] if match else None


def extract_fiber(text: str) -> float | None:
    match = _search_value(text, FIBER_KEYWORDS, GRAM_UNIT)
    return match[0] if match else None


def extract_protein(text: str) -> float | None:
    match = _search_value(text, PROTEIN_KEYWORDS, GRAM_UNIT)
    return match[0] if match else None


def extract_salt(text: str) -> float | None:
    match = _search_value(text, SALT_KEYWORDS, GRAM_UNIT)
    return match[0] if match else None


def extract_sodium(text: str) -> float | None:
    """Sodyum genelde mg olarak yazilir, bazen g olarak da gorulebilir (g ise mg'a cevrilir)."""
    mg_match = _search_value(text, SODIUM_KEYWORDS, r"mg")
    if mg_match:
        return mg_match[0]

    g_match = _search_value(text, SODIUM_KEYWORDS, GRAM_UNIT)
    if g_match:
        return g_match[0] * 1000

    return None


def extract_nutrition_facts(text: str) -> NutritionFacts:
    """OCR metninden tum besin degerlerini cikarip NutritionFacts nesnesine doldurur."""
    text = _realign_column_table(text)
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
