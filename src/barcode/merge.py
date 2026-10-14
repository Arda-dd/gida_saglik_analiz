"""Barkod (Open Food Facts) verisi ile etiketten OCR ile okunan verinin birlestirilmesi.

Oncelik OFF'ta: OCR'in alan dogrulugu en iyi motorda bile ~%67 iken OFF kaydi urunun tam besin
tablosudur. OFF'ta bos olan alanlar OCR'dan tamamlanir. Iki kaynak ayni alanda belirgin sekilde
ayrisiyorsa (tarif degismis, OFF kaydi hatali ya da OCR yanlis okumus olabilir) kullaniciya
UYARI verilir - hangisinin dogru oldugu otomatik bilinemez, OFF degeri kullanilir.
"""

from __future__ import annotations

from src.common.schema import NutritionFacts

FIELD_LABELS_TR = {
    "energy_kcal": "enerji (kcal)",
    "energy_kj": "enerji (kJ)",
    "fat_g": "yag",
    "saturated_fat_g": "doymus yag",
    "carbohydrate_g": "karbonhidrat",
    "sugar_g": "seker",
    "fiber_g": "lif",
    "protein_g": "protein",
    "salt_g": "tuz",
    "sodium_mg": "sodyum",
}


def values_agree(a: float, b: float, rel_tol: float = 0.15, abs_tol: float = 1.0) -> bool:
    """OCR degerlendirmesiyle (src/ocr/evaluate.py) ayni tolerans: max(±1.0, ±%15)."""
    return abs(a - b) <= max(abs_tol, abs(b) * rel_tol)


def merge_nutrition(
    barcode_nutrition: NutritionFacts, ocr_nutrition: NutritionFacts
) -> tuple[NutritionFacts, list[str], list[str]]:
    """(birlesik besin, OCR'dan tamamlanan alanlar, celisen alanlarin aciklamalari) doner."""
    base = barcode_nutrition.model_dump()
    ocr = ocr_nutrition.model_dump()
    filled_from_ocr: list[str] = []
    conflicts: list[str] = []
    for field_name, barcode_value in base.items():
        ocr_value = ocr.get(field_name)
        if barcode_value is None:
            if ocr_value is not None:
                base[field_name] = ocr_value
                filled_from_ocr.append(field_name)
        elif ocr_value is not None and not values_agree(ocr_value, barcode_value):
            label = FIELD_LABELS_TR.get(field_name, field_name)
            conflicts.append(f"{label}: etiket {ocr_value:g}, veritabani {barcode_value:g}")
    return NutritionFacts(**base), filled_from_ocr, conflicts
