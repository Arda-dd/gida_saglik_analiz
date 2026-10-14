"""Gorsel-dil modeli (OpenAI) OCR'i icin uydurma korumasi.

OCR karsilastirma raporunda (docs/ocr_comparison_report.md, Bolum 5) OpenAI'in okunamayan
etiketlerde etikette olmayan, makul gorunen besin tablolari URETTIGI goruldu; talimat
iyilestirmesi bunu azaltti ama bitirmedi. Klasik OCR motorlari ise okuyamadiklari metni
uydurmaz, anlamsiz karakter uretir. Bu modul iki koruma saglar:

1. Capraz dogrulama: OpenAI'in okudugu her besin degeri, uydurma riski olmayan ikinci bir
   motorun (OCR.space) HAM metninde de sayi olarak geciyor mu? Gecmeyen degerler
   "dogrulanamadi" olarak isaretlenir (silinmez - ikinci motor da okuyamamis olabilir).
2. Gorsel kalite kontrolu: dusuk cozunurluk ya da bulaniklik, uydurmanin goruldugu kosullardi.
"""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

from src.common.schema import NutritionFacts

_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")

# Etikette dogrudan yazan alanlar; sodium_mg / energy_kj gibi alanlar cogu zaman digerlerinden
# HESAPLANIR (fill_missing_energy_and_salt) - metinde gecmeyen turetilmis degerler kontrol edilmez.
VERIFIABLE_FIELDS = (
    "energy_kcal", "energy_kj", "fat_g", "saturated_fat_g", "carbohydrate_g",
    "sugar_g", "fiber_g", "protein_g", "salt_g", "sodium_mg",
)

MIN_IMAGE_SIDE_PX = 600
MIN_SHARPNESS = 60.0  # Laplacian varyansi; altinda goruntu bulanik sayilir


def _numbers_in(text: str) -> set[float]:
    return {float(m.replace(",", ".")) for m in _NUMBER.findall(text or "")}


def _appears(value: float, numbers: set[float]) -> bool:
    return any(abs(value - n) < 0.051 for n in numbers)


def verify_nutrition(
    nutrition: NutritionFacts, primary_text: str, reference_text: str
) -> dict[str, bool]:
    """Birincil OCR metninde DOGRUDAN okunan her alan icin, degerin referans motorun ham
    metninde de gecip gecmedigini doner. Birincil metinde gecmeyen (turetilmis) alanlar atlanir."""
    primary_numbers = _numbers_in(primary_text)
    reference_numbers = _numbers_in(reference_text)
    result: dict[str, bool] = {}
    for field_name in VERIFIABLE_FIELDS:
        value = getattr(nutrition, field_name)
        if value is None or value == 0 or not _appears(value, primary_numbers):
            continue
        result[field_name] = _appears(value, reference_numbers)
    return result


def assess_image_quality(image_path: Path) -> list[str]:
    """Kullaniciya gosterilecek kalite uyarilarini doner (bos liste = sorun yok)."""
    try:
        image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    except OSError:
        return []
    if image is None:
        return []
    warnings: list[str] = []
    height, width = image.shape[:2]
    if max(height, width) < MIN_IMAGE_SIDE_PX:
        warnings.append(
            f"Fotograf cozunurlugu dusuk ({width}x{height}) - besin degerleri yanlis okunabilir; "
            "etiketi daha yakindan cekin."
        )
    sharpness = cv2.Laplacian(image, cv2.CV_64F).var()
    if sharpness < MIN_SHARPNESS:
        warnings.append("Fotograf bulanik gorunuyor - besin degerleri yanlis okunabilir; netleyip tekrar cekin.")
    return warnings
