"""Tesseract ve EasyOCR ile etiket gorsellerinden metin cikarimi.

Oneri formu 2.1: "Etiketlerde yer alan metinsel icerik, Tesseract ve EasyOCR gibi acik
kaynakli optik karakter tanima araclariyla cikarilacaktir." Iki motor da desteklenir ve
karsilastirma imkani sunar. Guven skorlari (confidence), OCR kalitesini bagimsiz
degisken olarak kaydetmek icin dondurulur (oneri formu 2.3).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pytesseract

TESSERACT_EXE = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
TESSDATA_DIR = (Path(__file__).resolve().parents[2] / "data" / "tessdata").resolve()

if TESSERACT_EXE.exists():
    pytesseract.pytesseract.tesseract_cmd = str(TESSERACT_EXE)

# Not: pytesseract'in config string'i Windows'ta shlex.split(posix=False) ile parse etmesi,
# tirnak icindeki --tessdata-dir degerinin tirnaklarini TEMIZLEMIYOR (bilinen bir sorun) -
# bu da yol sonuna literal '"' karakteri eklenmesine ve dosyanin bulunamamasina yol aciyor.
# Bu yuzden --tessdata-dir CLI parametresi yerine TESSDATA_PREFIX ortam degiskeni kullanilir.
os.environ["TESSDATA_PREFIX"] = str(TESSDATA_DIR)


@dataclass
class OCRResult:
    text: str
    mean_confidence: float  # 0-100 araliginda, motorlar arasi karsilastirilabilir
    engine: str


def extract_text_tesseract(image_path: Path, lang: str = "tur+eng") -> OCRResult:
    """Tesseract ile metin + kelime bazli guven skorlarinin ortalamasini cikarir.

    Dil verisi konumu TESSDATA_PREFIX ortam degiskeni ile belirlenir (modul yuklenirken
    ayarlanir) - --tessdata-dir CLI parametresi kullanilmaz (bkz. modul basindaki not).
    """
    data = pytesseract.image_to_data(
        str(image_path), lang=lang, output_type=pytesseract.Output.DICT
    )

    words: list[str] = []
    confidences: list[float] = []
    for text, conf in zip(data["text"], data["conf"]):
        conf_value = float(conf)
        if text.strip() and conf_value >= 0:
            words.append(text)
            confidences.append(conf_value)

    full_text = " ".join(words)
    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return OCRResult(text=full_text, mean_confidence=mean_conf, engine="tesseract")


@lru_cache(maxsize=1)
def _get_easyocr_reader(langs: tuple[str, ...] = ("tr", "en")):
    import easyocr

    return easyocr.Reader(list(langs), gpu=True)


def extract_text_easyocr(image_path: Path, langs: tuple[str, ...] = ("tr", "en")) -> OCRResult:
    """EasyOCR ile metin + guven skorlarinin ortalamasini (0-100 bazina cevrilmis) cikarir."""
    reader = _get_easyocr_reader(langs)
    results = reader.readtext(str(image_path))  # list[(bbox, text, confidence_0_1)]

    if not results:
        return OCRResult(text="", mean_confidence=0.0, engine="easyocr")

    texts = [r[1] for r in results]
    confidences = [r[2] * 100 for r in results]

    return OCRResult(
        text=" ".join(texts), mean_confidence=sum(confidences) / len(confidences), engine="easyocr"
    )
