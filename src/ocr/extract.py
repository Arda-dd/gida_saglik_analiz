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

import cv2
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


def _group_boxes_into_rows(boxes: list[dict]) -> str:
    """Genel satir gruplama mantigi: kutulari Y koordinatina gore siralar,
    benzer Y koordinatindakileri ayni satira alip soldan saga siralar ve birlestirir.
    """
    if not boxes:
        return ""

    # Y-koordinatina gore (yukaridan asagiya) sirala
    boxes = sorted(boxes, key=lambda b: b["ymin"])

    rows: list[list[dict]] = []
    for box in boxes:
        if not rows:
            rows.append([box])
            continue

        last_row = rows[-1]
        avg_height = sum(b["height"] for b in last_row) / len(last_row)
        avg_ymin = sum(b["ymin"] for b in last_row) / len(last_row)

        # Eger y-koordinatlari farki ortalama yuksekligin %60'indan az ise ayni satirdir
        if abs(box["ymin"] - avg_ymin) < (avg_height * 0.6):
            last_row.append(box)
        else:
            rows.append([box])

    # Her satiri kendi icinde X-koordinatina gore (soldan saga) sirala
    row_texts = []
    for row in rows:
        sorted_row = sorted(row, key=lambda b: b["xmin"])
        row_text = " ".join(b["text"] for b in sorted_row)
        row_texts.append(row_text)

    return "\n".join(row_texts)


def extract_text_tesseract(image_path: Path, lang: str = "tur+eng") -> OCRResult:
    """Tesseract ile metin + kelime bazli guven skorlarinin ortalamasini cikarir.
    Layout-aware satirlari gruplama mantigi ile calisir.
    """
    data = pytesseract.image_to_data(
        str(image_path), lang=lang, output_type=pytesseract.Output.DICT
    )

    boxes = []
    confidences = []
    n_boxes = len(data["text"])
    for i in range(n_boxes):
        text = data["text"][i]
        conf_value = float(data["conf"][i])
        if text.strip() and conf_value >= 0:
            boxes.append({
                "ymin": data["top"][i],
                "ymax": data["top"][i] + data["height"][i],
                "xmin": data["left"][i],
                "height": data["height"][i],
                "text": text
            })
            confidences.append(conf_value)

    full_text = _group_boxes_into_rows(boxes)
    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return OCRResult(text=full_text, mean_confidence=mean_conf, engine="tesseract")


@lru_cache(maxsize=1)
def _get_easyocr_reader(langs: tuple[str, ...] = ("tr", "en")):
    import easyocr
    import torch

    gpu_available = torch.cuda.is_available()
    return easyocr.Reader(list(langs), gpu=gpu_available)


def extract_text_easyocr(image_path: Path, langs: tuple[str, ...] = ("tr", "en")) -> OCRResult:
    """EasyOCR ile metin + guven skorlarinin ortalamasini cikarir.
    Layout-aware satir gruplama mantigi ile tablosal duzeni korur.
    """
    import numpy as np
    try:
        buffer = np.fromfile(str(image_path), dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    except Exception as e:
        raise ValueError(f"Gorsel okunamadi: {image_path} (Hata: {e})")

    if image is None:
        raise ValueError(f"Gorsel okunamadi: {image_path}")

    reader = _get_easyocr_reader(langs)
    results = reader.readtext(image)  # list[(bbox, text, confidence_0_1)]

    if not results:
        return OCRResult(text="", mean_confidence=0.0, engine="easyocr")

    boxes = []
    confidences = []
    for bbox, text, conf in results:
        # bbox = [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]
        xs = [pt[0] for pt in bbox]
        ys = [pt[1] for pt in bbox]
        ymin, ymax = min(ys), max(ys)
        xmin, xmax = min(xs), max(xs)
        height = ymax - ymin
        boxes.append({
            "ymin": ymin,
            "ymax": ymax,
            "xmin": xmin,
            "height": height,
            "text": text
        })
        confidences.append(conf * 100)

    full_text = _group_boxes_into_rows(boxes)
    return OCRResult(
        text=full_text, mean_confidence=sum(confidences) / len(confidences), engine="easyocr"
    )
