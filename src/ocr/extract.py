"""Tesseract, EasyOCR ve OCR.space ile etiket gorsellerinden metin cikarimi.

Oneri formu 2.1: "Etiketlerde yer alan metinsel icerik, Tesseract ve EasyOCR gibi acik
kaynakli optik karakter tanima araclariyla cikarilacaktir." Iki motor da desteklenir ve
karsilastirma imkani sunar. Guven skorlari (confidence), OCR kalitesini bagimsiz
degisken olarak kaydetmek icin dondurulur (oneri formu 2.3).

OCR.space (ucretsiz bulut API, 2026-08-15'te eklendi) uculcu bir kiyas motoru olarak
eklendi: kok neden analizinde (bkz. docs/ocr_results_notes.md) asil darbogazin coklu
sutunlu tablo yapisi oldugu tespit edilmisti - OCR.space'in isTable parametresi, satirlari
TAB ile ayirarak (Engine 3'te Markdown tablo olarak) dondurup bu sorunu manuel satir/sutun
gruplamasi (_group_boxes_into_rows) yapmadan hedefliyor.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import pytesseract
import requests
from dotenv import load_dotenv

load_dotenv()

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


OCR_SPACE_ENDPOINT = "https://api.ocr.space/parse/image"


def extract_text_ocrspace(
    image_path: Path,
    api_key: str | None = None,
    language: str = "tur",
    engine: int = 3,
    session: requests.Session | None = None,
) -> OCRResult:
    """OCR.space ucretsiz bulut API'si ile metin cikarir.

    isTable=True parametresi satirlari TAB ile ayirip (Engine 3'te Markdown tablo olarak)
    dondurdugundan, Tesseract/EasyOCR'daki gibi manuel bir _group_boxes_into_rows adimina
    gerek yok - sutunlu yapi API'nin kendisinden geliyor.

    Not: OCR.space'in ucretsiz katmani Tesseract/EasyOCR'in aksine per-kelime bir guven
    skoru DONDURMEZ - bu yuzden mean_confidence burada gercek bir OCR guveni degil, sadece
    istegin basarili olup olmadigini (FileParseExitCode==1) yansitan ikili bir vekildir
    (basarili=100.0, basarisiz=0.0).
    """
    key = api_key or os.environ.get("OCR_SPACE_API_KEY")
    if not key:
        raise ValueError("OCR_SPACE_API_KEY ayarlanmamis (ortam degiskeni veya api_key parametresi)")

    http = session or requests.Session()
    with open(image_path, "rb") as f:
        response = http.post(
            OCR_SPACE_ENDPOINT,
            files={"file": f},
            data={
                "apikey": key,
                "language": language,
                "OCREngine": engine,
                "isTable": True,
                "scale": True,
            },
            timeout=30,
        )
    response.raise_for_status()
    data = response.json()

    if data.get("IsErroredOnProcessing"):
        error_message = data.get("ErrorMessage") or "Bilinmeyen OCR.space hatasi"
        if isinstance(error_message, list):
            error_message = "; ".join(error_message)
        raise ValueError(f"OCR.space hatasi: {error_message}")

    results = data.get("ParsedResults") or []
    if not results:
        return OCRResult(text="", mean_confidence=0.0, engine="ocrspace")

    text = results[0].get("ParsedText", "") or ""
    exit_code = results[0].get("FileParseExitCode", 0)
    mean_confidence = 100.0 if exit_code == 1 else 0.0
    return OCRResult(text=text, mean_confidence=mean_confidence, engine="ocrspace")


def extract_text_with_ocrspace_fallback(
    image_path: Path, langs: tuple[str, ...] = ("tr", "en")
) -> OCRResult:
    """Canli pipeline icin: once OCR.space'i dener (gercek degerlendirmede EasyOCR'in ~3
    kati alan dogrulugu, bkz. docs/ocr_results_notes.md), basarisiz olursa (API anahtari
    yok, ucretsiz kota siniri/429, ag hatasi) SESSIZCE EasyOCR'a duser.

    Boylece kullanicinin istegi OCR.space'in kota sinirina takilsa bile hicbir zaman
    basarisiz olmaz - sadece dogruluk EasyOCR seviyesine iner (analiz pipeline'inin genel
    "opsiyonel katman basarisiz olursa kural tabanli sonuca devam et" felsefesiyle tutarli,
    bkz. api/pipeline.py generate_explanation try/except).
    """
    if os.environ.get("OCR_SPACE_API_KEY"):
        try:
            return extract_text_ocrspace(image_path)
        except Exception:
            pass
    return extract_text_easyocr(image_path, langs=langs)
