"""Tesseract ve EasyOCR ile etiket gorsellerinden metin cikarimi.

Oneri formu 2.1: "Etiketlerde yer alan metinsel icerik, Tesseract ve EasyOCR gibi acik
kaynakli optik karakter tanima araclariyla cikarilacaktir." Iki motor da desteklenir ve
karsilastirma imkani sunar. Guven skorlari (confidence), OCR kalitesini bagimsiz
degisken olarak kaydetmek icin dondurulur (oneri formu 2.3).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import pytesseract

from src.common.config import get_config

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


# --- PaddleOCR / PP-Structure (Faz 3, tablo-yapisi farkindali cikarim) ---------------
#
# Neden eklendi: EasyOCR/Tesseract besin tablosunu duz metne cevirdiginde "Per 100g" ve
# "Per porsiyon" sutunlari birbirine karisiyor ve etiket-deger bagi kopuyordu (bkz.
# docs/ocr_results_notes.md kok neden analizi madde 3). Satir gruplama bunu cozmedi
# (%16.2 -> %15.8), cunku sorun satirlar arasi degil SUTUNLAR arasiydi. PP-Structure
# tabloyu yapisiyla birlikte cikarir (<tr><td>etiket</td><td>deger</td></tr>), yani
# etiket-deger eslesmesi kaynakta korunur.

_TABLE_ROW_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
_TABLE_CELL_RE = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.IGNORECASE | re.DOTALL)
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _ensure_paddle_env() -> None:
    """PaddleOCR'in bu ortamda calismasi icin gereken iki ortam kosulunu ayarlar.

    Ikisi de gercek, tespit edilmis hatalardir - paddleocr import edilmeden ONCE
    ayarlanmalidir (paddlex bayraklari import aninda okur):

    1. ASCII olmayan onbellek yolu: Paddle'in C++ inference motoru, model yolunda
       ASCII disi karakter (or. "C:/Users/Semih Erdogan/.paddlex" icindeki "g")
       oldugunda dosyayi acamiyor ve bos girdi okudugu icin
       "[json.exception.parse_error.101] ... attempting to parse an empty input"
       hatasi veriyor. Cozum: yolu Windows 8.3 kisa adina cevirmek.
    2. oneDNN/MKLDNN + PIR yurutucu uyumsuzlugu: layout tespiti modelinde
       "NotImplementedError: ConvertPirAttribute2RuntimeAttribute not support
       [pir::ArrayAttribute<pir::DoubleAttribute>]" hatasina yol aciyor.
       Cozum: MKLDNN'i varsayilan olarak kapatmak.
    """
    os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "False")

    if "PADDLE_PDX_CACHE_HOME" in os.environ:
        return

    cache_home = Path.home() / ".paddlex"
    if str(cache_home).isascii():
        return

    cache_home.mkdir(parents=True, exist_ok=True)
    short = _windows_short_path(cache_home)
    if short is not None and short.isascii():
        os.environ["PADDLE_PDX_CACHE_HOME"] = short


def _windows_short_path(path: Path) -> str | None:
    """Bir yolun Windows 8.3 kisa adini doner (yol var olmalidir), yoksa None.

    8.3 ad uretimi bazi birimlerde kapali olabilir - o durumda None donulur ve
    cagiran taraf sessizce varsayilan yolla devam eder.
    """
    try:
        import ctypes
        from ctypes import wintypes

        get_short = ctypes.windll.kernel32.GetShortPathNameW
        get_short.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
        get_short.restype = wintypes.DWORD

        buffer = ctypes.create_unicode_buffer(1024)
        length = get_short(str(path), buffer, len(buffer))
        if 0 < length < len(buffer):
            return buffer.value
    except (OSError, AttributeError, ValueError):
        pass
    return None


def _table_html_to_lines(html: str) -> str:
    """PP-Structure'in tablo HTML'ini satir basina bir 'etiket deger' metnine cevirir.

    <tr><td>Glucides</td><td>35g</td></tr> -> "Glucides 35g"

    Cok sutunlu tablolarda (Per 100g | Per porsiyon) hucreler soldan saga sirali
    kalir; normalize.py anahtar kelimeden sonraki ILK sayiyi aldigindan, etikette
    once gelen 100g sutunu dogal olarak tercih edilir.
    """
    lines = []
    for row_html in _TABLE_ROW_RE.findall(html):
        cells = [_HTML_TAG_RE.sub("", cell).strip() for cell in _TABLE_CELL_RE.findall(row_html)]
        cells = [cell for cell in cells if cell]
        if cells:
            lines.append(" ".join(cells))
    return "\n".join(lines)


@lru_cache(maxsize=1)
def _get_pp_structure():
    """PP-Structure boru hattini kurar (ilk cagrida modeller indirilir, sonra onbellekten).

    Besin etiketlerinde bulunmayan alt modulleri (formul, muhur, grafik, bolge tespiti)
    kapatiyoruz: varsayilan yapilandirma ~10 model yukluyor ve bu, hem baslatmayi hem de
    goruntu basina cikarimi belirgin sekilde yavaslatiyordu. Belge yonu ve satir yonu
    siniflandirmasi ACIK birakildi - gercek telefon fotograflari egik/dondurulmus
    olabiliyor (bkz. EXIF yonu duzeltmesi, commit 2060dbe).
    """
    _ensure_paddle_env()
    from paddleocr import PPStructureV3

    return PPStructureV3(
        use_table_recognition=True,
        use_doc_orientation_classify=True,
        use_textline_orientation=True,
        use_doc_unwarping=False,  # kendi perspektif duzeltmemiz var (image_preprocessing.py)
        use_seal_recognition=False,
        use_formula_recognition=False,
        use_chart_recognition=False,
        use_region_detection=False,
    )


def extract_text_paddleocr(image_path: Path) -> OCRResult:
    """PP-Structure ile tablo yapisini koruyarak metin cikarir.

    Cikti metni iki bolumden olusur:
      1. Tespit edilen tablolarin satirlari ("etiket deger" ciftleri) - EN BASTA,
         cunku normalize.py ilk eslesmeyi alir ve en guvenilir eslesme budur.
      2. Sayfanin geri kalan OCR metni - "POUR 100g" / "porsiyon" gibi baz
         belirteclerinin (detect_nutrition_basis) tablonun DISINDA kalabilmesi icin
         gereklidir.
    """
    if not Path(image_path).exists():
        raise ValueError(f"Gorsel okunamadi: {image_path}")

    pipeline = _get_pp_structure()
    results = list(pipeline.predict(str(image_path)))
    if not results:
        return OCRResult(text="", mean_confidence=0.0, engine="paddleocr")

    res = results[0].json["res"]

    table_lines = [
        _table_html_to_lines(table.get("pred_html", ""))
        for table in res.get("table_res_list", [])
    ]

    overall = res.get("overall_ocr_res", {}) or {}
    rec_texts = overall.get("rec_texts", []) or []
    rec_scores = overall.get("rec_scores", []) or []

    boxes = []
    for text, box in zip(rec_texts, overall.get("rec_boxes", []) or []):
        # rec_boxes: [xmin, ymin, xmax, ymax]
        if len(box) < 4:
            continue
        xmin, ymin, xmax, ymax = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
        boxes.append(
            {"ymin": ymin, "ymax": ymax, "xmin": xmin, "height": ymax - ymin, "text": text}
        )

    page_text = _group_boxes_into_rows(boxes) if boxes else "\n".join(rec_texts)

    sections = [line for line in table_lines if line]
    if page_text:
        sections.append(page_text)

    mean_conf = (sum(rec_scores) / len(rec_scores) * 100) if rec_scores else 0.0
    return OCRResult(text="\n".join(sections), mean_confidence=mean_conf, engine="paddleocr")


# --- Motor secimi ---------------------------------------------------------------

OCR_ENGINES = {
    "tesseract": extract_text_tesseract,
    "easyocr": extract_text_easyocr,
    "paddleocr": extract_text_paddleocr,
}


def extract_text(image_path: Path) -> OCRResult:
    """config.yaml -> ocr.engine ile secilen motorla metin cikarir.

    Pipeline'in (api/pipeline.py) tek giris noktasidir; motoru degistirmek icin kod
    degil config yeterlidir (form Risk Yonetimi B-plani ile ayni felsefe).
    """
    engine = get_config()["ocr"].get("engine", "paddleocr")
    if engine not in OCR_ENGINES:
        raise ValueError(
            f"Bilinmeyen OCR motoru: {engine!r} (secenekler: {sorted(OCR_ENGINES)})"
        )
    return OCR_ENGINES[engine](image_path)
