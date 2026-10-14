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

OCR motor kiyas raporu icin (2026-10-06, bkz. docs/ocr_comparison_report.md) iki motor daha
eklendi - ikisi de ek yazilim/model indirmesi gerektirmez:
- Windows OCR (Windows.Media.Ocr): Windows 10/11'e yerlesik, PowerShell uzerinden cagrilir.
- OpenAI gorsel modeli (gpt-4.1-mini): bir gorsel-dil modeline (VLM) etiketi oldugu gibi
  yaziya dokturur; cikti diger motorlarla AYNI regex normalize katmanindan gecer, yani model
  degerleri yorumlamaz/yapilandirmaz, sadece metni okur (adil kiyas icin).
"""

from __future__ import annotations

import base64
import json
import mimetypes
import logging
import os
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import cv2
import pytesseract
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

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
    usage: dict | None = None  # ucretli API'lerde token kullanimi (maliyet hesabi icin)
    # extract_text_live: atlanan/basarisiz motorlar ("openai_vision: TimeoutError") - sessiz
    # yedege dusmeyi gorunur kilmak icin (api/pipeline.py bunlari kullanici uyarisina cevirir).
    fallback_notes: list[str] = field(default_factory=list)


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


WINDOWS_OCR_SCRIPT = Path(__file__).resolve().parent / "windows_ocr.ps1"


def extract_text_windows(image_path: Path, lang: str = "tr") -> OCRResult:
    """Windows'un yerlesik OCR motoru (Windows.Media.Ocr) ile metin cikarir.

    WinRT API'sine Python'dan ek paket (winrt/winsdk) kurmadan erismek icin PowerShell
    uzerinden cagrilir (windows_ocr.ps1). Kelime kutulari Tesseract/EasyOCR ile ayni
    _group_boxes_into_rows mantigiyla satirlara dizilir.

    Not: Windows OCR kelime bazli guven skoru DONDURMEZ - mean_confidence, OCR.space'teki
    gibi sadece metin cikip cikmadigini gosteren ikili bir vekildir (100.0 / 0.0). Her cagri
    yeni bir PowerShell sureci baslattigi icin olculen sureye ~0.3-1 sn surec acilisi dahildir.
    """
    completed = subprocess.run(
        [
            "powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
            "-File", str(WINDOWS_OCR_SCRIPT), "-Path", str(Path(image_path).resolve()), "-Lang", lang,
        ],
        capture_output=True,
        timeout=60,
    )
    if completed.returncode != 0:
        error = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"Windows OCR hatasi: {error[:300]}")

    stdout = completed.stdout.decode("utf-8-sig", errors="replace").strip()
    words = json.loads(stdout) if stdout else []
    boxes = [
        {"ymin": w["y"], "ymax": w["y"] + w["height"], "xmin": w["x"], "height": w["height"], "text": w["text"]}
        for w in words
        if w.get("text", "").strip()
    ]
    text = _group_boxes_into_rows(boxes)
    return OCRResult(text=text, mean_confidence=100.0 if text else 0.0, engine="windows")


# Prompt v2 (2026-10-06). Ilk surum Turkce yazilmisti ve model 30 gorselin 3'unde etikette
# OLMAYAN Turkce metin uydurdu/cevirdi (okunamayan bir su sisesi icin eksiksiz bir "Besin
# Degerleri" tablosu dahil) - bkz. docs/ocr_comparison_report.md, docs/ocr_openai_vision_prompt_v1.json.
# Bu yuzden talimat Ingilizce, orijinal dili koru / cevirme / tahmin etme vurgulu ve okunamayan
# gorsel icin acik bir cikis (UNREADABLE_MARKER) iceriyor.
UNREADABLE_MARKER = "[UNREADABLE]"
OPENAI_VISION_PROMPT = (
    "You are an OCR engine. Transcribe ONLY the text that is actually legible in this image, "
    "exactly as printed, in its ORIGINAL language and script. Do NOT translate. Do NOT guess, "
    "infer, complete or normalize anything, and never write a nutrition table or values that "
    "are not clearly visible. Keep cells of the same table row on one line, separated by ' | '. "
    "If a table cell contains several stacked lines (e.g. a nutrient name with a 'of which' "
    "sub-line below it), write each stacked line as its own row with its own values. "
    "Skip illegible parts. If no text is legible at all, output exactly "
    f"{UNREADABLE_MARKER}. Output only the transcription, no comments."
)


VLM_MAX_IMAGE_SIDE = 2048


def _encode_image_for_vlm(image_path: Path, max_side: int = VLM_MAX_IMAGE_SIDE) -> tuple[str, str]:
    """Gorseli (mime, base64) olarak hazirlar; uzun kenari max_side'i asiyorsa kucultur.

    OpenAI "detail: high" gorseli zaten 2048 piksele indirir - telefon fotograflarini (~4000 px,
    birkac MB) oldugu gibi gondermek dogruluga katki saglamadan yuklemeyi ve gecikmeyi artirir.
    """
    import numpy as np

    raw = Path(image_path).read_bytes()
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or max(image.shape[:2]) <= max_side:
        mime = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
        return mime, base64.b64encode(raw).decode("ascii")

    scale = max_side / max(image.shape[:2])
    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, buffer = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ValueError(f"Gorsel JPEG'e kodlanamadi: {image_path}")
    return "image/jpeg", base64.b64encode(buffer.tobytes()).decode("ascii")


def extract_text_openai_vision(
    image_path: Path,
    model: str | None = None,
    client=None,
) -> OCRResult:
    """OpenAI gorsel-dil modeliyle (VLM) etiketi yaziya doker.

    Model bilerek SADECE transkripsiyon yapar (yapilandirilmis JSON/besin degeri cikarmaz):
    cikti diger motorlarla ayni extract_and_normalize regex katmanindan gecer, boylece kiyas
    "metni kim daha iyi okuyor" sorusunu olcer. mean_confidence vekildir (100.0 / 0.0);
    token kullanimi `usage` alaninda maliyet hesabi icin dondurulur. model verilmezse
    config.yaml -> ocr.openai_vision_model kullanilir.
    """
    if model is None:
        from src.common.config import get_config

        model = get_config().get("ocr", {}).get("openai_vision_model", "gpt-4.1-mini")
    if client is None:
        import openai

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY bulunamadi (.env dosyasini kontrol edin)")
        client = openai.OpenAI(api_key=api_key, timeout=60)

    mime, encoded = _encode_image_for_vlm(image_path)
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=1500,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": OPENAI_VISION_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}", "detail": "high"}},
                ],
            }
        ],
    )
    text = response.choices[0].message.content or ""
    if text.strip() == UNREADABLE_MARKER:
        text = ""
    usage = None
    if response.usage is not None:
        usage = {
            "model": model,
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        }
    return OCRResult(text=text, mean_confidence=100.0 if text.strip() else 0.0, engine="openai_vision", usage=usage)


REQUIRED_API_KEYS = {"openai_vision": "OPENAI_API_KEY", "ocrspace": "OCR_SPACE_API_KEY"}


def _live_engine_order() -> tuple[str, ...]:
    from src.common.config import get_config

    return tuple(get_config().get("ocr", {}).get("live_engines", ["ocrspace", "easyocr"]))


def extract_text_live(
    image_path: Path,
    engines: tuple[str, ...] | None = None,
    langs: tuple[str, ...] = ("tr", "en"),
) -> OCRResult:
    """Canli pipeline icin: motorlari sirayla dener (varsayilan sira config.yaml -> ocr.live_engines,
    2026-10-14'ten beri OpenAI gorsel -> OCR.space -> EasyOCR).

    Bir motor atlanir ya da bir sonrakine gecilir: API anahtari yoksa, hata verirse (kota/429,
    ag hatasi, zaman asimi) ya da hic metin okuyamazsa (OpenAI'in [UNREADABLE] cevabi dahil).
    Boylece kullanicinin istegi bulut motorlarindan biri basarisiz olsa bile COKMEZ - en kotu
    durumda yerel EasyOCR'in sonucuna duser (pipeline'in "opsiyonel katman basarisiz olursa
    devam et" felsefesi, bkz. api/pipeline.py). Hangi motorun sonucu kullanildigi
    OCRResult.engine'de doner. Hepsi hata verirse son hata yukari firlatilir.
    """
    extractors = {
        "openai_vision": extract_text_openai_vision,
        "ocrspace": extract_text_ocrspace,
        "easyocr": lambda path: extract_text_easyocr(path, langs=langs),
        "tesseract": extract_text_tesseract,
        "windows": extract_text_windows,
    }
    last_result: OCRResult | None = None
    last_error: Exception | None = None
    notes: list[str] = []
    for engine in engines or _live_engine_order():
        key_name = REQUIRED_API_KEYS.get(engine)
        if key_name and not os.environ.get(key_name):
            notes.append(f"{engine}: {key_name} yok")
            continue
        try:
            result = extractors[engine](image_path)
        except Exception as exc:
            logger.warning("OCR motoru %s basarisiz, bir sonrakine geciliyor: %s", engine, exc)
            notes.append(f"{engine}: {type(exc).__name__}")
            last_error = exc
            continue
        if result.text.strip():
            result.fallback_notes = notes
            return result
        notes.append(f"{engine}: metin okunamadi")
        last_result = result
    if last_result is not None:
        last_result.fallback_notes = notes
        return last_result
    if last_error is not None:
        raise last_error
    raise ValueError("Hicbir OCR motoru kullanilamadi (API anahtarlari ve config.yaml ocr.live_engines)")


def extract_text_with_ocrspace_fallback(
    image_path: Path, langs: tuple[str, ...] = ("tr", "en")
) -> OCRResult:
    """Eski canli sira (2026-08-22 - 2026-10-14): once OCR.space, olmazsa EasyOCR."""
    return extract_text_live(image_path, engines=("ocrspace", "easyocr"), langs=langs)
