"""Gorselden ya da OCR metninden barkod (GTIN: EAN-13, UPC-A, EAN-8) cikarimi.

Iki kaynak:
1. OpenCV'nin yerlesik barkod cozucusu (cv2.barcode, ek kurulum yok) - en guvenilir kaynak.
2. OCR metnindeki barkod numarasi (OpenAI/OCR.space barkodun altindaki rakamlari okuyabiliyor,
   ornegin "6 111252 421568"). Gurultulu metinden sahte kod cikmamasi icin yalnizca 12-13
   haneli ve sagla hanesi (check digit) tutan diziler kabul edilir: 2026-10-14 olcumunde
   EasyOCR'in bozuk metninden sagla hanesi tutan ama YANLIS iki 8 haneli kod cikti, 13 hanelilerde
   yanlis kod cikmadi. 8 haneli EAN-8 bu yuzden sadece gercek cozucuden kabul edilir.
"""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

_TEXT_BARCODE = re.compile(r"(?<![\d.,])\d(?:[ ]?\d){11,12}(?![\d.,])")


def is_valid_gtin(code: str) -> bool:
    """GTIN sagla hanesi kontrolu (EAN-8, UPC-A/GTIN-12, EAN-13): sagdan sola 3-1 agirlik."""
    if not code.isdigit() or len(code) not in (8, 12, 13):
        return False
    digits = [int(c) for c in code]
    body, check = digits[:-1], digits[-1]
    total = sum(d * (3 if i % 2 == 0 else 1) for i, d in enumerate(reversed(body)))
    return (10 - total % 10) % 10 == check


def barcodes_from_text(text: str) -> list[str]:
    """OCR metnindeki gecerli 12-13 haneli barkodlari (bosluklar silinmis, sirali) doner."""
    found: list[str] = []
    for match in _TEXT_BARCODE.finditer(text or ""):
        code = match.group(0).replace(" ", "")
        if is_valid_gtin(code) and code not in found:
            found.append(code)
    return found


def decode_barcodes(image_path: Path) -> list[str]:
    """Gorseldeki barkodlari OpenCV ile cozer; bulamazsa gri + 2x buyutulmus halini de dener."""
    try:
        image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    except OSError:
        return []
    if image is None:
        return []
    detector = cv2.barcode.BarcodeDetector()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    upscaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    # OpenCV cozucusu cok keskin (ya da cok kucuk) cubuk kenarlarinda bazen basarisiz oluyor;
    # hafif bulaniklastirilmis ve buyutulmus varyantlar sirayla denenir (ilk basarili sonuc alinir).
    candidates = [
        image,
        cv2.GaussianBlur(image, (5, 5), 0),
        upscaled,
        cv2.GaussianBlur(upscaled, (5, 5), 0),
    ]
    for candidate in candidates:
        _ok, infos, _types, _points = detector.detectAndDecodeWithType(candidate)
        codes = [code for code in (infos or ()) if code and is_valid_gtin(code)]
        if codes:
            return list(dict.fromkeys(codes))
    return []
