"""Etiket gorselleri icin OCR-oncesi on isleme (gurultu azaltma, kontrast, renk dengeleme).

ONEMLI: Bu modul KESINLIKLE resize/kucultme yapmaz. Etiketler Faz 3'teki OCR icin
YUKSEK COZUNURLUKTE saklanmalidir (bkz. oneri formu 2.1: "Her etiket yuksek
cozunurlukte kaydedilecek"). 224x224 gibi kucultmeler SADECE Faz 2'nin CNN egitim
pipeline'inda `torchvision.transforms.Resize` ile bellek-ici/gecici olarak yapilir,
diske asla kalici kucuk halde yazilmaz.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def denoise_image(image: np.ndarray) -> np.ndarray:
    """Renkli gorseldeki gurultuyu azaltir (fastNlMeansDenoisingColored)."""
    return cv2.fastNlMeansDenoisingColored(image, None, h=7, hColor=7, templateWindowSize=7, searchWindowSize=21)


def correct_contrast_clahe(image: np.ndarray) -> np.ndarray:
    """CLAHE (Contrast Limited Adaptive Histogram Equalization) ile kontrasti iyilestirir.

    Islem LAB renk uzayinin sadece L (parlaklik) kanalinda yapilir; renk bilgisi bozulmaz.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)

    merged = cv2.merge((l_enhanced, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def white_balance_gray_world(image: np.ndarray) -> np.ndarray:
    """Gray World varsayimiyla basit beyaz dengesi (renk dengeleme) uygular."""
    result = image.astype(np.float32)
    mean_b, mean_g, mean_r = (result[:, :, i].mean() for i in range(3))
    mean_gray = (mean_b + mean_g + mean_r) / 3

    for i, channel_mean in enumerate((mean_b, mean_g, mean_r)):
        if channel_mean > 0:
            result[:, :, i] *= mean_gray / channel_mean

    return np.clip(result, 0, 255).astype(np.uint8)


def preprocess_label_image(src_path: Path, dest_path: Path) -> Path:
    """Bir etiket gorselini okur, gurultu/kontrast/renk duzeltir, ORIJINAL cozunurlukte kaydeder.

    Bu fonksiyon resize YAPMAZ - cikti boyutu girdiyle ayni olmalidir (bkz. modul dokstring'i).
    """
    # Windows'ta Turkce karakter iceren yollarla uyumluluk icin np.fromfile kullanilarak okunur
    try:
        buffer = np.fromfile(str(src_path), dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    except Exception as e:
        raise ValueError(f"Gorsel okunamadi: {src_path} (Hata: {e})")

    if image is None:
        raise ValueError(f"Gorsel okunamadi: {src_path}")

    original_shape = image.shape

    processed = denoise_image(image)
    processed = correct_contrast_clahe(processed)
    processed = white_balance_gray_world(processed)

    assert processed.shape == original_shape, "preprocess_label_image resize yapmamalidir"

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Windows'ta Turkce karakter iceren yollarla uyumluluk icin cv2.imencode + tofile kullanilarak yazilir
    ret, buf = cv2.imencode(".jpg", processed)
    if not ret:
        raise ValueError(f"Gorsel encode edilemedi: {dest_path}")
    buf.tofile(str(dest_path))
    
    return dest_path
