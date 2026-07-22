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
from PIL import Image, ImageOps


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


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Köşe koordinatlarını sıralar: sol-üst, sağ-üst, sağ-alt, sol-alt."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Sol-üst
    rect[2] = pts[np.argmax(s)]  # Sağ-alt

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # Sağ-üst
    rect[3] = pts[np.argmax(diff)] # Sol-alt
    return rect


MIN_LABEL_AREA_RATIO = 0.4
"""Bir dortgen konturun 'etiketin tamami' sayilabilmesi icin kaplamasi gereken minimum alan
orani. Onceki esik (%10) gercek bir OFF ornek goruntusunde cok dusuk cikti: sadece besin
degerleri tablosunun kucuk ic kutusu bu esigi gecip yanlislikla 'etiket siniri' sanildi ve
goruntu 400x300'den 103x134'e kirpilarak icindekiler listesi/marka/barkod (alerjen tespiti
icin gerekli) tamamen kayboldu. Gercek bir etiket fotografi CERCEVENIN COGUNU kaplar; bu
yuzden esik yukseltildi - kucuk/supheli bir dortgen adayi varsa YANLIS kirpma yerine
perspektif duzeltmeden vazgecilip orijinal goruntu kullanilir (daha guvenli varsayilan)."""


def correct_perspective(image: np.ndarray) -> np.ndarray:
    """Görüntüdeki, çerçevenin büyük kısmını kaplayan dörtgen konturu (etiket sınırları) bulup
    perspektif (Homografi) düzelterek kuşbakışı görünüm elde eder.
    Eğer bu kritere uyan bir dörtgen kontur bulunamazsa orijinal görüntüyü döner.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    screen_cnt = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            img_area = image.shape[0] * image.shape[1]
            if cv2.contourArea(c) > MIN_LABEL_AREA_RATIO * img_area:
                screen_cnt = approx
                break

    if screen_cnt is None:
        return image

    pts = screen_cnt.reshape(4, 2)
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect

    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    if max_width <= 0 or max_height <= 0:
        return image

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))
    return warped


def _load_image_bgr_exif_corrected(src_path: Path) -> np.ndarray:
    """Bir gorseli EXIF Orientation etiketini uygulayarak (dogru yonde) BGR (OpenCV) formatinda okur.

    Telefon kameralari sensor verisini genelde SABIT bir yonde kaydedip, "bu nasil gosterilmeli"
    bilgisini ayri bir EXIF Orientation etiketinde tutar - fotograf galerileri/tarayicilar bu
    etiketi otomatik uygulayip gorseli dogru yonde gosterir. cv2.imread/imdecode bu etiketi
    TAMAMEN YOK SAYAR: telefon yan tutularak (90 derece dondurulmus) cekilen, bizim gordugumuz
    (dogru yonde) fotograf, pipeline'a YAN/TERS piksel verisi olarak girer - bu da hem CNN
    kategori tahminini hem OCR metin cikarimini anlamsiz hale getirir (gercek kullanici
    fotograflarinda gozlemlendi, 2026-07-22). PIL uzerinden okuyup ImageOps.exif_transpose()
    ile fiziksel donusu uygulayarak bu farki kapatiyoruz - PIL, Windows'ta Turkce/Unicode
    dosya yollarini da cv2'nin aksine dogrudan destekler.
    """
    with Image.open(src_path) as img:
        img = ImageOps.exif_transpose(img)
        rgb = np.array(img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def preprocess_label_image(src_path: Path, dest_path: Path) -> Path:
    """Bir etiket gorselini okur, EXIF yonunu duzeltir, perspektif/gurultu/kontrast/renk
    duzeltir, yuksek cozunurlukte kaydeder."""
    try:
        image = _load_image_bgr_exif_corrected(src_path)
    except Exception as e:
        raise ValueError(f"Gorsel okunamadi: {src_path} (Hata: {e})")

    if image is None:
        raise ValueError(f"Gorsel okunamadi: {src_path}")

    processed = correct_perspective(image)
    processed = denoise_image(processed)
    processed = correct_contrast_clahe(processed)
    processed = white_balance_gray_world(processed)

    # Perspektif donusumu sonrasi boyut degisebilir, ancak yuksek cozunurluk korunmalidir
    assert processed.shape[0] >= 100 and processed.shape[1] >= 100, "preprocess_label_image cozunurluk dusurmemelidir"

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Windows'ta Turkce karakter iceren yollarla uyumluluk icin cv2.imencode + tofile kullanilarak yazilir
    ret, buf = cv2.imencode(".jpg", processed)
    if not ret:
        raise ValueError(f"Gorsel encode edilemedi: {dest_path}")
    buf.tofile(str(dest_path))
    
    return dest_path
