import cv2
import numpy as np

from src.data.image_preprocessing import (
    correct_contrast_clahe,
    denoise_image,
    preprocess_label_image,
    white_balance_gray_world,
)


def _noisy_flat_image(size=100, base_value=128, noise_std=25, seed=42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    flat = np.full((size, size, 3), base_value, dtype=np.float32)
    noise = rng.normal(0, noise_std, flat.shape)
    noisy = np.clip(flat + noise, 0, 255).astype(np.uint8)
    return noisy


def _low_contrast_image(size=100, low=100, high=120, seed=7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(low, high, (size, size, 3), dtype=np.uint8)


def test_denoise_image_reduces_noise_variance():
    noisy = _noisy_flat_image()
    denoised = denoise_image(noisy)

    assert denoised.std() < noisy.std()


def test_denoise_image_preserves_shape():
    noisy = _noisy_flat_image()
    denoised = denoise_image(noisy)
    assert denoised.shape == noisy.shape


def test_correct_contrast_clahe_widens_dynamic_range():
    low_contrast = _low_contrast_image()
    enhanced = correct_contrast_clahe(low_contrast)

    original_range = int(low_contrast.max()) - int(low_contrast.min())
    enhanced_range = int(enhanced.max()) - int(enhanced.min())

    assert enhanced_range > original_range


def test_white_balance_gray_world_equalizes_channel_means():
    # Kirmizi kanali yapay olarak yuksek olan bir goruntu
    image = np.zeros((50, 50, 3), dtype=np.uint8)
    image[:, :, 0] = 100  # B
    image[:, :, 1] = 100  # G
    image[:, :, 2] = 200  # R (renk dengesizligi)

    balanced = white_balance_gray_world(image)
    means = [balanced[:, :, i].mean() for i in range(3)]

    # Dengeleme sonrasi kanal ortalamalari birbirine orijinalden daha yakin olmali
    original_spread = max(100, 100, 200) - min(100, 100, 200)
    balanced_spread = max(means) - min(means)
    assert balanced_spread < original_spread


def test_preprocess_label_image_does_not_resize(tmp_path):
    src = tmp_path / "label.jpg"
    dest = tmp_path / "processed" / "label.jpg"

    original = _noisy_flat_image(size=137)  # tuhaf bir boyut - kasitli
    
    # Unicode safe write
    ret, buf = cv2.imencode(".jpg", original)
    assert ret
    buf.tofile(str(src))

    result_path = preprocess_label_image(src, dest)
    
    # Unicode safe read
    buffer = np.fromfile(str(result_path), dtype=np.uint8)
    result = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

    assert result_path == dest
    assert result.shape == original.shape


def test_preprocess_label_image_raises_on_missing_file(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        preprocess_label_image(tmp_path / "yok.jpg", tmp_path / "out.jpg")


def test_preprocess_label_image_corrects_exif_rotated_photo(tmp_path):
    """Regresyon: telefonla yan tutularak çekilen (EXIF Orientation etiketli) gerçek kullanıcı
    fotoğrafları (2026-07-22, canlı demoda karşılaşılan gerçek vaka - kalori/tuz/kategori hep
    anlamsız çıktı) için - cv2.imread/imdecode bu etiketi tamamen yok sayar, bu yüzden görüntü
    pipeline'a YAN piksel verisiyle girip hem CNN kategori tahminini hem OCR metin çıkarımını
    anlamsız hale getiriyordu. Sensörün yatay kaydedip EXIF Orientation=6 ile işaretlediği bir
    fotoğraf simüle edilir; çıktının gerçekten (sensör verisi değil, EXIF'in belirttiği) doğru
    yönde olduğu doğrulanır."""
    from PIL import Image as PILImage

    src = tmp_path / "rotated.jpg"
    dest = tmp_path / "processed" / "rotated.jpg"

    # Sensörün kaydettiği ham piksel verisi: yatay (200 genişlik x 100 yükseklik) - aslında
    # dikey bir etiketin 90 derece döndürülmüş hali
    sensor_data = np.full((100, 200, 3), 128, dtype=np.uint8)
    pil_img = PILImage.fromarray(sensor_data)
    exif = pil_img.getexif()
    exif[0x0112] = 6  # Orientation: goruntuleyici 90 derece dondurmeli
    pil_img.save(src, exif=exif)

    preprocess_label_image(src, dest)

    buffer = np.fromfile(str(dest), dtype=np.uint8)
    result = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

    # EXIF dogru uygulandiysa cikti DIKEY olmali (yatay sensor verisi degil)
    assert result.shape[0] > result.shape[1]


def test_correct_perspective_warps_skewed_quadrilateral():
    from src.data.image_preprocessing import correct_perspective
    # 400x400 siyah arka plan
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    # Beyaz bir dörtgen çizelim (etiket simülasyonu) - çerçevenin büyük kısmını kaplıyor
    pts = np.array([[50, 50], [350, 70], [320, 320], [70, 300]], dtype=np.int32)
    cv2.fillPoly(img, [pts], (255, 255, 255))

    warped = correct_perspective(img)
    assert warped.shape != img.shape
    assert warped[10, 10].mean() > 200


def test_correct_perspective_rejects_small_quad_not_covering_most_of_frame():
    """Regresyon: gercek bir OFF ornek goruntusunde (2026-07-14, Semih'in PR incelemesi)
    sadece besin degerleri tablosunun kucuk ic kutusu yanlislikla 'etiket siniri' sanilip
    warp edildi - goruntu 400x300'den 103x134'e kucularak icindekiler listesi/marka/barkod
    tamamen kayboldu. Cerceve'nin kucuk bir kismini (~%15) kaplayan bir dortgen artik
    secilmemeli; fonksiyon orijinal goruntuyu degismeden dondurmelidir."""
    from src.data.image_preprocessing import correct_perspective

    img = np.zeros((300, 400, 3), dtype=np.uint8)
    # Buyuk ama dortgene indirgenmeyecek bir sekil (yildiz benzeri, quad testini gecmemeli)
    star_pts = np.array(
        [[300, 60], [320, 120], [385, 120], [335, 155], [355, 215],
         [300, 178], [245, 215], [265, 155], [215, 120], [280, 120]],
        dtype=np.int32,
    )
    cv2.fillPoly(img, [star_pts], (255, 255, 255))
    # Kucuk bir dortgen (~%15 alan) - gercek etikette sadece besin tablosu gibi kucuk bir
    # alt-bolge - cercevenin cogunu kaplamadigi icin secilmemeli
    small_quad = np.array([[20, 20], [170, 25], [165, 145], [15, 140]], dtype=np.int32)
    cv2.fillPoly(img, [small_quad], (255, 255, 255))

    warped = correct_perspective(img)
    assert warped.shape == img.shape
    assert np.array_equal(warped, img)
