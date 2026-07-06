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
    cv2.imwrite(str(src), original)

    result_path = preprocess_label_image(src, dest)
    result = cv2.imread(str(result_path))

    assert result_path == dest
    assert result.shape == original.shape


def test_preprocess_label_image_raises_on_missing_file(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        preprocess_label_image(tmp_path / "yok.jpg", tmp_path / "out.jpg")
