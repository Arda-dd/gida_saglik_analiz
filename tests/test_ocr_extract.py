from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

from src.ocr.extract import OCRResult, extract_text_easyocr, extract_text_tesseract

FONT_PATH = Path(r"C:\Windows\Fonts\arial.ttf")


def _make_text_image(tmp_path, text, size=(500, 150), font_size=36):
    image = Image.new("RGB", size, color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(FONT_PATH), font_size) if FONT_PATH.exists() else ImageFont.load_default()
    draw.text((10, 40), text, fill=(0, 0, 0), font=font)

    path = tmp_path / "text_image.png"
    image.save(path)
    return path


def test_extract_text_tesseract_reads_simple_english_text(tmp_path):
    image_path = _make_text_image(tmp_path, "Energy 450 kcal")
    result = extract_text_tesseract(image_path, lang="eng")

    assert isinstance(result, OCRResult)
    assert result.engine == "tesseract"
    assert "450" in result.text
    assert result.mean_confidence > 0


def test_extract_text_tesseract_reads_turkish_text(tmp_path):
    image_path = _make_text_image(tmp_path, "Enerji 450 kcal Tuz 1.2 g")
    result = extract_text_tesseract(image_path, lang="tur+eng")

    assert "450" in result.text
    assert result.mean_confidence > 0


def test_extract_text_tesseract_blank_image_has_low_confidence(tmp_path):
    blank_path = tmp_path / "blank.png"
    Image.new("RGB", (200, 100), color=(255, 255, 255)).save(blank_path)

    result = extract_text_tesseract(blank_path, lang="eng")
    assert result.text.strip() == ""
    assert result.mean_confidence == 0.0


def test_extract_text_easyocr_reads_simple_text(tmp_path):
    image_path = _make_text_image(tmp_path, "Enerji 450 kcal")
    result = extract_text_easyocr(image_path, langs=("en",))

    assert isinstance(result, OCRResult)
    assert result.engine == "easyocr"
    assert "450" in result.text
    assert 0 <= result.mean_confidence <= 100


@pytest.mark.parametrize("engine_fn", [extract_text_tesseract, extract_text_easyocr])
def test_extract_functions_raise_or_handle_missing_file_gracefully(tmp_path, engine_fn):
    missing_path = tmp_path / "does_not_exist.png"
    with pytest.raises(Exception):
        engine_fn(missing_path)
