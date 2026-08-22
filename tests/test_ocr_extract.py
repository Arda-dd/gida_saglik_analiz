from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image, ImageDraw, ImageFont

from src.ocr.extract import (
    OCRResult,
    TESSERACT_EXE,
    extract_text_easyocr,
    extract_text_ocrspace,
    extract_text_tesseract,
    extract_text_with_ocrspace_fallback,
)

FONT_PATH = Path(r"C:\Windows\Fonts\arial.ttf")
tesseract_not_installed = not TESSERACT_EXE.exists()


def _make_text_image(tmp_path, text, size=(500, 150), font_size=36):
    image = Image.new("RGB", size, color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(FONT_PATH), font_size) if FONT_PATH.exists() else ImageFont.load_default()
    draw.text((10, 40), text, fill=(0, 0, 0), font=font)

    path = tmp_path / "text_image.png"
    image.save(path)
    return path


@pytest.mark.skipif(tesseract_not_installed, reason="Tesseract OCR is not installed")
def test_extract_text_tesseract_reads_simple_english_text(tmp_path):
    image_path = _make_text_image(tmp_path, "Energy 450 kcal")
    result = extract_text_tesseract(image_path, lang="eng")

    assert isinstance(result, OCRResult)
    assert result.engine == "tesseract"
    assert "450" in result.text
    assert result.mean_confidence > 0


@pytest.mark.skipif(tesseract_not_installed, reason="Tesseract OCR is not installed")
def test_extract_text_tesseract_reads_turkish_text(tmp_path):
    image_path = _make_text_image(tmp_path, "Enerji 450 kcal Tuz 1.2 g")
    result = extract_text_tesseract(image_path, lang="tur+eng")

    assert "450" in result.text
    assert result.mean_confidence > 0


@pytest.mark.skipif(tesseract_not_installed, reason="Tesseract OCR is not installed")
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


def test_extract_functions_raise_or_handle_missing_file_gracefully(tmp_path):
    missing_path = tmp_path / "does_not_exist.png"
    
    # EasyOCR
    with pytest.raises(Exception):
        extract_text_easyocr(missing_path)
        
    # Tesseract (sadece yuklu ise test et)
    if not tesseract_not_installed:
        with pytest.raises(Exception):
            extract_text_tesseract(missing_path)


def _mock_ocrspace_response(json_data: dict) -> MagicMock:
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = json_data
    return response


def test_extract_text_ocrspace_parses_successful_table_response(tmp_path):
    image_path = tmp_path / "label.jpg"
    image_path.write_bytes(b"fake-jpeg-bytes")

    session = MagicMock()
    session.post.return_value = _mock_ocrspace_response(
        {
            "IsErroredOnProcessing": False,
            "ParsedResults": [
                {"ParsedText": "Energie\t450 kcal\r\nSeker\t10 g\r\n", "FileParseExitCode": 1}
            ],
        }
    )

    result = extract_text_ocrspace(image_path, api_key="fake-key", session=session)

    assert isinstance(result, OCRResult)
    assert result.engine == "ocrspace"
    assert "450" in result.text
    assert result.mean_confidence == 100.0
    # apikey dogru gonderilmis mi kontrol et
    _, kwargs = session.post.call_args
    assert kwargs["data"]["apikey"] == "fake-key"
    assert kwargs["data"]["isTable"] is True


def test_extract_text_ocrspace_raises_on_api_error(tmp_path):
    image_path = tmp_path / "label.jpg"
    image_path.write_bytes(b"fake-jpeg-bytes")

    session = MagicMock()
    session.post.return_value = _mock_ocrspace_response(
        {"IsErroredOnProcessing": True, "ErrorMessage": ["Invalid API key"]}
    )

    with pytest.raises(ValueError, match="Invalid API key"):
        extract_text_ocrspace(image_path, api_key="fake-key", session=session)


def test_extract_text_ocrspace_raises_without_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("OCR_SPACE_API_KEY", raising=False)
    image_path = tmp_path / "label.jpg"
    image_path.write_bytes(b"fake-jpeg-bytes")

    with pytest.raises(ValueError, match="OCR_SPACE_API_KEY"):
        extract_text_ocrspace(image_path, api_key=None, session=MagicMock())


def test_extract_text_ocrspace_empty_parsed_results_returns_empty_text(tmp_path):
    image_path = tmp_path / "label.jpg"
    image_path.write_bytes(b"fake-jpeg-bytes")

    session = MagicMock()
    session.post.return_value = _mock_ocrspace_response(
        {"IsErroredOnProcessing": False, "ParsedResults": []}
    )

    result = extract_text_ocrspace(image_path, api_key="fake-key", session=session)
    assert result.text == ""
    assert result.mean_confidence == 0.0


def test_fallback_uses_ocrspace_when_key_present_and_call_succeeds(tmp_path, monkeypatch):
    monkeypatch.setenv("OCR_SPACE_API_KEY", "fake-key")
    image_path = tmp_path / "label.jpg"
    image_path.write_bytes(b"fake-jpeg-bytes")

    fake_result = OCRResult(text="Enerji 450 kcal", mean_confidence=100.0, engine="ocrspace")
    with patch("src.ocr.extract.extract_text_ocrspace", return_value=fake_result) as mocked:
        result = extract_text_with_ocrspace_fallback(image_path)

    mocked.assert_called_once()
    assert result.engine == "ocrspace"
    assert result.text == "Enerji 450 kcal"


def test_fallback_drops_to_easyocr_when_ocrspace_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("OCR_SPACE_API_KEY", "fake-key")
    image_path = _make_text_image(tmp_path, "Enerji 450 kcal")

    with patch(
        "src.ocr.extract.extract_text_ocrspace", side_effect=RuntimeError("429 Too Many Requests")
    ):
        result = extract_text_with_ocrspace_fallback(image_path, langs=("en",))

    assert result.engine == "easyocr"
    assert "450" in result.text


def test_fallback_uses_easyocr_directly_when_no_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("OCR_SPACE_API_KEY", raising=False)
    image_path = _make_text_image(tmp_path, "Enerji 450 kcal")

    result = extract_text_with_ocrspace_fallback(image_path, langs=("en",))

    assert result.engine == "easyocr"
    assert "450" in result.text
