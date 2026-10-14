import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image, ImageDraw, ImageFont

from src.ocr.extract import (
    OCRResult,
    TESSERACT_EXE,
    extract_text_easyocr,
    extract_text_ocrspace,
    extract_text_openai_vision,
    extract_text_tesseract,
    extract_text_windows,
    extract_text_live,
    extract_text_with_ocrspace_fallback,
    _encode_image_for_vlm,
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


@pytest.mark.skipif(os.name != "nt", reason="Windows OCR sadece Windows'ta var")
def test_extract_text_windows_reads_simple_text_for_real(tmp_path):
    # Gercek Windows OCR cagrisi (ag/anahtar gerektirmez, Windows'a yerlesik).
    image_path = _make_text_image(tmp_path, "Energy 450 kcal")
    result = extract_text_windows(image_path)

    assert result.engine == "windows"
    assert "450" in result.text
    assert result.mean_confidence == 100.0


def test_extract_text_windows_groups_word_boxes_into_rows(tmp_path):
    words = [
        {"text": "450", "x": 120, "y": 10, "height": 12},
        {"text": "Enerji", "x": 5, "y": 11, "height": 12},
        {"text": "Seker", "x": 5, "y": 40, "height": 12},
        {"text": "12", "x": 120, "y": 41, "height": 12},
    ]
    completed = MagicMock(returncode=0, stdout=json.dumps(words).encode("utf-8"), stderr=b"")
    with patch("src.ocr.extract.subprocess.run", return_value=completed):
        result = extract_text_windows(tmp_path / "x.png")

    assert result.text == "Enerji 450\nSeker 12"


def test_extract_text_windows_empty_output_returns_empty_text(tmp_path):
    completed = MagicMock(returncode=0, stdout=b"[]", stderr=b"")
    with patch("src.ocr.extract.subprocess.run", return_value=completed):
        result = extract_text_windows(tmp_path / "x.png")

    assert result.text == ""
    assert result.mean_confidence == 0.0


def test_extract_text_windows_raises_on_powershell_error(tmp_path):
    completed = MagicMock(returncode=1, stdout=b"", stderr="dil desteklenmiyor".encode("utf-8"))
    with patch("src.ocr.extract.subprocess.run", return_value=completed):
        with pytest.raises(ValueError, match="Windows OCR hatasi"):
            extract_text_windows(tmp_path / "x.png")


def _fake_openai_client(text, prompt_tokens=400, completion_tokens=150):
    client = MagicMock()
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = text
    response.usage.prompt_tokens = prompt_tokens
    response.usage.completion_tokens = completion_tokens
    client.chat.completions.create.return_value = response
    return client


def test_extract_text_openai_vision_returns_transcription_and_usage(tmp_path):
    image_path = _make_text_image(tmp_path, "Energy 450 kcal")
    client = _fake_openai_client("Enerji | 450 kcal")

    result = extract_text_openai_vision(image_path, model="gpt-4.1-mini", client=client)

    assert result.engine == "openai_vision"
    assert result.text == "Enerji | 450 kcal"
    assert result.usage == {"model": "gpt-4.1-mini", "input_tokens": 400, "output_tokens": 150}


def test_extract_text_openai_vision_sends_image_as_base64_data_url(tmp_path):
    image_path = _make_text_image(tmp_path, "Energy 450 kcal")
    client = _fake_openai_client("x")

    extract_text_openai_vision(image_path, client=client)

    content = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    image_part = next(part for part in content if part["type"] == "image_url")
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")
    assert client.chat.completions.create.call_args.kwargs["temperature"] == 0


def test_extract_text_openai_vision_raises_without_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        extract_text_openai_vision(tmp_path / "x.png")


def test_extract_text_openai_vision_unreadable_marker_becomes_empty_text(tmp_path):
    # Okunamayan gorselde model uydurmak yerine isaretci donmeli; bu bos metin sayilir.
    image_path = _make_text_image(tmp_path, "x")
    client = _fake_openai_client("[UNREADABLE]")

    result = extract_text_openai_vision(image_path, client=client)

    assert result.text == ""
    assert result.mean_confidence == 0.0


def _ok(engine, text="Enerji 450 kcal"):
    return OCRResult(text=text, mean_confidence=100.0, engine=engine)


def test_live_ocr_uses_openai_first_when_key_present(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    monkeypatch.setenv("OCR_SPACE_API_KEY", "fake")
    with patch("src.ocr.extract.extract_text_openai_vision", return_value=_ok("openai_vision")), patch(
        "src.ocr.extract.extract_text_ocrspace"
    ) as ocrspace:
        result = extract_text_live(tmp_path / "x.jpg", engines=("openai_vision", "ocrspace", "easyocr"))

    assert result.engine == "openai_vision"
    ocrspace.assert_not_called()


def test_live_ocr_falls_back_when_openai_errors_or_reads_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    monkeypatch.setenv("OCR_SPACE_API_KEY", "fake")
    engines = ("openai_vision", "ocrspace", "easyocr")
    with patch("src.ocr.extract.extract_text_openai_vision", side_effect=TimeoutError()), patch(
        "src.ocr.extract.extract_text_ocrspace", return_value=_ok("ocrspace")
    ):
        assert extract_text_live(tmp_path / "x.jpg", engines=engines).engine == "ocrspace"

    # [UNREADABLE] -> bos metin: bir sonraki motor denenir.
    with patch("src.ocr.extract.extract_text_openai_vision", return_value=_ok("openai_vision", "")), patch(
        "src.ocr.extract.extract_text_ocrspace", return_value=_ok("ocrspace")
    ):
        assert extract_text_live(tmp_path / "x.jpg", engines=engines).engine == "ocrspace"


def test_live_ocr_skips_engines_without_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OCR_SPACE_API_KEY", raising=False)
    with patch("src.ocr.extract.extract_text_openai_vision") as openai_call, patch(
        "src.ocr.extract.extract_text_easyocr", return_value=_ok("easyocr")
    ):
        result = extract_text_live(tmp_path / "x.jpg", engines=("openai_vision", "ocrspace", "easyocr"))

    openai_call.assert_not_called()
    assert result.engine == "easyocr"


def test_live_ocr_returns_empty_result_when_no_engine_reads_text(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    with patch("src.ocr.extract.extract_text_openai_vision", return_value=_ok("openai_vision", "")), patch(
        "src.ocr.extract.extract_text_easyocr", return_value=_ok("easyocr", "  ")
    ):
        result = extract_text_live(tmp_path / "x.jpg", engines=("openai_vision", "easyocr"))

    assert result.engine == "easyocr"
    assert result.text.strip() == ""


def test_live_ocr_default_order_comes_from_config():
    from src.common.config import get_config

    assert get_config()["ocr"]["live_engines"][0] == "openai_vision"


def test_encode_image_for_vlm_downscales_large_photos(tmp_path):
    import base64

    import cv2
    import numpy as np

    path = tmp_path / "big.png"
    Image.new("RGB", (4000, 3000), color=(255, 255, 255)).save(path)

    mime, encoded = _encode_image_for_vlm(path, max_side=2048)

    decoded = cv2.imdecode(np.frombuffer(base64.b64decode(encoded), np.uint8), cv2.IMREAD_COLOR)
    assert mime == "image/jpeg"
    assert max(decoded.shape[:2]) == 2048


def test_encode_image_for_vlm_keeps_small_images_as_is(tmp_path):
    path = _make_text_image(tmp_path, "x")
    mime, _encoded = _encode_image_for_vlm(path, max_side=2048)
    assert mime == "image/png"
