from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

from src.ocr.extract import OCRResult, extract_text_easyocr, extract_text_tesseract, TESSERACT_EXE

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


# --- PaddleOCR / PP-Structure ---------------------------------------------------
# Bu testler GERCEK modeli yuklemez (agir + ag gerektirir); PP-Structure boru hatti
# mock'lanir. Boylece paketin geri kalani gibi bu testler de bagimsiz calisir.

from src.ocr.extract import (  # noqa: E402
    _ensure_paddle_env,
    _table_html_to_lines,
    extract_text_paddleocr,
)

TWO_COLUMN_HTML = (
    "<html><body><table>"
    "<tr><td>Enerji</td><td>566 kcal</td></tr>"
    "<tr><td>Seker</td><td>30 g</td></tr>"
    "</table></body></html>"
)


def test_table_html_to_lines_pairs_label_with_value():
    """PP-Structure'in asil kazanci: etiket ve deger AYNI satirda kalir."""
    lines = _table_html_to_lines(TWO_COLUMN_HTML).splitlines()

    assert lines == ["Enerji 566 kcal", "Seker 30 g"]


def test_table_html_to_lines_keeps_column_order_in_multi_column_table():
    """'Per 100g' ve 'Per porsiyon' sutunlari yan yana ise soldan saga sira korunur -
    normalize.py ilk sayiyi aldigindan 100g sutunu dogal olarak tercih edilir."""
    html = (
        "<table><tr><td>Enerji</td><td>159 kcal</td><td>300 kcal</td></tr></table>"
    )

    assert _table_html_to_lines(html) == "Enerji 159 kcal 300 kcal"


def test_table_html_to_lines_ignores_empty_cells_and_returns_empty_for_no_table():
    html = "<table><tr><td></td><td>  </td></tr></table>"

    assert _table_html_to_lines(html) == ""
    assert _table_html_to_lines("<p>tablo yok</p>") == ""


def test_ensure_paddle_env_disables_mkldnn():
    """oneDNN + PIR yurutucu uyumsuzlugu layout modelinde cokmeye yol aciyordu."""
    import os

    os.environ.pop("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", None)
    _ensure_paddle_env()

    assert os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] == "False"


def test_extract_text_paddleocr_raises_on_missing_file(tmp_path):
    with pytest.raises(ValueError):
        extract_text_paddleocr(tmp_path / "yok.png")


class _FakeResult:
    def __init__(self, payload):
        self.json = {"res": payload}


class _FakePipeline:
    def __init__(self, payload):
        self._payload = payload

    def predict(self, _path):
        return [_FakeResult(self._payload)]


def _patch_pipeline(monkeypatch, payload):
    monkeypatch.setattr(
        "src.ocr.extract._get_pp_structure", lambda: _FakePipeline(payload)
    )


def test_extract_text_paddleocr_puts_table_rows_before_page_text(monkeypatch, tmp_path):
    """Tablo satirlari EN BASTA olmali: normalize.py ilk eslesmeyi aldigi icin
    degerler sayfanin dagilmis metninden degil, yapisi korunmus tablodan gelmeli."""
    image_path = tmp_path / "label.png"
    image_path.write_bytes(b"stub")

    _patch_pipeline(
        monkeypatch,
        {
            "table_res_list": [{"pred_html": TWO_COLUMN_HTML}],
            "overall_ocr_res": {
                "rec_texts": ["POUR 100g", "Enerji", "566 kcal"],
                "rec_scores": [0.9, 0.8, 1.0],
                "rec_boxes": [[0, 0, 50, 10], [0, 20, 50, 30], [60, 20, 110, 30]],
            },
        },
    )

    result = extract_text_paddleocr(image_path)
    lines = result.text.splitlines()

    assert result.engine == "paddleocr"
    assert lines[0] == "Enerji 566 kcal"
    assert lines[1] == "Seker 30 g"
    # Baz belirteci ("100g") tablonun DISINDA kalabilir - sayfa metni de eklenmeli.
    assert "POUR 100g" in result.text


def test_extract_text_paddleocr_scales_confidence_to_0_100(monkeypatch, tmp_path):
    """rec_scores 0-1 araliginda gelir; OCRResult sozlesmesi 0-100 bekler."""
    image_path = tmp_path / "label.png"
    image_path.write_bytes(b"stub")

    _patch_pipeline(
        monkeypatch,
        {
            "table_res_list": [],
            "overall_ocr_res": {"rec_texts": ["a", "b"], "rec_scores": [0.5, 0.9]},
        },
    )

    result = extract_text_paddleocr(image_path)

    assert result.mean_confidence == pytest.approx(70.0)


def test_extract_text_paddleocr_handles_empty_prediction(monkeypatch, tmp_path):
    image_path = tmp_path / "label.png"
    image_path.write_bytes(b"stub")
    monkeypatch.setattr("src.ocr.extract._get_pp_structure", lambda: _FakePipeline(None))
    monkeypatch.setattr(
        "src.ocr.extract._get_pp_structure",
        lambda: type("P", (), {"predict": lambda self, _p: []})(),
    )

    result = extract_text_paddleocr(image_path)

    assert result.text == ""
    assert result.mean_confidence == 0.0


def test_extract_text_dispatches_to_engine_from_config(monkeypatch):
    """Motor degistirmek icin kod degil config yeterli olmali."""
    import src.ocr.extract as extract_module

    called = {}

    def _fake(path):
        called["path"] = path
        return OCRResult(text="ok", mean_confidence=1.0, engine="easyocr")

    monkeypatch.setitem(extract_module.OCR_ENGINES, "easyocr", _fake)
    monkeypatch.setattr(
        extract_module, "get_config", lambda: {"ocr": {"engine": "easyocr"}}
    )

    result = extract_module.extract_text(Path("x.png"))

    assert result.engine == "easyocr"
    assert called["path"] == Path("x.png")


def test_extract_text_rejects_unknown_engine(monkeypatch):
    import src.ocr.extract as extract_module

    monkeypatch.setattr(
        extract_module, "get_config", lambda: {"ocr": {"engine": "yok_boyle_motor"}}
    )

    with pytest.raises(ValueError, match="Bilinmeyen OCR motoru"):
        extract_module.extract_text(Path("x.png"))
