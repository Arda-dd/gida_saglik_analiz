from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from src.barcode.detect import barcodes_from_text, decode_barcodes, is_valid_gtin
from src.barcode.lookup import BarcodeProduct, fetch_off_product, lookup_barcode
from src.barcode.merge import merge_nutrition
from src.common.schema import Allergen, NutritionFacts, ProductCategory, ProductRecord

# --- EAN-13 cizimi (cozucuyu gercek bir barkodla test etmek icin) ---
_L = ["0001101", "0011001", "0010011", "0111101", "0100011", "0110001", "0101111", "0111011", "0110111", "0001011"]
_G = ["0100111", "0110011", "0011011", "0100001", "0011101", "0111001", "0000101", "0010001", "0001001", "0010111"]
_R = ["1110010", "1100110", "1101100", "1000010", "1011100", "1001110", "1010000", "1000100", "1001000", "1110100"]
_PARITY = ["LLLLLL", "LLGLGG", "LLGGLG", "LLGGGL", "LGLLGG", "LGGLLG", "LGGGLL", "LGLGLG", "LGLGGL", "LGGLGL"]


def _ean13_image(code: str, path, module_px: int = 3, height: int = 150):
    digits = [int(c) for c in code]
    bits = "101"
    for digit, parity in zip(digits[1:7], _PARITY[digits[0]]):
        bits += (_L if parity == "L" else _G)[digit]
    bits += "01010"
    for digit in digits[7:]:
        bits += _R[digit]
    bits += "101"
    quiet = 12
    row = np.where(np.array(list("0" * quiet + bits + "0" * quiet)) == "1", 0, 255).astype(np.uint8)
    image = np.repeat(np.repeat(row[np.newaxis, :], height, axis=0), module_px, axis=1)
    image = np.pad(image, 40, constant_values=255)
    Image.fromarray(image).convert("RGB").save(path)
    return path


def test_is_valid_gtin():
    assert is_valid_gtin("3046920022606")  # EAN-13
    assert is_valid_gtin("6111252421568")
    assert not is_valid_gtin("3046920022607")  # sagla hanesi yanlis
    assert not is_valid_gtin("12345")


def test_barcodes_from_text_reads_spaced_digits_and_rejects_noise():
    text = "Ingredients ... 6 111252 421568\nEnergie 1962 kJ / 467 kcal\n3046920022607"
    assert barcodes_from_text(text) == ["6111252421568"]


def test_barcodes_from_text_ignores_8_digit_codes_from_noisy_ocr():
    # EasyOCR'in bozuk metninden sagla hanesi tutan ama yanlis 8 haneli kodlar cikmisti.
    assert barcodes_from_text("WMsniien 69914959 falee") == []


def test_decode_barcodes_reads_real_ean13_image(tmp_path):
    path = _ean13_image("3046920022606", tmp_path / "ean.png")
    assert decode_barcodes(path) == ["3046920022606"]


def test_decode_barcodes_missing_or_blank_image(tmp_path):
    assert decode_barcodes(tmp_path / "yok.jpg") == []
    blank = tmp_path / "blank.png"
    Image.new("RGB", (300, 200), "white").save(blank)
    assert decode_barcodes(blank) == []


def test_merge_prefers_barcode_fills_gaps_and_reports_conflicts():
    barcode = NutritionFacts(energy_kcal=467, sugar_g=32, fat_g=None)
    ocr = NutritionFacts(energy_kcal=470, sugar_g=12, fat_g=17)

    merged, filled, conflicts = merge_nutrition(barcode, ocr)

    assert merged.energy_kcal == 467  # uyumlu: veritabani degeri
    assert merged.sugar_g == 32  # celiskili: veritabani degeri + uyari
    assert merged.fat_g == 17  # veritabaninda yok: OCR'dan
    assert filled == ["fat_g"]
    assert conflicts == ["seker: etiket 12, veritabani 32"]


def _response(status_code: int, payload: dict | None = None):
    response = MagicMock(status_code=status_code)
    response.json.return_value = payload or {}
    response.raise_for_status.side_effect = None if status_code < 400 or status_code == 404 else RuntimeError(status_code)
    return response


def test_fetch_off_product_not_found_returns_none():
    session = MagicMock()
    session.get.return_value = _response(200, {"status": 0})
    assert fetch_off_product("0000000000000", session=session) is None
    session.get.return_value = _response(404)
    assert fetch_off_product("0000000000000", session=session) is None


def test_lookup_barcode_maps_off_product_and_traces():
    lookup_barcode.cache_clear()
    product = {
        "code": "3046920022606",
        "product_name": "Excellence 85% cacao",
        "nutriments": {"energy-kcal_100g": 592, "fat_100g": 55, "sugars_100g": 7, "proteins_100g": 10},
        "categories_tags": ["en:snacks", "en:chocolates"],
        "allergens_tags": ["en:milk"],
        "traces_tags": ["en:nuts"],
    }
    with patch("src.barcode.lookup.fetch_off_product", return_value=product):
        result = lookup_barcode("3046920022606")

    assert result.product_name == "Excellence 85% cacao"
    assert result.record.nutrition.energy_kcal == 592
    assert result.has_usable_nutrition
    assert Allergen.LAKTOZ in result.record.allergens
    assert Allergen.FINDIK in result.traces


def test_barcode_product_with_too_few_fields_is_not_usable():
    record = ProductRecord(product_id="off_1", nutrition=NutritionFacts(energy_kcal=100))
    assert not BarcodeProduct(barcode="1", record=record).has_usable_nutrition
