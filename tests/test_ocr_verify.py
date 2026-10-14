from PIL import Image, ImageFilter

from src.common.schema import NutritionFacts
from src.ocr.verify import assess_image_quality, verify_nutrition


def test_verify_marks_values_found_in_reference_text():
    nutrition = NutritionFacts(sugar_g=35.0, salt_g=0.8, protein_g=6.3)
    primary = "Sucres 35 g | Sel 0,8 g | Protéines 6,3 g"
    reference = "Sucres 35g Proteines 6.3 g Sel 0,5 g"

    assert verify_nutrition(nutrition, primary, reference) == {
        "protein_g": True, "sugar_g": True, "salt_g": False,
    }


def test_verify_skips_derived_and_zero_fields():
    # sodium_mg tuzdan hesaplandi (metinde yok), fiber 0 - ikisi de kontrol edilmez.
    nutrition = NutritionFacts(salt_g=1.2, sodium_mg=480.0, fiber_g=0.0)
    assert verify_nutrition(nutrition, "Tuz 1,2 g Lif 0 g", "Tuz 1.2 g") == {"salt_g": True}


def test_quality_flags_low_resolution_and_blur(tmp_path):
    small = tmp_path / "small.png"
    Image.new("RGB", (400, 300), "white").save(small)
    warnings = assess_image_quality(small)
    assert any("cozunurlugu dusuk" in w for w in warnings)
    assert any("bulanik" in w for w in warnings)  # tek renk = hic kenar yok


def test_quality_passes_sharp_high_resolution_image(tmp_path):
    import numpy as np

    rng = np.random.default_rng(0)
    sharp = tmp_path / "sharp.png"
    Image.fromarray(rng.integers(0, 255, (1200, 1600), dtype=np.uint8)).save(sharp)
    assert assess_image_quality(sharp) == []

    blurry = tmp_path / "blurry.png"
    Image.open(sharp).filter(ImageFilter.GaussianBlur(12)).save(blurry)
    assert any("bulanik" in w for w in assess_image_quality(blurry))


def test_quality_missing_file_returns_no_warnings(tmp_path):
    assert assess_image_quality(tmp_path / "yok.jpg") == []
