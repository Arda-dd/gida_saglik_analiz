import cv2
import numpy as np
import pytest
from PIL import Image

from src.common.schema import ProductCategory
from src.data.local_intake import (
    anonymize_image_exif,
    assign_anonymous_product_id,
    build_intake_manifest,
    check_image_quality,
    init_counter_state_from_existing,
    organize_raw_image,
)


def _make_jpeg_with_exif(path, size=(200, 150)) -> None:
    image = Image.new("RGB", size, color=(120, 130, 140))
    exif = image.getexif()
    exif[271] = "TestCameraMake"  # Make tag
    exif[34853] = None  # GPSInfo tag placeholder (bazi Pillow surumlerinde desteklenmeyebilir)
    image.save(path, format="JPEG", exif=exif)


def test_anonymize_image_exif_strips_metadata(tmp_path):
    src = tmp_path / "with_exif.jpg"
    dest = tmp_path / "clean.jpg"
    _make_jpeg_with_exif(src)

    original = Image.open(src)
    assert len(original.getexif()) > 0  # onkosul: gercekten exif var

    anonymize_image_exif(src, dest)
    cleaned = Image.open(dest)
    assert len(cleaned.getexif()) == 0


def test_check_image_quality_rejects_low_resolution(tmp_path):
    path = tmp_path / "small.jpg"
    small_image = np.full((100, 100, 3), 128, dtype=np.uint8)
    cv2.imwrite(str(path), small_image)

    report = check_image_quality(path, min_width=1600, min_height=1600)

    assert report.is_valid is False
    assert any("cozunurluk_dusuk" in r for r in report.reasons)


def test_check_image_quality_rejects_blurry_image(tmp_path):
    path = tmp_path / "blurry.jpg"
    rng = np.random.default_rng(1)
    sharp = rng.integers(0, 255, (1800, 1800, 3), dtype=np.uint8)
    blurry = cv2.GaussianBlur(sharp, (51, 51), 0)
    cv2.imwrite(str(path), blurry)

    report = check_image_quality(path, min_width=1600, min_height=1600, blur_var_threshold=120.0)

    assert report.is_valid is False
    assert any("bulanik" in r for r in report.reasons)


def test_check_image_quality_accepts_sharp_high_res_image(tmp_path):
    path = tmp_path / "sharp.jpg"
    rng = np.random.default_rng(2)
    sharp = rng.integers(0, 255, (1800, 1800, 3), dtype=np.uint8)  # yuksek frekansli gurultu = "keskin"
    cv2.imwrite(str(path), sharp)

    report = check_image_quality(path, min_width=1600, min_height=1600, blur_var_threshold=120.0)

    assert report.is_valid is True
    assert report.reasons == []


def test_assign_anonymous_product_id_format_and_uniqueness():
    counter_state: dict[str, int] = {}
    id1 = assign_anonymous_product_id(ProductCategory.SUT_URUNU, counter_state)
    id2 = assign_anonymous_product_id(ProductCategory.SUT_URUNU, counter_state)
    id3 = assign_anonymous_product_id(ProductCategory.ICECEK, counter_state)

    assert id1 == "local_sut_urunu_0001"
    assert id2 == "local_sut_urunu_0002"
    assert id3 == "local_icecek_0001"
    assert len({id1, id2, id3}) == 3


def test_organize_raw_image_moves_and_anonymizes(tmp_path):
    src = tmp_path / "inbox" / "photo1.jpg"
    src.parent.mkdir(parents=True)
    _make_jpeg_with_exif(src, size=(1700, 1700))

    raw_dir = tmp_path / "raw_local"
    counter_state: dict[str, int] = {}

    dest = organize_raw_image(src, ProductCategory.ICECEK, raw_dir, counter_state)

    assert dest.exists()
    assert dest.name == "local_icecek_0001.jpg"
    assert dest.parent.name == "icecek"
    assert len(Image.open(dest).getexif()) == 0


def test_init_counter_state_from_existing_continues_numbering(tmp_path):
    raw_dir = tmp_path / "raw_local"
    category_dir = raw_dir / "icecek"
    category_dir.mkdir(parents=True)
    (category_dir / "local_icecek_0001.jpg").write_bytes(b"fake")
    (category_dir / "local_icecek_0003.jpg").write_bytes(b"fake")

    counter_state = init_counter_state_from_existing(raw_dir)
    assert counter_state["icecek"] == 3


def test_init_counter_state_from_existing_empty_dir_returns_empty(tmp_path):
    counter_state = init_counter_state_from_existing(tmp_path / "does_not_exist")
    assert counter_state == {}


def test_build_intake_manifest_has_expected_columns(tmp_path):
    raw_dir = tmp_path / "raw_local"
    category_dir = raw_dir / "icecek"
    category_dir.mkdir(parents=True)

    image = np.full((1700, 1700, 3), 128, dtype=np.uint8)
    cv2.imwrite(str(category_dir / "local_icecek_0001.jpg"), image)

    manifest = build_intake_manifest(raw_dir)

    expected_columns = {
        "product_id", "category", "image_path", "width", "height", "blur_score", "is_valid", "reasons"
    }
    assert set(manifest.columns) == expected_columns
    assert len(manifest) == 1
    assert manifest.iloc[0]["product_id"] == "local_icecek_0001"


def test_build_intake_manifest_empty_dir_returns_empty_dataframe(tmp_path):
    manifest = build_intake_manifest(tmp_path / "does_not_exist")
    assert manifest.empty
