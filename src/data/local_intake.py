"""Yerel market etiket fotograflarini organize eden intake araci.

Arda ve Semih'in fiziksel olarak cektigi fotograflari `inbox/`'a atmasi, bu
modulun EXIF/GPS temizleyip (anonimlestirme), kalite kontrolunden gecirip,
anonim product_id atayarak kategoriye gore klasorlemesi icindir. Fiziksel
fotograf cekimi bu koddan bagimsiz bir insan gorevidir - bkz.
docs/local_data_collection_protocol.md.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import pandas as pd
from PIL import Image, ImageOps

from src.common.schema import ProductCategory

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass
class QualityReport:
    is_valid: bool
    blur_score: float
    width: int
    height: int
    reasons: list[str] = field(default_factory=list)


def anonymize_image_exif(src_path: Path, dest_path: Path) -> Path:
    """EXIF/GPS/cihaz metadata'sini siler; fiziksel dondurmeyi (orientation) once uygular.

    Not: Marka/logo GORSELDEN kirpilmaz (urun bilgisi olarak kalmasi gerekir) -
    sadece kisisel/cihaz metadata'si (konum, cihaz modeli, cekim zamani vb.) silinir.
    """
    image = Image.open(src_path)
    image = ImageOps.exif_transpose(image)  # once fiziksel donusu uygula
    image = image.convert("RGB")  # yeniden olusturulan nesne EXIF tasimaz

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(dest_path, format="JPEG", quality=95)
    return dest_path


def check_image_quality(
    image_path: Path,
    min_width: int = 1600,
    min_height: int = 1600,
    blur_var_threshold: float = 120.0,
) -> QualityReport:
    """Cozunurluk ve bulaniklik (Laplacian varyansi) kontrolu yapar."""
    import numpy as np
    try:
        buffer = np.fromfile(str(image_path), dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    except Exception:
        image = None

    if image is None:
        return QualityReport(is_valid=False, blur_score=0.0, width=0, height=0, reasons=["okunamadi"])

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    reasons: list[str] = []
    if width < min_width or height < min_height:
        reasons.append(f"cozunurluk_dusuk({width}x{height})")
    if blur_score < blur_var_threshold:
        reasons.append(f"bulanik(blur_score={blur_score:.1f})")

    return QualityReport(is_valid=not reasons, blur_score=blur_score, width=width, height=height, reasons=reasons)


def assign_anonymous_product_id(category: ProductCategory, counter_state: dict[str, int]) -> str:
    """Format: local_{category}_{seq:04d}. counter_state cagiran tarafindan yonetilir/kalicilastirilir."""
    key = category.value
    counter_state[key] = counter_state.get(key, 0) + 1
    return f"local_{key}_{counter_state[key]:04d}"


def init_counter_state_from_existing(raw_dir: Path) -> dict[str, int]:
    """raw_dir altindaki mevcut dosyalardan devam edecek sekilde sayaci baslatir."""
    counter_state: dict[str, int] = {}
    pattern = re.compile(r"^local_(?P<category>[a-z_]+)_(?P<seq>\d{4})$")

    if not raw_dir.exists():
        return counter_state

    for category_dir in raw_dir.iterdir():
        if not category_dir.is_dir():
            continue
        max_seq = 0
        for file in category_dir.glob("local_*.jpg"):
            match = pattern.match(file.stem)
            if match:
                max_seq = max(max_seq, int(match.group("seq")))
        if max_seq > 0:
            counter_state[category_dir.name] = max_seq

    return counter_state


def organize_raw_image(
    src_path: Path, category: ProductCategory, raw_dir: Path, counter_state: dict[str, int]
) -> Path:
    """Bir ham fotografi anonimlestirip anonim ID ile kategori klasorune tasir."""
    product_id = assign_anonymous_product_id(category, counter_state)
    dest_path = raw_dir / category.value / f"{product_id}.jpg"
    return anonymize_image_exif(src_path, dest_path)


def build_intake_manifest(raw_local_dir: Path) -> pd.DataFrame:
    """raw_local_dir altindaki tum organize edilmis gorselleri tarayip kalite manifestosu uretir."""
    rows = []
    if raw_local_dir.exists():
        for category_dir in sorted(raw_local_dir.iterdir()):
            if not category_dir.is_dir():
                continue
            for image_path in sorted(category_dir.glob("*.jpg")):
                report = check_image_quality(image_path)
                rows.append(
                    {
                        "product_id": image_path.stem,
                        "category": category_dir.name,
                        "image_path": str(image_path),
                        "width": report.width,
                        "height": report.height,
                        "blur_score": report.blur_score,
                        "is_valid": report.is_valid,
                        "reasons": ";".join(report.reasons),
                    }
                )

    return pd.DataFrame(
        rows,
        columns=["product_id", "category", "image_path", "width", "height", "blur_score", "is_valid", "reasons"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Yerel market etiket fotograflarini organize eder.")
    parser.add_argument("--inbox", required=True, type=Path, help="Ham fotograflarin oldugu klasor")
    parser.add_argument(
        "--category",
        required=True,
        choices=[c.value for c in ProductCategory],
        help="Bu inbox'taki tum fotograflarin kategorisi",
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=Path("data/raw/local"), help="Organize edilmis cikti klasoru"
    )
    args = parser.parse_args()

    category = ProductCategory(args.category)
    counter_state = init_counter_state_from_existing(args.raw_dir)

    image_files = [p for p in sorted(args.inbox.iterdir()) if p.suffix.lower() in IMAGE_EXTENSIONS]
    print(f"{len(image_files)} gorsel bulundu, '{category.value}' kategorisine organize ediliyor...")

    for src_path in image_files:
        dest_path = organize_raw_image(src_path, category, args.raw_dir, counter_state)
        print(f"  {src_path.name} -> {dest_path}")

    manifest = build_intake_manifest(args.raw_dir)
    invalid_count = (~manifest["is_valid"]).sum() if not manifest.empty else 0
    print(f"Toplam organize edilmis gorsel: {len(manifest)}, kalite sorunlu: {invalid_count}")


if __name__ == "__main__":
    main()
