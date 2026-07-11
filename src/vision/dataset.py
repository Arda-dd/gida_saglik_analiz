"""Gorsel siniflandirma (Faz 2) icin veri yukleme, bolme ve augmentation.

ONEMLI: Resize burada SADECE egitim/degerlendirme aninda, bellek-ici olarak yapilir
(albumentations transform'lari araciligiyla). Diskteki gorseller (data/raw/.../images/,
data/processed/images_hq/) hicbir zaman kalici olarak kucultulmez - bu ayrim Faz 1'de
belirlenen bir tasarim ilkesidir (bkz. src/data/image_preprocessing.py dokstring'i).
"""

from __future__ import annotations

from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Dataset

from src.common.config import get_config

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_category_list() -> list[str]:
    """config.yaml -> vision.categories (tek dogruluk kaynagi, label sirasini belirler)."""
    return get_config()["vision"]["categories"]


def load_vision_dataframe(csv_path: Path) -> pd.DataFrame:
    """dataset.csv'yi okur; gorsel siniflandirma icin uygun olmayan satirlari eler.

    Elenenler: image_path'i bos olanlar (henuz OCR/foto eslesmemis), diskte dosyasi
    olmayanlar, ve kategorisi 'bilinmiyor' olanlar (siniflandirma hedefi olamaz).
    """
    df = pd.read_csv(csv_path)
    df = df[df["image_path"].notna()]
    df = df[df["image_path"].apply(lambda p: Path(p).exists())]
    df = df[df["category"] != "bilinmiyor"]
    return df.reset_index(drop=True)


def stratified_train_test_split(
    df: pd.DataFrame, test_size: float = 0.15, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Kategori oranlarini koruyarak train/test bolme yapar."""
    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df["category"], random_state=random_state
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def get_kfold_indices(
    df: pd.DataFrame, n_splits: int = 5, random_state: int = 42
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Stratified k-fold cross-validation icin (train_idx, val_idx) ciftleri uretir."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(skf.split(df, df["category"]))


def get_train_transforms(image_size: int = 224) -> A.Compose:
    """Egitim icin augmentation + normalize + tensor donusumu (resize bellek-ici, gecicidir)."""
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def get_eval_transforms(image_size: int = 224) -> A.Compose:
    """Dogrulama/test icin augmentation olmadan resize + normalize + tensor donusumu."""
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


class LabelImageDataset(Dataset):
    """Etiket gorsellerini ve kategori etiketlerini PyTorch'a sunan Dataset sinifi."""

    def __init__(self, dataframe: pd.DataFrame, categories: list[str], transform: A.Compose | None = None):
        self.df = dataframe.reset_index(drop=True)
        self.categories = categories
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        try:
            buffer = np.fromfile(str(row["image_path"]), dtype=np.uint8)
            image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        except Exception:
            image = None

        if image is None:
            raise ValueError(f"Gorsel okunamadi: {row['image_path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = self.categories.index(row["category"])

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        return image, label
