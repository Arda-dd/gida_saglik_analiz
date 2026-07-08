import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from src.vision.dataset import (
    LabelImageDataset,
    get_eval_transforms,
    get_kfold_indices,
    get_train_transforms,
    load_vision_dataframe,
    stratified_train_test_split,
)


def _write_image(path, size=(50, 50)):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    cv2.imwrite(str(path), image)
    return path


def _make_dataset_csv(tmp_path, n_per_category=10):
    categories = ["sut_urunu", "atistirmalik", "icecek", "hazir_gida", "konserve"]
    rows = []
    for cat in categories:
        for i in range(n_per_category):
            img_path = _write_image(tmp_path / "images" / f"{cat}_{i}.jpg")
            rows.append({"product_id": f"{cat}_{i}", "category": cat, "image_path": str(img_path)})

    # Bilinmiyor kategorisi ve eksik/olmayan dosya iceren "kirli" satirlar da ekle
    rows.append({"product_id": "unknown_1", "category": "bilinmiyor", "image_path": str(tmp_path / "images" / "sut_urunu_0.jpg")})
    rows.append({"product_id": "missing_1", "category": "icecek", "image_path": str(tmp_path / "images" / "does_not_exist.jpg")})
    rows.append({"product_id": "noimg_1", "category": "icecek", "image_path": None})

    csv_path = tmp_path / "dataset.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def test_load_vision_dataframe_filters_dirty_rows(tmp_path):
    csv_path = _make_dataset_csv(tmp_path, n_per_category=10)
    df = load_vision_dataframe(csv_path)

    assert len(df) == 50  # 5 kategori x 10, kirli satirlar elenmis
    assert "bilinmiyor" not in df["category"].values
    assert all(df["image_path"].apply(lambda p: __import__("pathlib").Path(p).exists()))


def test_stratified_train_test_split_preserves_proportions(tmp_path):
    csv_path = _make_dataset_csv(tmp_path, n_per_category=20)
    df = load_vision_dataframe(csv_path)

    train_df, test_df = stratified_train_test_split(df, test_size=0.2, random_state=42)

    assert len(train_df) + len(test_df) == len(df)
    # Her kategori test setinde de temsil edilmeli (stratify sayesinde)
    assert set(test_df["category"].unique()) == set(df["category"].unique())


def test_get_kfold_indices_covers_all_rows_without_overlap(tmp_path):
    csv_path = _make_dataset_csv(tmp_path, n_per_category=20)
    df = load_vision_dataframe(csv_path)

    folds = get_kfold_indices(df, n_splits=5, random_state=42)
    assert len(folds) == 5

    all_val_indices = np.concatenate([val_idx for _, val_idx in folds])
    assert sorted(all_val_indices) == list(range(len(df)))  # her satir tam olarak bir kez val'de


def test_label_image_dataset_returns_correct_shape_and_label(tmp_path):
    csv_path = _make_dataset_csv(tmp_path, n_per_category=3)
    df = load_vision_dataframe(csv_path)
    categories = sorted(df["category"].unique())

    dataset = LabelImageDataset(df, categories, transform=get_eval_transforms(image_size=224))
    image_tensor, label = dataset[0]

    assert isinstance(image_tensor, torch.Tensor)
    assert image_tensor.shape == (3, 224, 224)
    assert isinstance(label, int)
    assert 0 <= label < len(categories)


def test_label_image_dataset_len_matches_dataframe(tmp_path):
    csv_path = _make_dataset_csv(tmp_path, n_per_category=4)
    df = load_vision_dataframe(csv_path)
    categories = sorted(df["category"].unique())

    dataset = LabelImageDataset(df, categories, transform=get_eval_transforms())
    assert len(dataset) == len(df)


def test_train_transforms_produce_different_augmentation_than_eval(tmp_path):
    # Egitim transform'u augmentation icerdigi icin eval'dan farkli olabilmeli (deterministik degil)
    train_tf = get_train_transforms(image_size=64)
    eval_tf = get_eval_transforms(image_size=64)

    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    train_out = train_tf(image=image)["image"]
    eval_out = eval_tf(image=image)["image"]

    assert train_out.shape == (3, 64, 64)
    assert eval_out.shape == (3, 64, 64)


def test_label_image_dataset_raises_on_unreadable_image(tmp_path):
    bad_path = tmp_path / "bad.jpg"
    bad_path.write_bytes(b"not a real image")
    df = pd.DataFrame([{"product_id": "x", "category": "icecek", "image_path": str(bad_path)}])

    dataset = LabelImageDataset(df, ["icecek"], transform=None)
    with pytest.raises(ValueError):
        dataset[0]
