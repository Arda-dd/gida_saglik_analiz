import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.vision.dataset import LabelImageDataset, get_eval_transforms, get_train_transforms
from src.vision.model import build_model
from src.vision.train import evaluate, run_kfold_cv, train_model, train_one_epoch

CATEGORIES = ["sut_urunu", "atistirmalik", "icecek"]


def _make_synthetic_df(tmp_path, n_per_category=4):
    rows = []
    for cat in CATEGORIES:
        base = 100 if cat == "sut_urunu" else 150
        for i in range(n_per_category):
            img = np.clip(np.random.normal(base, 10, (40, 40, 3)), 0, 255).astype(np.uint8)
            path = tmp_path / f"{cat}_{i}.jpg"
            ret, buf = cv2.imencode(".jpg", img)
            if not ret:
                raise ValueError("imencode failed")
            buf.tofile(str(path))
            rows.append({"product_id": f"{cat}_{i}", "category": cat, "image_path": str(path)})
    return pd.DataFrame(rows)


def test_train_one_epoch_returns_finite_loss(tmp_path):
    df = _make_synthetic_df(tmp_path)
    ds = LabelImageDataset(df, CATEGORIES, transform=get_train_transforms(image_size=32))
    loader = DataLoader(ds, batch_size=4, shuffle=True)

    model = build_model("mobilenetv3", num_classes=len(CATEGORIES), pretrained=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    loss = train_one_epoch(model, loader, optimizer, criterion, device="cpu")
    assert np.isfinite(loss)
    assert loss > 0


def test_evaluate_returns_accuracy_and_f1_in_valid_range(tmp_path):
    df = _make_synthetic_df(tmp_path)
    ds = LabelImageDataset(df, CATEGORIES, transform=get_eval_transforms(image_size=32))
    loader = DataLoader(ds, batch_size=4, shuffle=False)

    model = build_model("mobilenetv3", num_classes=len(CATEGORIES), pretrained=False)
    metrics = evaluate(model, loader, device="cpu")

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1_macro"] <= 1.0


def test_train_model_improves_or_maintains_best_f1_over_epochs(tmp_path):
    df = _make_synthetic_df(tmp_path, n_per_category=8)
    train_df = df.iloc[:18].reset_index(drop=True)
    val_df = df.iloc[18:].reset_index(drop=True)

    model, result = train_model(
        "mobilenetv3", train_df, val_df, CATEGORIES, epochs=2, batch_size=4, device="cpu", pretrained=False
    )

    assert isinstance(model, torch.nn.Module)
    assert len(result["history"]) == 2
    assert result["best_f1_macro"] == max(h["f1_macro"] for h in result["history"])


def test_run_kfold_cv_produces_correct_fold_count(tmp_path):
    df = _make_synthetic_df(tmp_path, n_per_category=6)  # 18 satir, 3 kategori

    cv_result = run_kfold_cv(
        "mobilenetv3", df, CATEGORIES, n_splits=3, epochs=1, device="cpu", pretrained=False
    )

    assert len(cv_result["fold_metrics"]) == 3
    assert 0.0 <= cv_result["mean_accuracy"] <= 1.0
    assert 0.0 <= cv_result["mean_f1_macro"] <= 1.0
    assert "std_accuracy" in cv_result and "std_f1_macro" in cv_result
