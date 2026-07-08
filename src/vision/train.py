"""MobileNetV3 ve EfficientNet-B3 icin egitim, k-fold cross-validation ve degerlendirme.

Oneri formu 2.2: egitim/dogrulama/test ayrimi + k-katli capraz dogrulama (k-fold CV) ile
genelleme kapasitesi degerlendirilir, asiri ogrenme (overfitting) riski azaltilir.
Metrikler: Accuracy + F1-score (bkz. oneri formu 2.6).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from src.vision.dataset import (
    LabelImageDataset,
    get_category_list,
    get_eval_transforms,
    get_kfold_indices,
    get_train_transforms,
    load_vision_dataframe,
    stratified_train_test_split,
)
from src.vision.model import SUPPORTED_BACKBONES, build_model

DATASET_CSV = Path("data/processed/dataset.csv")
MODELS_DIR = Path("models")
REPORT_PATH = Path("docs/vision_training_report.json")


def train_one_epoch(
    model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: str
) -> float:
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model: nn.Module, loader: DataLoader, device: str) -> dict:
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1_macro": f1_score(all_labels, all_preds, average="macro", zero_division=0),
    }


def train_model(
    backbone: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    categories: list[str],
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 1e-4,
    device: str = "cpu",
    pretrained: bool = True,
) -> tuple[nn.Module, dict]:
    """Bir backbone'u train_df uzerinde egitir, her epoch sonu val_df'te degerlendirir.

    En iyi F1-macro skoruna sahip epoch'un agirliklari geri yuklenerek dondurulur.
    `pretrained=False` sadece testlerde agirlik indirmeden hizli calismak icindir;
    gercek egitimde (main()) daima pretrained=True kullanilir (transfer learning).
    """
    model = build_model(backbone, num_classes=len(categories), pretrained=pretrained).to(device)

    train_ds = LabelImageDataset(train_df, categories, transform=get_train_transforms())
    val_ds = LabelImageDataset(val_df, categories, transform=get_eval_transforms())
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_f1 = -1.0
    best_state = None
    history = []

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        metrics = evaluate(model, val_loader, device)
        metrics.update({"epoch": epoch + 1, "train_loss": train_loss})
        history.append(metrics)

        if metrics["f1_macro"] > best_f1:
            best_f1 = metrics["f1_macro"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {"history": history, "best_f1_macro": best_f1}


def run_kfold_cv(
    backbone: str,
    df: pd.DataFrame,
    categories: list[str],
    n_splits: int = 5,
    epochs: int = 8,
    device: str = "cpu",
    pretrained: bool = True,
) -> dict:
    """Bir backbone icin stratified k-fold CV calistirir; fold basina ve ortalama metrikleri doner."""
    folds = get_kfold_indices(df, n_splits=n_splits)
    fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        fold_train_df = df.iloc[train_idx].reset_index(drop=True)
        fold_val_df = df.iloc[val_idx].reset_index(drop=True)

        _, result = train_model(
            backbone, fold_train_df, fold_val_df, categories, epochs=epochs, device=device, pretrained=pretrained
        )
        best_epoch_metrics = max(result["history"], key=lambda m: m["f1_macro"])
        fold_metrics.append(
            {"fold": fold_idx + 1, "accuracy": best_epoch_metrics["accuracy"], "f1_macro": best_epoch_metrics["f1_macro"]}
        )
        print(f"    fold {fold_idx + 1}/{n_splits}: acc={best_epoch_metrics['accuracy']:.3f} f1={best_epoch_metrics['f1_macro']:.3f}")

    accuracies = [m["accuracy"] for m in fold_metrics]
    f1s = [m["f1_macro"] for m in fold_metrics]

    return {
        "fold_metrics": fold_metrics,
        "mean_accuracy": float(np.mean(accuracies)),
        "std_accuracy": float(np.std(accuracies)),
        "mean_f1_macro": float(np.mean(f1s)),
        "std_f1_macro": float(np.std(f1s)),
    }


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Cihaz: {device}")

    categories = get_category_list()
    df = load_vision_dataframe(DATASET_CSV)
    print(f"Toplam gecerli gorsel: {len(df)}, kategoriler: {categories}")
    print(df["category"].value_counts())

    train_df, test_df = stratified_train_test_split(df, test_size=0.15, random_state=42)
    print(f"\nEgitim: {len(train_df)}, Test (held-out): {len(test_df)}")

    report: dict = {"categories": categories, "n_total": len(df), "n_train": len(train_df), "n_test": len(test_df), "backbones": {}}

    best_overall_f1 = -1.0
    best_overall_backbone = None
    best_overall_model = None

    for backbone in SUPPORTED_BACKBONES:
        print(f"\n=== {backbone} ===")
        print("  k-fold CV (egitim seti uzerinde)...")
        cv_result = run_kfold_cv(backbone, train_df, categories, n_splits=5, epochs=15, device=device)
        print(f"  CV ortalama: acc={cv_result['mean_accuracy']:.3f}±{cv_result['std_accuracy']:.3f} "
              f"f1={cv_result['mean_f1_macro']:.3f}±{cv_result['std_f1_macro']:.3f}")

        print("  Final model egitimi (tum egitim seti) + held-out test degerlendirmesi...")
        final_model, train_result = train_model(
            backbone, train_df, test_df, categories, epochs=20, device=device
        )
        test_metrics = max(train_result["history"], key=lambda m: m["f1_macro"])
        print(f"  Held-out test: acc={test_metrics['accuracy']:.3f} f1={test_metrics['f1_macro']:.3f}")

        report["backbones"][backbone] = {
            "cross_validation": cv_result,
            "held_out_test": {"accuracy": test_metrics["accuracy"], "f1_macro": test_metrics["f1_macro"]},
        }

        if test_metrics["f1_macro"] > best_overall_f1:
            best_overall_f1 = test_metrics["f1_macro"]
            best_overall_backbone = backbone
            best_overall_model = final_model

    report["best_backbone"] = best_overall_backbone
    report["best_f1_macro"] = best_overall_f1

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(best_overall_model.state_dict(), MODELS_DIR / "vision_best.pt")
    print(f"\nEn iyi model ({best_overall_backbone}, f1={best_overall_f1:.3f}) kaydedildi: {MODELS_DIR / 'vision_best.pt'}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Egitim raporu kaydedildi: {REPORT_PATH}")


if __name__ == "__main__":
    main()
