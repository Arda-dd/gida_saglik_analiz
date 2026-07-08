"""Egitilmis gorsel siniflandirma modeliyle tek goruntu uzerinde tahmin (inference).

Bu modul, api/pipeline.py (Faz 6) icinde etiket fotografindan urun kategorisini
belirlemek icin kullanilacaktir (bkz. oneri formu 2.2 - kategori, sonraki asamalardaki
semantik baglami olusturur).
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import torch

from src.vision.dataset import get_eval_transforms
from src.vision.model import build_model

DEFAULT_CHECKPOINT = Path("models/vision_best.pt")
DEFAULT_REPORT = Path("docs/vision_training_report.json")


def load_trained_model(
    checkpoint_path: Path, backbone: str, num_classes: int, device: str = "cpu"
) -> torch.nn.Module:
    """Egitilmis agirliklari (state_dict) bir model iskeletine yukler, eval moduna alir."""
    model = build_model(backbone, num_classes=num_classes, pretrained=False)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def predict_category(
    model: torch.nn.Module,
    image_path: Path,
    categories: list[str],
    device: str = "cpu",
    image_size: int = 224,
) -> tuple[str, float]:
    """Tek bir etiket goruntusu icin (kategori, guven_skoru) doner (guven = softmax olasiligi)."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Gorsel okunamadi: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform = get_eval_transforms(image_size=image_size)
    tensor = transform(image=image)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        top_idx = int(torch.argmax(probs).item())

    return categories[top_idx], float(probs[top_idx].item())


def load_best_model_from_report(
    report_path: Path = DEFAULT_REPORT,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    device: str = "cpu",
) -> tuple[torch.nn.Module, list[str]]:
    """Faz 2 egitim raporundan en iyi backbone'u okuyup vision_best.pt agirliklarini yukler."""
    with Path(report_path).open("r", encoding="utf-8") as f:
        report = json.load(f)

    categories = report["categories"]
    backbone = report["best_backbone"]
    model = load_trained_model(checkpoint_path, backbone, num_classes=len(categories), device=device)
    return model, categories
