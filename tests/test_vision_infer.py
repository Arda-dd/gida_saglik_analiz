import json

import cv2
import numpy as np
import pytest
import torch

from src.vision.infer import load_best_model_from_report, load_trained_model, predict_category
from src.vision.model import build_model

CATEGORIES = ["sut_urunu", "atistirmalik", "icecek"]


def test_load_trained_model_restores_saved_weights(tmp_path):
    model = build_model("mobilenetv3", num_classes=len(CATEGORIES), pretrained=False)
    checkpoint_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), checkpoint_path)

    loaded = load_trained_model(checkpoint_path, "mobilenetv3", num_classes=len(CATEGORIES), device="cpu")

    assert not loaded.training  # eval() modunda olmali
    original_params = list(model.state_dict().values())
    loaded_params = list(loaded.state_dict().values())
    assert all(torch.equal(a, b) for a, b in zip(original_params, loaded_params))


def test_predict_category_returns_valid_category_and_confidence(tmp_path):
    model = build_model("mobilenetv3", num_classes=len(CATEGORIES), pretrained=False)
    model.eval()

    image_path = tmp_path / "label.jpg"
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(image_path), image)

    category, confidence = predict_category(model, image_path, CATEGORIES, device="cpu", image_size=64)

    assert category in CATEGORIES
    assert 0.0 <= confidence <= 1.0


def test_predict_category_raises_on_unreadable_image(tmp_path):
    model = build_model("mobilenetv3", num_classes=len(CATEGORIES), pretrained=False)
    bad_path = tmp_path / "bad.jpg"
    bad_path.write_bytes(b"not an image")

    with pytest.raises(ValueError):
        predict_category(model, bad_path, CATEGORIES, device="cpu")


def test_load_best_model_from_report_uses_reported_backbone(tmp_path):
    model = build_model("efficientnet_b3", num_classes=len(CATEGORIES), pretrained=False)
    checkpoint_path = tmp_path / "vision_best.pt"
    torch.save(model.state_dict(), checkpoint_path)

    report_path = tmp_path / "report.json"
    report = {"categories": CATEGORIES, "best_backbone": "efficientnet_b3"}
    report_path.write_text(json.dumps(report), encoding="utf-8")

    loaded_model, categories = load_best_model_from_report(report_path, checkpoint_path, device="cpu")

    assert categories == CATEGORIES
    assert not loaded_model.training
