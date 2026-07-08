import pytest
import torch

from src.vision.model import (
    SUPPORTED_BACKBONES,
    build_model,
    count_total_parameters,
    count_trainable_parameters,
    freeze_backbone_layers,
)


def test_build_model_raises_on_unknown_backbone():
    with pytest.raises(ValueError):
        build_model("resnet_bilinmeyen", num_classes=5)


@pytest.mark.parametrize("backbone", list(SUPPORTED_BACKBONES.keys()))
def test_build_model_output_shape_matches_num_classes(backbone):
    model = build_model(backbone, num_classes=5, pretrained=False)
    model.eval()

    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (2, 5)


def test_freeze_backbone_layers_leaves_only_head_trainable():
    model = build_model("mobilenetv3", num_classes=5, pretrained=False)
    total_before = count_total_parameters(model)
    trainable_before = count_trainable_parameters(model)
    assert trainable_before == total_before  # baslangicta hepsi egitilebilir

    freeze_backbone_layers(model)
    trainable_after = count_trainable_parameters(model)

    assert 0 < trainable_after < total_before


def test_count_parameters_are_positive():
    model = build_model("efficientnet_b3", num_classes=5, pretrained=False)
    assert count_total_parameters(model) > 0
    assert count_trainable_parameters(model) == count_total_parameters(model)
