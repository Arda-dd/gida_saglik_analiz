"""MobileNetV3 ve EfficientNet-B3 icin transfer learning model fabrikasi (timm tabanli).

Oneri formu 2.2: "Egitim surecinde transfer learning yaklasimi benimsenerek onceden
buyuk olcekli veri kumeleri (or. ImageNet) uzerinde egitilmis aglarin agirliklarindan
yararlanilacak, boylece daha az veriyle yuksek siniflandirma basarimi elde edilecektir."
"""

from __future__ import annotations

import timm
import torch.nn as nn

SUPPORTED_BACKBONES: dict[str, str] = {
    "mobilenetv3": "mobilenetv3_large_100",
    "efficientnet_b3": "efficientnet_b3",
    "efficientnet_b4": "efficientnet_b4",
    "vit": "vit_base_patch16_224",
}


def build_model(backbone: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """timm uzerinden ImageNet on-egitimli bir siniflandirma modeli olusturur."""
    if backbone not in SUPPORTED_BACKBONES:
        raise ValueError(
            f"Desteklenmeyen backbone: '{backbone}'. Secenekler: {list(SUPPORTED_BACKBONES)}"
        )

    timm_name = SUPPORTED_BACKBONES[backbone]
    return timm.create_model(timm_name, pretrained=pretrained, num_classes=num_classes)


def freeze_backbone_layers(model: nn.Module) -> None:
    """Siniflandirma katmani (head) haric tum agirliklari dondurur.

    Kucuk veri setiyle asiri ogrenmeyi (overfitting) azaltmak icin kullanilabilecek
    opsiyonel bir transfer learning stratejisidir (yalnizca son katman egitilir).
    """
    for param in model.parameters():
        param.requires_grad = False

    classifier = model.get_classifier()
    for param in classifier.parameters():
        param.requires_grad = True


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
