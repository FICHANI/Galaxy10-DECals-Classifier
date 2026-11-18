import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

from config import ProjectConfig


def build_resnet18(num_classes=ProjectConfig.NUM_CLASSES, pretrained=True):
    """ResNet18 with ImageNet weights (if requested), fc swapped for Galaxy10."""
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model   = resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def freeze_backbone(model):
    """Freeze everything except the final classifier layer."""
    for name, param in model.named_parameters():
        if not name.startswith("fc."):
            param.requires_grad = False


def unfreeze_all(model):
    """Unfreeze every parameter — used for the fine-tuning phase."""
    for param in model.parameters():
        param.requires_grad = True