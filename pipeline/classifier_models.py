from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms


CLASS_NAMES = ["kgo_empty", "kgo_full"]
MODEL_INPUT_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
SUPPORTED_MODELS = ("resnet18", "resnet34", "resnet50", "efficientnet_v2_s", "convnext_tiny", "yolo")


def create_classifier(model_name: str, num_classes: int = 2, pretrained: bool = False) -> nn.Module:
    if model_name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if model_name == "resnet34":
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet34(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes) 
        return model

    if model_name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if model_name == "efficientnet_v2_s":
        weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_v2_s(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model

    if model_name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.convnext_tiny(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model

    raise ValueError(f"Unsupported model '{model_name}'. Supported: {', '.join(SUPPORTED_MODELS)}")


def build_inference_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def load_classifier_checkpoint(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(str(checkpoint_path), map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model_name = checkpoint.get("model_name", "resnet50")
        class_names = checkpoint.get("class_names", CLASS_NAMES)
        state_dict = checkpoint["state_dict"]
    else:
        model_name = "resnet50"
        class_names = CLASS_NAMES
        state_dict = checkpoint

    model = create_classifier(model_name, num_classes=len(class_names), pretrained=False)
    model.load_state_dict(state_dict)
    return model, model_name, class_names
