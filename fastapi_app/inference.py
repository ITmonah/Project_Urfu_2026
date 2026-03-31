import base64
import os
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from ultralytics import YOLO

from pipeline.classifier_models import (
    SUPPORTED_MODELS,
    build_inference_transform,
    load_classifier_checkpoint,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_YOLO_CHECKPOINT = ROOT_DIR / "pipeline" / "yolo_source.pt"
CLASSIFIER_DIR = ROOT_DIR / "artifacts" / "classification"
TARGET_DETECTION_CLASS = os.getenv("KGO_YOLO_CLASS_NAME", "kgo_platform")

CHECKPOINT_BY_MODEL = {
    "resnet50": CLASSIFIER_DIR / "best_resnet50_kgo.pth",
    "efficientnet_v2_s": CLASSIFIER_DIR / "best_efficientnet_v2_s_kgo.pth",
    "convnext_tiny": CLASSIFIER_DIR / "best_convnext_tiny_kgo.pth",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRANSFORM = build_inference_transform()


def list_available_models() -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    for model_name in SUPPORTED_MODELS:
        checkpoint_path = CHECKPOINT_BY_MODEL.get(model_name)
        models.append(
            {
                "name": model_name,
                "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
                "available": bool(checkpoint_path and checkpoint_path.exists()),
            }
        )
    return models


def resolve_classifier_checkpoint(model_name: str) -> Path:
    if model_name not in CHECKPOINT_BY_MODEL:
        raise ValueError(
            f"Model '{model_name}' is not configured for the web app. "
            f"Available: {', '.join(CHECKPOINT_BY_MODEL)}"
        )

    checkpoint_path = CHECKPOINT_BY_MODEL[model_name]
    if checkpoint_path.exists():
        return checkpoint_path.resolve()

    raise FileNotFoundError(f"Classifier checkpoint not found: {checkpoint_path}")


def resolve_yolo_checkpoint() -> Path:
    checkpoint_path = Path(os.getenv("KGO_YOLO_CHECKPOINT", DEFAULT_YOLO_CHECKPOINT))
    if checkpoint_path.exists():
        return checkpoint_path.resolve()
    raise FileNotFoundError(f"YOLO checkpoint not found: {checkpoint_path}")


@lru_cache(maxsize=1)
def load_detector(yolo_checkpoint: str) -> YOLO:
    return YOLO(yolo_checkpoint)


@lru_cache(maxsize=8)
def load_classifier(classifier_checkpoint: str):
    model, model_name, class_names = load_classifier_checkpoint(Path(classifier_checkpoint), DEVICE)
    model = model.to(DEVICE)
    model.eval()
    return model, model_name, class_names


def is_target_detection(result, detector: YOLO, cls_id: int) -> bool:
    class_names = getattr(result, "names", None) or getattr(detector, "names", {})
    class_name = class_names.get(cls_id) if isinstance(class_names, dict) else class_names[cls_id]
    return class_name == TARGET_DETECTION_CLASS


def image_to_data_url(image_bytes: bytes, content_type: str) -> str:
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{content_type};base64,{encoded}"


def run_inference(image_bytes: bytes, model_name: str) -> dict[str, Any]:
    classifier_checkpoint = resolve_classifier_checkpoint(model_name)
    yolo_checkpoint = resolve_yolo_checkpoint()

    detector = load_detector(str(yolo_checkpoint))
    classifier_model, classifier_name, class_names = load_classifier(str(classifier_checkpoint))

    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    yolo_results = detector(image)

    predictions: list[dict[str, Any]] = []
    for result in yolo_results:
        for box in result.boxes:
            cls_id = int(box.cls)
            if not is_target_detection(result, detector, cls_id):
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            crop = image.crop((x1, y1, x2, y2))
            input_tensor = TRANSFORM(crop).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                output = classifier_model(input_tensor)
                pred_idx = torch.argmax(output, dim=1).item()

            predictions.append(
                {
                    "label": class_names[pred_idx],
                    "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                }
            )

    return {
        "model": model_name,
        "classifier_name": classifier_name,
        "classifier_checkpoint": str(classifier_checkpoint),
        "yolo_checkpoint": str(yolo_checkpoint),
        "predictions": predictions,
    }
