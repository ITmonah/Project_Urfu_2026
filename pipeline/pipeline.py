import os
import torch
from ultralytics import YOLO
from PIL import Image
import base64
import io
from pathlib import Path

try:
    from .classifier_models import build_inference_transform, load_classifier_checkpoint
except ImportError:
    from classifier_models import build_inference_transform, load_classifier_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEBUG_IMAGE_PATH = ""
USE_DEBUG_IMAGE = False
TARGET_DETECTION_CLASS = os.getenv("KGO_YOLO_CLASS_NAME", "kgo_platform")

current_dir = Path(__file__).parent
yolo_checkpoint_path = Path(os.getenv("KGO_YOLO_CHECKPOINT", current_dir / "yolo_source.pt"))
classifier_checkpoint_path = Path(os.getenv("KGO_CLASSIFIER_CHECKPOINT", current_dir / "best_resnet_kgo.pth"))

if not yolo_checkpoint_path.exists():
    raise FileNotFoundError(f"YOLO checkpoint not found: {yolo_checkpoint_path}")
if not classifier_checkpoint_path.exists():
    raise FileNotFoundError(f"Classifier checkpoint not found: {classifier_checkpoint_path}")

yolo_model = YOLO(str(yolo_checkpoint_path))
classifier_model, classifier_name, class_names = load_classifier_checkpoint(classifier_checkpoint_path, device)
classifier_model = classifier_model.to(device)
classifier_model.eval()
transform = build_inference_transform()


def is_target_detection(result, cls_id):
    class_names = getattr(result, "names", None) or getattr(yolo_model, "names", {})
    class_name = class_names.get(cls_id) if isinstance(class_names, dict) else class_names[cls_id]
    return class_name == TARGET_DETECTION_CLASS

def process_images(images):
    if USE_DEBUG_IMAGE:
        if DEBUG_IMAGE_PATH:
            img = Image.open(DEBUG_IMAGE_PATH).convert("RGB")
            images = [img]
        else:
            return []
    results = []
    for img_input in images:
        if isinstance(img_input, str):
            img_data = base64.b64decode(img_input)
            img = Image.open(io.BytesIO(img_data)).convert("RGB")
        else:
            img = img_input
        yolo_results = yolo_model(img)
        if USE_DEBUG_IMAGE:
            yolo_results[0].save(str(current_dir / "debug.jpg"))
        detections = []
        for result in yolo_results:
            for box in result.boxes:
                cls = int(box.cls)
                if is_target_detection(result, cls):
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    crop = img.crop((x1, y1, x2, y2))
                    detections.append(crop)
        if not detections:
            results.append([])
            continue
        img_results = []
        for crop in detections:
            input_tensor = transform(crop).unsqueeze(0).to(device)
            with torch.no_grad():
                output = classifier_model(input_tensor)
                pred = torch.argmax(output, dim=1).item()
                img_results.append(class_names[pred])
        results.append(img_results)
    if USE_DEBUG_IMAGE:
        print(f"Classifier: {classifier_name}")
        print(results)
    return results
