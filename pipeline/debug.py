import os
import shutil
from collections import Counter
from pathlib import Path

import torch
import numpy as np
from PIL import Image, ImageDraw
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from ultralytics import YOLO

try:
    from .classifier_models import build_inference_transform, load_classifier_checkpoint, CLASS_NAMES
except ImportError:
    from classifier_models import build_inference_transform, load_classifier_checkpoint, CLASS_NAMES

current_dir = Path(__file__).parent
images_root = current_dir / "images"
yolo_checkpoint_path = current_dir / "best2mver.pt"
wrong_predictions_root = current_dir / "wrong_predictions2"
report_path = current_dir / "report2.txt"

TARGET_YOLO_CLASS = os.getenv("KGO_YOLO_CLASS_NAME", "kgo_full")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"

# Параметры фильтрации боксов YOLO (можно менять по необходимости)
YOLO_CONF_THRESHOLD = 0.3
MIN_BOX_AREA = 500


def find_classifier_checkpoints():
    return sorted(current_dir.glob("best_*_kgo.pth"))


def load_classifiers(device):
    classifiers = []
    for checkpoint_path in find_classifier_checkpoints():
        model, classifier_name, class_names = load_classifier_checkpoint(checkpoint_path, device)
        model = model.to(device)
        model.eval()
        classifiers.append(
            {
                "checkpoint_path": checkpoint_path,
                "name": classifier_name,
                "model": model,
                "class_names": class_names,
            }
        )
    return classifiers


def is_target_detection(result, cls_id):
    class_names = getattr(result, "names", None) or getattr(result, "model", None)
    if isinstance(class_names, dict):
        class_name = class_names.get(cls_id)
    else:
        class_name = class_names[cls_id]
    return class_name == TARGET_YOLO_CLASS


def gather_image_paths():
    if not images_root.exists():
        raise FileNotFoundError(f"Images folder not found: {images_root}")

    image_paths = []
    for class_dir in sorted(images_root.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"):
            for image_path in sorted(class_dir.glob(ext)):
                image_paths.append((class_name, image_path))
    if not image_paths:
        raise RuntimeError(f"No image files found under {images_root}")
    return image_paths


def detect_crops(image, yolo_model, conf_thresh=YOLO_CONF_THRESHOLD, min_area=MIN_BOX_AREA):
    """Возвращает список кропов и координат боксов для целевого класса."""
    results = yolo_model(image, conf=conf_thresh, device=DEVICE_STR)
    crops = []
    boxes = []
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls)
            if is_target_detection(result, cls_id):
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                area = (x2 - x1) * (y2 - y1)
                if area >= min_area:
                    crops.append(image.crop((x1, y1, x2, y2)))
                    boxes.append((x1, y1, x2, y2))
    return crops, boxes


def draw_boxes(image, boxes, color="red", width=3):
    """Рисует прямоугольники на копии изображения."""
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    for (x1, y1, x2, y2) in boxes:
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    return img_draw


def classify_crops(crops, classifier, transform, device, confidence_threshold=0.0):
    if not crops:
        return "no_detection", 0.0, False

    class_names = classifier["class_names"]
    model = classifier["model"]
    prob_sums = {name: 0.0 for name in class_names}
    max_prob = 0.0
    has_full = False

    for crop in crops:
        tensor = transform(crop).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(tensor)
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        pred_class = class_names[pred_idx]
        prob = probs[pred_idx]

        if prob > max_prob:
            max_prob = prob

        if prob >= confidence_threshold:
            prob_sums[pred_class] += prob

        if pred_class == "kgo_full":
            has_full = True

    if has_full:
        prediction = "kgo_full"
    else:
        if any(prob_sums.values()):
            prediction = max(prob_sums, key=prob_sums.get)
        else:
            all_probs = []
            for crop in crops:
                tensor = transform(crop).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model(tensor)
                    probs = torch.softmax(output, dim=1).cpu().numpy()[0]
                all_probs.append(probs)
            avg_probs = np.mean(all_probs, axis=0)
            prediction = class_names[np.argmax(avg_probs)]
            max_prob = np.max(avg_probs)

    return prediction, max_prob, has_full


def save_wrong_prediction(image_pil, classifier_name, true_label, predicted_label, existing_count, image_stem, image_suffix):
    target_dir = wrong_predictions_root / classifier_name / true_label
    target_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{image_stem}_pred_{predicted_label}_{existing_count}{image_suffix}"
    destination = target_dir / filename
    image_pil.save(destination)


def build_metrics(y_true, y_pred, class_names):
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=class_names, average="macro", zero_division=0)
    macro_precision = precision_score(y_true, y_pred, labels=class_names, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, labels=class_names, average="macro", zero_division=0)
    report = classification_report(
        y_true,
        y_pred,
        labels=class_names,
        target_names=class_names,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=class_names)
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "report_text": report,
        "confusion_matrix": matrix.tolist(),
    }


def write_report(overall_stats):
    lines = []

    for classifier_name, stats in overall_stats.items():
        lines.extend([
            f"## Classifier: {classifier_name}",
            "--------------------------------------",
            f"Total images: {stats['total_count']}",
            f"Wrong predictions: {stats['wrong_count']}",
            f"Overall accuracy: {stats['accuracy']*100:.2f}%",
            f"Macro F1: {stats['macro_f1']*100:.2f}%",
            f"Macro precision: {stats['macro_precision']*100:.2f}%",
            f"Macro recall: {stats['macro_recall']*100:.2f}%",
            "",
        ])

    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"Report written to {report_path}")


def main():
    yolo_model = YOLO(str(yolo_checkpoint_path))
    image_paths = gather_image_paths()
    classifiers = load_classifiers(DEVICE)
    transform = build_inference_transform()

    overall_stats = {}
    predictions_by_classifier = {
        classifier["name"]: {
            "y_true": [],
            "y_pred": [],
            "wrong_count": 0,
            "no_detection_count": 0,
            "total_count": 0,
        }
        for classifier in classifiers
    }
    existing_wrong_counts = {classifier["name"]: 0 for classifier in classifiers}

    wrong_predictions_root.mkdir(parents=True, exist_ok=True)

    for true_label, image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        crops, boxes = detect_crops(image, yolo_model)

        # Рисуем боксы один раз для всех классификаторов
        if boxes:
            image_with_boxes = draw_boxes(image, boxes)
        else:
            image_with_boxes = image

        for classifier in classifiers:
            classifier_name = classifier["name"]
            # Распаковываем три значения, возвращаемые classify_crops
            prediction, prob, has_full = classify_crops(crops, classifier, transform, DEVICE)

            predictions_by_classifier[classifier_name]["y_true"].append(true_label)
            predictions_by_classifier[classifier_name]["y_pred"].append(prediction)
            predictions_by_classifier[classifier_name]["total_count"] += 1

            if prediction == "no_detection":
                predictions_by_classifier[classifier_name]["no_detection_count"] += 1

            if prediction != true_label:
                predictions_by_classifier[classifier_name]["wrong_count"] += 1
                existing_wrong_counts[classifier_name] += 1
                save_wrong_prediction(
                    image_with_boxes,
                    classifier_name,
                    true_label,
                    prediction,
                    existing_wrong_counts[classifier_name],
                    image_path.stem,
                    image_path.suffix,
                )

    for classifier in classifiers:
        classifier_name = classifier["name"]
        stats = predictions_by_classifier[classifier_name]
        metrics = build_metrics(
            stats["y_true"],
            stats["y_pred"],
            CLASS_NAMES,
        )
        overall_stats[classifier_name] = {
            **metrics,
            "wrong_count": stats["wrong_count"],
            "no_detection_count": stats["no_detection_count"],
            "total_count": stats["total_count"],
        }

    write_report(overall_stats)


if __name__ == "__main__":
    main()