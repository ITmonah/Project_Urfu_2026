import os
import shutil
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
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
yolo_checkpoint_path = current_dir / "best.pt"
wrong_predictions_root = current_dir / "wrong_predictions"
report_path = current_dir / "report.txt"

TARGET_YOLO_CLASS = os.getenv("KGO_YOLO_CLASS_NAME", "kgo_platform")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"


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


def detect_crops(image, yolo_model):
    result_list = yolo_model(image, device=DEVICE_STR)
    crops = []
    for result in result_list:
        for box in result.boxes:
            cls_id = int(box.cls)
            if is_target_detection(result, cls_id):
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                crops.append(image.crop((x1, y1, x2, y2)))
    return crops


def classify_crops(crops, classifier, transform, device):
    if not crops:
        return "no_detection"

    predictions = []
    for crop in crops:
        tensor = transform(crop).unsqueeze(0).to(device)
        with torch.no_grad():
            output = classifier["model"](tensor)
            predicted_index = int(output.argmax(dim=1).item())
            predictions.append(classifier["class_names"][predicted_index])

    most_common = Counter(predictions).most_common(1)
    return most_common[0][0] if most_common else predictions[0]


def save_wrong_prediction(image_path, classifier_name, true_label, predicted_label, existing_count):
    target_dir = wrong_predictions_root / classifier_name / true_label
    target_dir.mkdir(parents=True, exist_ok=True)
    destination = target_dir / f"{image_path.stem}_pred_{predicted_label}_{existing_count}{image_path.suffix}"
    shutil.copy2(image_path, destination)


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
        crops = detect_crops(image, yolo_model)

        for classifier in classifiers:
            classifier_name = classifier["name"]
            prediction = classify_crops(crops, classifier, transform, DEVICE)

            predictions_by_classifier[classifier_name]["y_true"].append(true_label)
            predictions_by_classifier[classifier_name]["y_pred"].append(prediction)
            predictions_by_classifier[classifier_name]["total_count"] += 1

            if prediction == "no_detection":
                predictions_by_classifier[classifier_name]["no_detection_count"] += 1

            if prediction != true_label:
                predictions_by_classifier[classifier_name]["wrong_count"] += 1
                existing_wrong_counts[classifier_name] += 1
                save_wrong_prediction(
                    image_path,
                    classifier_name,
                    true_label,
                    prediction,
                    existing_wrong_counts[classifier_name],
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