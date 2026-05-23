import argparse
import os
from pathlib import Path
from urllib.request import urlretrieve

from PIL import Image

from pipeline.classifier_models import SUPPORTED_MODELS


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_YOLO_CHECKPOINT = ROOT_DIR / "pipeline" / "yolo_source.pt"
CLASSIFIER_DIR = ROOT_DIR / "artifacts" / "classification"
CHECKPOINT_BY_MODEL = {
    "resnet50": CLASSIFIER_DIR / "best_resnet50_kgo.pth",
    "efficientnet_v2_s": CLASSIFIER_DIR / "best_efficientnet_v2_s_kgo.pth",
    "convnext_tiny": CLASSIFIER_DIR / "best_convnext_tiny_kgo.pth",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run one image through YOLO + selected classifier.")
    parser.add_argument("image", type=Path, help="Path to input image.")
    parser.add_argument("--model", choices=SUPPORTED_MODELS, default="convnext_tiny")
    parser.add_argument("--classifier-checkpoint", type=Path, default=None)
    parser.add_argument("--yolo-checkpoint", type=Path, default=DEFAULT_YOLO_CHECKPOINT)
    parser.add_argument("--yolo-url", default=os.getenv("KGO_YOLO_URL"))
    return parser.parse_args()


def resolve_classifier_checkpoint(args) -> Path:
    checkpoint_path = args.classifier_checkpoint or CHECKPOINT_BY_MODEL[args.model]
    if checkpoint_path.exists():
        return checkpoint_path.resolve()
    raise FileNotFoundError(f"Classifier checkpoint not found: {checkpoint_path}")


def ensure_yolo_checkpoint(checkpoint_path: Path, yolo_url: str | None) -> Path:
    if checkpoint_path.exists():
        return checkpoint_path.resolve()
    if not yolo_url:
        raise FileNotFoundError(
            f"YOLO checkpoint not found: {checkpoint_path}. "
        )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading YOLO checkpoint to {checkpoint_path}...")
    urlretrieve(yolo_url, checkpoint_path)
    return checkpoint_path.resolve()


def main():
    args = parse_args()
    image_path = args.image.resolve()
    classifier_checkpoint = resolve_classifier_checkpoint(args)
    yolo_checkpoint = ensure_yolo_checkpoint(args.yolo_checkpoint, args.yolo_url)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    os.environ["KGO_CLASSIFIER_CHECKPOINT"] = str(classifier_checkpoint)
    os.environ["KGO_YOLO_CHECKPOINT"] = str(yolo_checkpoint)

    from pipeline import process_images

    image = Image.open(image_path).convert("RGB")
    predictions = process_images([image])[0]

    print(f"image: {image_path}")
    print(f"classifier: {classifier_checkpoint}")
    print(f"yolo: {yolo_checkpoint}")

    if not predictions:
        print("result: no kgo_platform detections")
        return

    print("result:")
    for index, label in enumerate(predictions, start=1):
        print(f"  {index}. {label}")


if __name__ == "__main__":
    main()
