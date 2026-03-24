import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from pipeline.classifier_models import (
    CLASS_NAMES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MODEL_INPUT_SIZE,
    SUPPORTED_MODELS,
    create_classifier,
)


DEFAULT_DATASET_ROOT = Path("datasetCreation/cropped_datumaro")
DEFAULT_OUTPUT_DIR = Path("artifacts/classification")


class CropDataset(Dataset):
    def __init__(self, image_paths, labels, transform):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert("RGB")
        return self.transform(image), self.labels[index]


def parse_args():
    parser = argparse.ArgumentParser(description="Train multiple classifiers on cropped KGO dataset.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["efficientnet_v2_s", "convnext_tiny"],
        choices=SUPPORTED_MODELS,
        help="Models to train. Use resnet50 to keep the previous baseline in the comparison.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--early-stop-patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms():
    train_transform = transforms.Compose(
        [
            transforms.Resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


def collect_samples(dataset_root: Path):
    image_paths = []
    labels = []
    for label_index, class_name in enumerate(CLASS_NAMES):
        class_dir = dataset_root / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Class directory not found: {class_dir}")
        for image_path in sorted(class_dir.glob("*.jpg")):
            image_paths.append(image_path)
            labels.append(label_index)

    if not image_paths:
        raise RuntimeError(f"No .jpg files found in {dataset_root}")

    return image_paths, labels


def split_samples(image_paths, labels, seed: int):
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths,
        labels,
        test_size=0.30,
        random_state=seed,
        stratify=labels,
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        test_size=0.50,
        random_state=seed,
        stratify=temp_labels,
    )
    return {
        "train": (train_paths, train_labels),
        "val": (val_paths, val_labels),
        "test": (test_paths, test_labels),
    }


def create_dataloaders(splits, batch_size: int, num_workers: int):
    train_transform, eval_transform = build_transforms()
    train_paths, train_labels = splits["train"]
    val_paths, val_labels = splits["val"]
    test_paths, test_labels = splits["test"]

    train_ds = CropDataset(train_paths, train_labels, train_transform)
    val_ds = CropDataset(val_paths, val_labels, eval_transform)
    test_ds = CropDataset(test_paths, test_labels, eval_transform)

    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
    }


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            total_loss += loss.item() * images.size(0)
            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs[:, 1].cpu().numpy().tolist())

    average_loss = total_loss / len(dataloader.dataset)
    metrics = {
        "loss": average_loss,
        "accuracy": accuracy_score(all_labels, all_preds),
        "macro_f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist(),
        "classification_report": classification_report(
            all_labels,
            all_preds,
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0,
            output_dict=True,
        ),
    }

    if len(set(all_labels)) > 1:
        metrics["roc_auc"] = roc_auc_score(all_labels, all_probs)
    else:
        metrics["roc_auc"] = None

    return metrics


def train_model(model_name, dataloaders, args, device):
    model = create_classifier(model_name, num_classes=len(CLASS_NAMES), pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_state = None
    best_metrics = None
    best_epoch = -1
    stale_epochs = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_labels = []
        train_preds = []

        for images, labels in dataloaders["train"]:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * images.size(0)
            train_labels.extend(labels.detach().cpu().numpy().tolist())
            train_preds.extend(outputs.argmax(dim=1).detach().cpu().numpy().tolist())

        train_metrics = {
            "loss": train_loss / len(dataloaders["train"].dataset),
            "accuracy": accuracy_score(train_labels, train_preds),
            "macro_f1": f1_score(train_labels, train_preds, average="macro", zero_division=0),
        }
        val_metrics = evaluate(model, dataloaders["val"], criterion, device)
        scheduler.step(val_metrics["loss"])

        epoch_metrics = {
            "epoch": epoch,
            "train": train_metrics,
            "val": {
                "loss": val_metrics["loss"],
                "accuracy": val_metrics["accuracy"],
                "macro_f1": val_metrics["macro_f1"],
                "roc_auc": val_metrics["roc_auc"],
            },
        }
        history.append(epoch_metrics)

        is_better = (
            best_metrics is None
            or val_metrics["macro_f1"] > best_metrics["macro_f1"]
            or (
                val_metrics["macro_f1"] == best_metrics["macro_f1"]
                and val_metrics["loss"] < best_metrics["loss"]
            )
        )

        print(
            f"[{model_name}] epoch {epoch:02d}/{args.epochs} "
            f"train_loss={train_metrics['loss']:.4f} train_f1={train_metrics['macro_f1']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_f1={val_metrics['macro_f1']:.4f}"
        )

        if is_better:
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            best_metrics = val_metrics
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.early_stop_patience:
                break

    if best_state is None:
        raise RuntimeError(f"Training failed to produce a checkpoint for {model_name}")

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, dataloaders["test"], criterion, device)

    return {
        "model_name": model_name,
        "best_epoch": best_epoch,
        "history": history,
        "best_val": best_metrics,
        "test": test_metrics,
        "state_dict": best_state,
    }


def save_result(result, output_dir: Path, splits, seed: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"best_{result['model_name']}_kgo.pth"
    metrics_path = output_dir / f"metrics_{result['model_name']}.json"

    torch.save(
        {
            "model_name": result["model_name"],
            "class_names": CLASS_NAMES,
            "input_size": MODEL_INPUT_SIZE,
            "state_dict": result["state_dict"],
        },
        checkpoint_path,
    )

    payload = {
        "model_name": result["model_name"],
        "best_epoch": result["best_epoch"],
        "best_val": result["best_val"],
        "test": result["test"],
        "split_sizes": {split_name: len(split_data[0]) for split_name, split_data in splits.items()},
        "seed": seed,
    }
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_leaderboard(results, output_dir: Path):
    leaderboard = []
    for result in results:
        leaderboard.append(
            {
                "model_name": result["model_name"],
                "best_epoch": result["best_epoch"],
                "val_macro_f1": result["best_val"]["macro_f1"],
                "val_loss": result["best_val"]["loss"],
                "test_macro_f1": result["test"]["macro_f1"],
                "test_accuracy": result["test"]["accuracy"],
                "test_roc_auc": result["test"]["roc_auc"],
            }
        )

    leaderboard.sort(key=lambda item: (-item["val_macro_f1"], item["val_loss"]))
    (output_dir / "leaderboard.json").write_text(
        json.dumps(leaderboard, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main():
    args = parse_args()
    dataset_root = args.dataset_root
    output_dir = args.output_dir

    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset root not found: {dataset_root}. "
            "Run datasetCreation/prepare_datumaro_dataset.py first."
        )

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    image_paths, labels = collect_samples(dataset_root)
    splits = split_samples(image_paths, labels, seed=args.seed)
    dataloaders = create_dataloaders(splits, batch_size=args.batch_size, num_workers=args.num_workers)

    print(
        "Split sizes:",
        {split_name: len(split_data[0]) for split_name, split_data in splits.items()},
    )

    results = []
    for model_name in args.models:
        started_at = time.time()
        result = train_model(model_name, dataloaders, args, device)
        result["elapsed_seconds"] = round(time.time() - started_at, 2)
        save_result(result, output_dir, splits, args.seed)
        results.append(result)
        print(
            f"[{model_name}] done in {result['elapsed_seconds']}s "
            f"test_f1={result['test']['macro_f1']:.4f} test_acc={result['test']['accuracy']:.4f}"
        )

    save_leaderboard(results, output_dir)
    print(f"Saved checkpoints and metrics to {output_dir}")


if __name__ == "__main__":
    main()
