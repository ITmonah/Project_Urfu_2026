import argparse
import json
import random
import time
from pathlib import Path
import sys
from pathlib import Path

# Поднимаемся на два уровня вверх (из train_classifier/ в корень проекта) и добавляем в sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ----------------------------------------------------------------------
# Конфигурация обучения (гиперпараметры)
# ----------------------------------------------------------------------
EPOCHS = 50
EARLY_STOP_PATIENCE = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 4
SEED = 42
MODEL_NAME = "efficientnet_v2_s"
MODEL_INPUT_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DATASET_ROOT = Path("classification_data")
OUTPUT_DIR = Path("artifacts/classification_efficientnet")

# Классы в том же порядке, в котором они лежат в classification_data
CLASS_NAMES = ("kgo_empty", "kgo_full", "kgo_none")

# ----------------------------------------------------------------------
# Вспомогательные классы и функции
# ----------------------------------------------------------------------
class ClassificationDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

def collect_samples(dataset_root, class_names):
    samples = []
    for label_idx, class_name in enumerate(class_names):
        class_dir = dataset_root / class_name
        if not class_dir.exists():
            print(f"Предупреждение: папка {class_dir} не найдена")
            continue
        for image_path in sorted(class_dir.glob("*.jpg")):
            samples.append((image_path, label_idx))
    if not samples:
        raise RuntimeError(f"Не найдено .jpg файлов в {dataset_root}")
    image_paths, labels = zip(*samples)
    return list(image_paths), list(labels)

def split_dataset(image_paths, labels, seed):
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels, test_size=0.30, random_state=seed, stratify=labels
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.50, random_state=seed, stratify=temp_labels
    )
    return {
        "train": (train_paths, train_labels),
        "val": (val_paths, val_labels),
        "test": (test_paths, test_labels),
    }

def build_transforms(input_size, mean, std):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return train_transform, eval_transform

def create_dataloaders(splits, batch_size, num_workers, train_transform, eval_transform):
    dataloaders = {}
    for split_name, (paths, labels) in splits.items():
        transform = train_transform if split_name == "train" else eval_transform
        dataset = ClassificationDataset(paths, labels, transform=transform)
        dataloaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            num_workers=num_workers,
            pin_memory=True,
        )
    return dataloaders

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            total_loss += loss.item() * images.size(0)
            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy())
    avg_loss = total_loss / len(dataloader.dataset)
    all_probs = np.array(all_probs)
    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy_score(all_labels, all_preds),
        "macro_f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(all_labels, all_preds),
        "roc_auc": None,
        "average_precision": None,
    }
    if len(set(all_labels)) > 1:
        try:
            metrics["roc_auc"] = roc_auc_score(all_labels, all_probs[:, 1])
            metrics["average_precision"] = average_precision_score(all_labels, all_probs[:, 1])
        except:
            pass
    return metrics, all_labels, all_preds, all_probs

def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(dataloader.dataset)

# ----------------------------------------------------------------------
# Основной процесс
# ----------------------------------------------------------------------
def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используемое устройство: {device}")

    # Сбор данных
    print("Сбор образцов из", DATASET_ROOT)
    image_paths, labels = collect_samples(DATASET_ROOT, CLASS_NAMES)
    print(f"Всего образцов: {len(image_paths)}")
    splits = split_dataset(image_paths, labels, SEED)
    for split_name, (paths, _) in splits.items():
        print(f"  {split_name}: {len(paths)}")

    train_transform, eval_transform = build_transforms(MODEL_INPUT_SIZE, IMAGENET_MEAN, IMAGENET_STD)
    dataloaders = create_dataloaders(splits, BATCH_SIZE, NUM_WORKERS, train_transform, eval_transform)

    # Создание модели
    from pipeline.classifier_models import create_classifier
    model = create_classifier(MODEL_NAME, num_classes=len(CLASS_NAMES), pretrained=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_f1 = 0.0
    best_state = None
    best_epoch = -1
    stale_epochs = 0

    print(f"\nНачало обучения модели {MODEL_NAME}")
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, dataloaders["train"], criterion, optimizer, scaler, device)
        val_metrics, _, _, _ = evaluate_model(model, dataloaders["val"], criterion, device)
        scheduler.step(val_metrics["loss"])

        print(f"Epoch {epoch:2d}/{EPOCHS} | train_loss: {train_loss:.4f} | "
              f"val_loss: {val_metrics['loss']:.4f} | val_f1: {val_metrics['macro_f1']:.4f}")

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= EARLY_STOP_PATIENCE:
                print(f"Ранняя остановка на эпохе {epoch}")
                break

    # Загрузка лучшей модели и оценка на тесте
    model.load_state_dict(best_state)
    test_metrics, test_labels, test_preds, test_probs = evaluate_model(model, dataloaders["test"], criterion, device)

    print("\nРезультаты на тестовой выборке:")
    print(f"  Accuracy:            {test_metrics['accuracy']:.4f}")
    print(f"  Macro F1:            {test_metrics['macro_f1']:.4f}")
    print(f"  Balanced Accuracy:   {test_metrics['balanced_accuracy']:.4f}")
    if test_metrics["roc_auc"] is not None:
        print(f"  ROC AUC:             {test_metrics['roc_auc']:.4f}")
    if test_metrics["average_precision"] is not None:
        print(f"  Average Precision:   {test_metrics['average_precision']:.4f}")
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=CLASS_NAMES, zero_division=0))

    # Сохранение артефактов
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_name": MODEL_NAME, "class_names": CLASS_NAMES, "state_dict": best_state},
        OUTPUT_DIR / f"best_{MODEL_NAME}_kgo.pth",
    )
    metrics_data = {
        "best_epoch": best_epoch,
        "val_f1": best_val_f1,
        "test": {
            "accuracy": test_metrics["accuracy"],
            "macro_f1": test_metrics["macro_f1"],
            "balanced_accuracy": test_metrics["balanced_accuracy"],
            "roc_auc": test_metrics["roc_auc"],
            "average_precision": test_metrics["average_precision"],
        },
    }
    with open(OUTPUT_DIR / f"metrics_{MODEL_NAME}.json", "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"\nМодель и метрики сохранены в {OUTPUT_DIR}")

if __name__ == "__main__":
    main()