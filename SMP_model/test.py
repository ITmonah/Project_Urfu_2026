"""
test.py – Тестирование пайплайна с сохранением визуализаций в success/errors.
Показывает процент заполнения на изображении.
Структура:
    images/
        kgo_full/
        kgo_empty/
Результат:
    success/   – верно классифицированные
    errors/    – ошибочные (включая случаи, когда зона не найдена)
"""

import os
import glob
import cv2
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

from pipeline import KGOFillPipeline


# ---------- сбор данных ----------
def collect_dataset(root_dir: str):
    image_paths = []
    true_labels = []
    for label in ["kgo_full", "kgo_empty"]:
        folder = os.path.join(root_dir, label)
        if not os.path.isdir(folder):
            print(f"Папка {folder} не найдена, пропускаем.")
            continue
        for ext in ["jpg", "jpeg", "png", "bmp", "tiff"]:
            for path in glob.glob(os.path.join(folder, f"*.{ext}")):
                image_paths.append(path)
                true_labels.append(label)
    return image_paths, true_labels


# ---------- визуализация ----------
def draw_segmentation_overlay(crop, mask, alpha=0.5):
    """
    Накладывает цветную маску на кроп.
    crop: исходный BGR-кроп
    mask: np.ndarray (H,W) с классами 0..3
    alpha: прозрачность
    Возвращает BGR-изображение.
    """
    colors = {
        0: (0, 0, 0),         # фон
        1: (255, 0, 0),       # пол – синий
        2: (0, 255, 0),       # стена – зелёный
        3: (0, 0, 255),       # мусор – красный
    }
    overlay = crop.copy()
    for cls_id, color in colors.items():
        if cls_id == 0:
            continue
        overlay[mask == cls_id] = color
    blended = cv2.addWeighted(crop, 1 - alpha, overlay, alpha, 0)
    return blended


def make_visualization(original_image, bbox, mask, segm_size, fill_ratio=None):
    """
    Создаёт итоговое изображение для сохранения.
    original_image: исходное BGR-изображение
    bbox: (x1,y1,x2,y2) или None
    mask: np.ndarray маски или None
    segm_size: размер маски (для ресайза)
    fill_ratio: float (0..1) или None
    Возвращает BGR-изображение.
    """
    vis = original_image.copy()

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
        crop = original_image[y1:y2, x1:x2].copy()
        if mask is not None and crop.size > 0:
            mask_resized = cv2.resize(
                mask.astype(np.uint8),
                (crop.shape[1], crop.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            crop_vis = draw_segmentation_overlay(crop, mask_resized)
            vis[y1:y2, x1:x2] = crop_vis
    else:
        cv2.putText(
            vis,
            "No KGO detected",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 0, 255),
            3,
        )

    # Отображение процента заполнения
    if fill_ratio is not None:
        text = f"Fill: {fill_ratio:.1%}"
    else:
        text = "Fill: N/A"

    # Белый фон для читаемости
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    cv2.rectangle(vis, (10, vis.shape[0] - 10 - text_h - 10), (10 + text_w + 10, vis.shape[0] - 10), (255, 255, 255), -1)
    cv2.putText(vis, text, (15, vis.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    return vis


# ---------- оценка ----------
def evaluate_and_save(pipeline, image_paths, true_labels, output_dir="output"):
    success_dir = os.path.join(output_dir, "success")
    errors_dir = os.path.join(output_dir, "errors")
    os.makedirs(success_dir, exist_ok=True)
    os.makedirs(errors_dir, exist_ok=True)

    y_pred = []
    y_true = []

    for path, true_lbl in zip(image_paths, true_labels):
        image = cv2.imread(path)
        if image is None:
            print(f"Не удалось загрузить {path}, пропускаем.")
            continue

        status, mask, bbox, fill_ratio = pipeline.predict_with_mask(image)
        pred_lbl = status if status != "" else "no_zone"
        y_pred.append(pred_lbl)
        y_true.append(true_lbl)

        vis = make_visualization(image, bbox, mask, pipeline.segm_input_size, fill_ratio)

        is_correct = pred_lbl == true_lbl
        out_folder = success_dir if is_correct else errors_dir

        base_name = os.path.splitext(os.path.basename(path))[0]
        out_name = f"{base_name}_true_{true_lbl}_pred_{pred_lbl}.png"
        out_path = os.path.join(out_folder, out_name)
        cv2.imwrite(out_path, vis)

    return y_true, y_pred


# ---------- отчёт ----------
def print_report(y_true, y_pred):
    labels = ["kgo_empty", "kgo_full", "no_zone"]
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    print("=" * 60)
    print("         РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ ПАЙПЛАЙНА")
    print("=" * 60)
    print(f"Всего изображений: {len(y_true)}")
    print(f"Accuracy : {acc:.4f}\n")

    for cls, p, r, f, s in zip(labels, precision, recall, f1, support):
        if s > 0:
            print(f"--- {cls} (support={s}) ---")
            print(f"  Precision : {p:.4f}")
            print(f"  Recall    : {r:.4f}")
            print(f"  F1-score  : {f:.4f}\n")

    active = [i for i, s in enumerate(support) if s > 0]
    if active:
        macro_p = np.mean([precision[i] for i in active])
        macro_r = np.mean([recall[i] for i in active])
        macro_f1 = np.mean([f1[i] for i in active])
        print("--- Macro Average (присутствующие классы) ---")
        print(f"  Precision : {macro_p:.4f}")
        print(f"  Recall    : {macro_r:.4f}")
        print(f"  F1-score  : {macro_f1:.4f}")

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("\nConfusion Matrix (true \\ pred)")
    header = "               " + " ".join(f"{lbl:>12}" for lbl in labels)
    print(header)
    for i, lbl_true in enumerate(labels):
        row = f"{lbl_true:>14} " + " ".join(f"{cm[i,j]:>12}" for j in range(len(labels)))
        print(row)


# ---------- main ----------
if __name__ == "__main__":
    IMAGE_ROOT = "images"
    YOLO_WEIGHTS = "Yolo26s_kgo.pt"
    SEGM_WEIGHTS = "best_model.pth"
    FILL_THRESHOLD = 0.5
    USE_FLOOR_ONLY = False
    OUTPUT_DIR = "output"

    paths, true_labels = collect_dataset(IMAGE_ROOT)
    print(f"Найдено: {len(paths)} изображений")

    pipeline = KGOFillPipeline(
        yolo_path=YOLO_WEIGHTS,
        segm_path=SEGM_WEIGHTS,
        fill_threshold=FILL_THRESHOLD,
        use_floor_only=USE_FLOOR_ONLY,
    )

    y_true, y_pred = evaluate_and_save(pipeline, paths, true_labels, OUTPUT_DIR)
    print(f"Визуализации сохранены в {OUTPUT_DIR}/ (success/errors)")
    print_report(y_true, y_pred)