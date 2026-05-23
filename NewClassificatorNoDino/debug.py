import os
import shutil
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter
from pipeline import load_models, process_image


TEST_FOLDER = r"../pipeline/images" 

# поддерживаемые классы
CLASS_NAMES = ["kgo_empty", "kgo_full", "kgo_none"] 

def get_true_label(file_path: Path, test_root: Path):
    try:
        # Берём папку первого уровня внутри TEST_FOLDER
        class_name = file_path.relative_to(test_root).parts[0].lower()
        if class_name in CLASS_NAMES:
            return class_name
    except ValueError:
        pass
    return None

def main():
    test_folder = Path(TEST_FOLDER)
    if not test_folder.exists():
        raise FileNotFoundError(f"Папка {test_folder} не найдена.")

    print("Загрузка моделей...")
    load_models()

    image_extensions = {".jpg", ".jpeg", ".png"}
    image_files = [p for p in test_folder.rglob("*") if p.suffix.lower() in image_extensions]

    y_true = []
    y_pred = []

    wrong_dir = Path("wrongPredictions")
    wrong_dir.mkdir(parents=True, exist_ok=True)

    stats = Counter()
    miss_classified = []
    for img_path in image_files:
        true_label = get_true_label(img_path, test_folder)
        if true_label is None:
            continue

        y_true.append(true_label)
        pred = process_image(str(img_path))
        if pred == "":
            pred_label = "kgo_none"
        else:
            pred_label = pred

        y_pred.append(pred_label)
        stats["total"] += 1
        if true_label != pred_label:
            miss_classified.append((str(img_path), true_label, pred_label))
            stats["errors"] += 1

            safe_name = "_".join(img_path.relative_to(test_folder).parts)
            dest_name = f"{true_label}_pred_{pred_label}_{safe_name}"
            shutil.copy2(img_path, wrong_dir / dest_name)

        if pred_label == "kgo_none":
            stats["misses"] += 1

    print(f"\nОбработано изображений: {stats['total']}")
    print(f"Ошибок: {stats['errors']}")
    print(f"Пропусков (не обнаружено): {stats['misses']}")

    report = classification_report(y_true, y_pred, labels=CLASS_NAMES,
                                   target_names=CLASS_NAMES, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    with open("report.txt", "w", encoding="utf-8") as f:
        f.write(f"Папка: {test_folder}\n")
        f.write(f"Количество изображений: {stats['total']}\n")
        f.write(f"Распределение истинных классов: {Counter(y_true)}\n")
        f.write(f"Точность (accuracy): {accuracy:.4f}\n\n")
        f.write(report)
        f.write("\nСписок ошибочных предсказаний (первые 50):\n")
        for path, true, pred in miss_classified[:50]:
            f.write(f"{path} | истина: {true} | предсказано: {pred}\n")

    print(f"Результаты сохранены в report.txt")
    print(f"Точность (accuracy): {accuracy:.4f}")
    print(report)

if __name__ == "__main__":
    main()