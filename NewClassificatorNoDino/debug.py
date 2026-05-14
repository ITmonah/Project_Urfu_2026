import os
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter
from pipeline import load_models, process_image

# ============================================================
# НАСТРОЙКИ (укажите нужную папку)
# ============================================================
TEST_FOLDER = r"../pipeline/images"  # <-- замените на ваш путь

# поддерживаемые классы
CLASS_NAMES = ["kgo_empty", "kgo_full", "kgo_none"]  # kgo_none соответствует "" (нет обнаружений)

def get_true_label(file_path: Path):
    """Определяем истинную метку по имени родительской папки."""
    # Проходим по родителям, ищем папку с именем kgo_empty или kgo_full
    for parent in file_path.parents:
        if parent.name.lower() == "kgo_empty":
            return "kgo_empty"
        if parent.name.lower() == "kgo_full":
            return "kgo_full"
    return None

def main():
    test_folder = Path(TEST_FOLDER)
    if not test_folder.exists():
        raise FileNotFoundError(f"Папка {test_folder} не найдена.")

    # Загрузка моделей
    print("Загрузка моделей...")
    load_models()

    # Сбор файлов (jpg, png) рекурсивно
    image_extensions = {".jpg", ".jpeg", ".png"}
    image_files = [p for p in test_folder.rglob("*") if p.suffix.lower() in image_extensions]

    y_true = []
    y_pred = []

    stats = Counter()
    miss_classified = []
    for img_path in image_files:
        true_label = get_true_label(img_path)
        if true_label is None:
            continue  # пропускаем файлы не из целевых папок

        y_true.append(true_label)
        pred = process_image(str(img_path))
        # заменяем "" на "kgo_none"
        if pred == "":
            pred_label = "kgo_none"
        else:
            pred_label = pred

        y_pred.append(pred_label)
        stats["total"] += 1
        if true_label != pred_label:
            miss_classified.append((str(img_path), true_label, pred_label))
            stats["errors"] += 1
        if pred_label == "kgo_none":
            stats["misses"] += 1

    # Метрики
    print(f"\nОбработано изображений: {stats['total']}")
    print(f"Ошибок: {stats['errors']}")
    print(f"Пропусков (не обнаружено): {stats['misses']}")

    # Генерируем отчёт
    report = classification_report(y_true, y_pred, labels=CLASS_NAMES,
                                   target_names=CLASS_NAMES, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    # Сохраняем в report.txt
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