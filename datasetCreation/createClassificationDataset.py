import os
import json
from pathlib import Path
from PIL import Image

IMAGES_DIR = Path("images/train")    
LABELS_DIR = Path("labels/train")    

JSON_DIR = Path("jsons")             

TARGET_CLASS = 2

OUTPUT_ROOT = Path("cropped")
FULL_DIR = OUTPUT_ROOT / "kgo_full"
EMPTY_DIR = OUTPUT_ROOT / "kgo_empty"


def main():
    FULL_DIR.mkdir(parents=True, exist_ok=True)
    EMPTY_DIR.mkdir(parents=True, exist_ok=True)

    if not JSON_DIR.exists() or not JSON_DIR.is_dir():
        print(f"Папка с JSON не найдена: {JSON_DIR}")
        return

    json_files = sorted(JSON_DIR.glob("*.json"))
    if not json_files:
        print(f"В папке {JSON_DIR} не найдено ни одного .json файла")
        return

    data = {}
    for json_path in json_files:
        try:
            with open(json_path, encoding="utf-8") as f:
                part = json.load(f)
                data.update(part)          
        except Exception as e:
            print(f"Ошибка при чтении {json_path.name}: {e}")


    processed = 0
    skipped_no_class2 = 0
    skipped_no_json = 0

    for img_path in sorted(IMAGES_DIR.glob("*.jpg")):
        stem = img_path.stem 

        txt_path = LABELS_DIR / f"{stem}.txt"
        if not txt_path.exists():
            continue

        if stem not in data:
            skipped_no_json += 1
            continue

        img_info = data[stem]
        labels = img_info.get("labels", []) if isinstance(img_info, dict) else []
        if not isinstance(labels, list):
            labels = [labels]

        tag = None
        for lbl in labels:
            if lbl in ("kgo_full", 1, "1", True):
                tag = "kgo_full"
                break
            elif lbl in ("kgo_empty", 0, "0", False):
                tag = "kgo_empty"
                break

        if tag is None:
            continue

        boxes = []
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    continue
                cls = int(parts[0])
                if cls == TARGET_CLASS:
                    x_center, y_center, width, height = map(float, parts[1:])
                    boxes.append((x_center, y_center, width, height))

        if not boxes:
            skipped_no_class2 += 1
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            w, h = img.size
        except Exception as e:
            continue

        for idx, (x_center, y_center, bw, bh) in enumerate(boxes):
            x1 = int((x_center - bw / 2) * w)
            y1 = int((y_center - bh / 2) * h)
            x2 = int((x_center + bw / 2) * w)
            y2 = int((y_center + bh / 2) * h)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            crop = img.crop((x1, y1, x2, y2))

            crop_name = f"{stem}_crop_{idx}.jpg"
            save_path = (FULL_DIR if tag == "kgo_full" else EMPTY_DIR) / crop_name

            crop.save(save_path, quality=95)
            processed += 1


if __name__ == "__main__":
    main()