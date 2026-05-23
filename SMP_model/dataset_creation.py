import os
import random
import shutil
import numpy as np
from PIL import Image

BASE_DIR = '.'                              # Корень, куда разархивировали CVAT
LABELMAP_PATH = os.path.join(BASE_DIR, 'labelmap.txt')
IMAGES_DIR = os.path.join(BASE_DIR, 'JPEGImages/images/train')
MASKS_SRC_DIR = os.path.join(BASE_DIR, 'SegmentationClass/images/train')
DEFAULT_TXT = os.path.join(BASE_DIR, 'ImageSets', 'Segmentation', 'default.txt')

DATASET_DIR = 'dataset'            # Итоговая папка
TRAIN_RATIO = 1
VAL_RATIO   = 0
TEST_RATIO  = 0
SEED = 42

# Расширения
IMG_EXT = '.jpg'
MASK_EXT = '.png'

print("Чтение labelmap и создание маппинга цвет → индекс...")

# Назначение: фон 0, пол 1, стена 2, мусор 3
class_name_to_target = {
    'background':    0,
    'kgo_floor':     1,
    'kgo_wall':      2,
    'kgo_garbage':   3,
    'garbage_can':   3,
    'kgo_empty':     3,
    'kgo_full':      3,
    'kgo_platform':  3,
}

color_to_target = {}
with open(LABELMAP_PATH, 'r') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split(':')
        class_name = parts[0]
        rgb = tuple(map(int, parts[1].split(',')))
        if class_name in class_name_to_target:
            color_to_target[rgb] = class_name_to_target[class_name]
        else:
            color_to_target[rgb] = 0

print(f"Загружено {len(color_to_target)} цветов из labelmap")

print("Конвертация масок...")
MASKS_TEMP_DIR = os.path.join(DATASET_DIR, 'masks_temp')
os.makedirs(MASKS_TEMP_DIR, exist_ok=True)

converted_files = set()
for fname in os.listdir(MASKS_SRC_DIR):
    if not fname.endswith(MASK_EXT):
        continue
    img = Image.open(os.path.join(MASKS_SRC_DIR, fname)).convert('RGB')
    arr = np.array(img)
    h, w = arr.shape[:2]
    mask_out = np.zeros((h, w), dtype=np.uint8)

    for rgb, tgt_idx in color_to_target.items():
        match = np.all(arr == np.array(rgb), axis=2)
        mask_out[match] = tgt_idx

    out_path = os.path.join(MASKS_TEMP_DIR, fname)
    Image.fromarray(mask_out, mode='L').save(out_path)
    converted_files.add(fname.replace(MASK_EXT, ''))

print(f"Конвертировано масок: {len(converted_files)}")

print("Поиск полных пар изображение-маска...")
img_files = {f.replace(IMG_EXT, '') for f in os.listdir(IMAGES_DIR) if f.endswith(IMG_EXT)}
common_ids = sorted(img_files & converted_files)


print(f"Найдено полных пар изображение+маска: {len(common_ids)}")

# Разбиение на train/val/test
random.seed(SEED)
random.shuffle(common_ids)

n_total = len(common_ids)
n_train = int(n_total * TRAIN_RATIO)
n_val   = int(n_total * VAL_RATIO)

train_ids = common_ids[:n_train]
val_ids   = common_ids[n_train:n_train + n_val]
test_ids  = common_ids[n_train + n_val:]

print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

print("Копирование файлов в dataset_4classes/...")
for split_name, ids in [('train', train_ids), ('val', val_ids), ('test', test_ids)]:
    img_dst = os.path.join(DATASET_DIR, split_name, 'images')
    mask_dst = os.path.join(DATASET_DIR, split_name, 'masks')
    os.makedirs(img_dst, exist_ok=True)
    os.makedirs(mask_dst, exist_ok=True)

    for file_id in ids:
        src_img = os.path.join(IMAGES_DIR, file_id + IMG_EXT)
        if os.path.exists(src_img):
            shutil.copy2(src_img, img_dst)
        else:
            print(f"Пропущено изображение: {src_img}")

        src_mask = os.path.join(MASKS_TEMP_DIR, file_id + MASK_EXT)
        if os.path.exists(src_mask):
            shutil.copy2(src_mask, mask_dst)
        else:
            print(f"Пропущена маска: {src_mask}")

shutil.rmtree(MASKS_TEMP_DIR)

print("\nГотово! Структура датасета:")
print(DATASET_DIR)
for split in ['train', 'val', 'test']:
    n_imgs = len(os.listdir(os.path.join(DATASET_DIR, split, 'images')))
    n_msks = len(os.listdir(os.path.join(DATASET_DIR, split, 'masks')))
    print(f"  {split}/  images: {n_imgs}  masks: {n_msks}")