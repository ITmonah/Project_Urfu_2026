import os
import sys
import logging
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

logging.getLogger('ultralytics').setLevel(logging.CRITICAL)

logging.getLogger('transformers').setLevel(logging.CRITICAL)

logging.getLogger('torch').setLevel(logging.CRITICAL)

import torch
from PIL import Image
from ultralytics import YOLO
import torchvision.transforms as transforms
from torchvision.models import efficientnet_v2_s

YOLO_PATH = "Yolo26s_kgo.pt"
CLS_PATH = "best_efficientnet_v2_s_kgo.pth"
FULL_CLS_PATH = "best_efficientnet_for_full.pth"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

det_model = None
cls_model = None
full_cls_model = None

CLASS_NAMES = []
FULL_CLASS_NAMES = []

def load_models(yolo_path=YOLO_PATH, cls_path=CLS_PATH, full_cls_path=FULL_CLS_PATH):
    global det_model, cls_model, full_cls_model, CLASS_NAMES, FULL_CLASS_NAMES

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if not os.path.exists(yolo_path):
            raise FileNotFoundError(f'YOLO не найден: {yolo_path}')
        det_model = YOLO(yolo_path)

        if not os.path.exists(cls_path):
            raise FileNotFoundError(f'Классификатор (2 класса) не найден: {cls_path}')
        checkpoint_old = torch.load(cls_path, map_location=device)
        if 'class_names' in checkpoint_old:
            CLASS_NAMES = checkpoint_old['class_names']
            num_classes = len(CLASS_NAMES)
        else:
            raise ValueError("Не найдены class_names в старом классификаторе")
        cls_model = efficientnet_v2_s(weights=None, num_classes=num_classes)
        cls_model.load_state_dict(checkpoint_old['state_dict'])
        cls_model.to(device)
        cls_model.eval()

        if not os.path.exists(full_cls_path):
            raise FileNotFoundError(f'Классификатор (3 класса) не найден: {full_cls_path}')
        checkpoint_new = torch.load(full_cls_path, map_location=device)
        if 'class_names' in checkpoint_new:
            FULL_CLASS_NAMES = checkpoint_new['class_names']
            num_classes_full = len(FULL_CLASS_NAMES)
        else:
            raise ValueError("Не найдены class_names в новом классификаторе")
        full_cls_model = efficientnet_v2_s(weights=None, num_classes=num_classes_full)
        full_cls_model.load_state_dict(checkpoint_new['state_dict'])
        full_cls_model.to(device)
        full_cls_model.eval()

full_cls_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def process_image(image_path):
    if det_model is None or cls_model is None or full_cls_model is None:
        raise RuntimeError('Модели не загружены')

    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        raise ValueError(f'Не удалось открыть изображение: {e}')

    results = det_model(image, verbose=False)
    yolo_boxes = results[0].boxes
    has_platform = False
    platform_box = None

    if yolo_boxes is not None and len(yolo_boxes) > 0:
        names = results[0].names
        target_id = None
        for idx, name in names.items():
            if name == 'kgo_platform':
                target_id = idx
                break
        if target_id is not None:
            for box in yolo_boxes:
                if int(box.cls[0]) == target_id:
                    platform_box = box
                    has_platform = True
                    break

    if not has_platform:
        return ""

    x1, y1, x2, y2 = platform_box.xyxy[0].tolist()
    crop = image.crop((x1, y1, x2, y2))

    preprocess_crop = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    input_crop = preprocess_crop(crop).unsqueeze(0).to(device)
    with torch.no_grad():
        out_old = cls_model(input_crop)
        prob_old = torch.softmax(out_old, dim=1)
        max_prob_old, pred_old = torch.max(prob_old, dim=1)
    old_class = CLASS_NAMES[pred_old.item()]
    old_conf = max_prob_old.item()

    input_full = full_cls_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        out_new = full_cls_model(input_full)
        prob_new = torch.softmax(out_new, dim=1)
        max_prob_new, pred_new = torch.max(prob_new, dim=1)
    new_class = FULL_CLASS_NAMES[pred_new.item()]
    new_conf = max_prob_new.item()

    if old_conf >= new_conf:
        return old_class
    else:
        return "" if new_class == 'kgo_none' else new_class
