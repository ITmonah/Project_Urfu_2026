import os
import sys
import logging
import warnings

# ----------------------------------------------------------------------
# Подавляем все сторонние логи, предупреждения и выводы
# ----------------------------------------------------------------------
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Логи ultralytics (YOLO)
logging.getLogger('ultralytics').setLevel(logging.CRITICAL)

# Логи transformers и groundingdino
logging.getLogger('transformers').setLevel(logging.CRITICAL)
logging.getLogger('groundingdino').setLevel(logging.CRITICAL)
logging.getLogger('GroundingDINO').setLevel(logging.CRITICAL)

# Логи torch
logging.getLogger('torch').setLevel(logging.CRITICAL)

import torch
from PIL import Image
from ultralytics import YOLO
import torchvision.transforms as transforms
from torchvision.models import efficientnet_v2_s
import groundingdino.datasets.transforms as T
from groundingdino.util.inference import load_model as load_grounding_dino_model, predict

# ----------------------------------------------------------------------
# Параметры (можно менять при необходимости)
# ----------------------------------------------------------------------
YOLO_PATH = "Yolo26s_kgo.pt"
CLS_PATH = "best_efficientnet_v2_s_kgo.pth"
FULL_CLS_PATH = "best_efficientnet_for_full.pth"

GROUNDING_DINO_CONFIG = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_WEIGHTS = "GroundingDINO/weights/groundingdino_swint_ogc.pth"
GROUNDING_DINO_BOX_THRESHOLD = 0.35
GROUNDING_DINO_TEXT_THRESHOLD = 0.25

DINO_CONFIG_URL = "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py"
DINO_WEIGHTS_URL = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swint_ogc.pth"

# ----------------------------------------------------------------------
# Глобальные модели
# ----------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

det_model = None
cls_model = None
full_cls_model = None
grounding_dino = None

CLASS_NAMES = []
FULL_CLASS_NAMES = []

# ----------------------------------------------------------------------
# Скачивание файлов (молча)
# ----------------------------------------------------------------------
def download_file(url: str, destination: str):
    if os.path.exists(destination):
        return
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    try:
        from urllib.request import urlretrieve
        urlretrieve(url, destination)
    except Exception:
        torch.hub.download_url_to_file(url, destination)

# ----------------------------------------------------------------------
# Загрузка моделей (вывод скрыт)
# ----------------------------------------------------------------------
def load_models():
    global det_model, cls_model, full_cls_model, grounding_dino, CLASS_NAMES, FULL_CLASS_NAMES

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # 1. YOLO
        if not os.path.exists(YOLO_PATH):
            raise FileNotFoundError(f'YOLO не найден: {YOLO_PATH}')
        det_model = YOLO(YOLO_PATH)

        # 2. Старый классификатор
        if not os.path.exists(CLS_PATH):
            raise FileNotFoundError(f'Классификатор (2 класса) не найден: {CLS_PATH}')
        checkpoint_old = torch.load(CLS_PATH, map_location=device)
        if 'class_names' in checkpoint_old:
            CLASS_NAMES = checkpoint_old['class_names']
            num_classes = len(CLASS_NAMES)
        else:
            raise ValueError("Не найдены class_names в старом классификаторе")
        cls_model = efficientnet_v2_s(weights=None, num_classes=num_classes)
        cls_model.load_state_dict(checkpoint_old['state_dict'])
        cls_model.to(device)
        cls_model.eval()

        # 3. Новый классификатор (целое изображение)
        if not os.path.exists(FULL_CLS_PATH):
            raise FileNotFoundError(f'Классификатор (3 класса) не найден: {FULL_CLS_PATH}')
        checkpoint_new = torch.load(FULL_CLS_PATH, map_location=device)
        if 'class_names' in checkpoint_new:
            FULL_CLASS_NAMES = checkpoint_new['class_names']
            num_classes_full = len(FULL_CLASS_NAMES)
        else:
            raise ValueError("Не найдены class_names в новом классификаторе")
        full_cls_model = efficientnet_v2_s(weights=None, num_classes=num_classes_full)
        full_cls_model.load_state_dict(checkpoint_new['state_dict'])
        full_cls_model.to(device)
        full_cls_model.eval()

        # 4. Grounding DINO (при необходимости скачивается)
        if not os.path.exists(GROUNDING_DINO_CONFIG):
            download_file(DINO_CONFIG_URL, GROUNDING_DINO_CONFIG)
        if not os.path.exists(GROUNDING_DINO_WEIGHTS):
            download_file(DINO_WEIGHTS_URL, GROUNDING_DINO_WEIGHTS)

        grounding_dino = load_grounding_dino_model(GROUNDING_DINO_CONFIG, GROUNDING_DINO_WEIGHTS)

# ----------------------------------------------------------------------
# Предобработка для нового классификатора
# ----------------------------------------------------------------------
full_cls_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ----------------------------------------------------------------------
# Предобработка для DINO (преобразование PIL -> Tensor)
# ----------------------------------------------------------------------
dino_transform = T.Compose([
    T.RandomResize([800], max_size=1333),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ----------------------------------------------------------------------
# Обработка одного изображения
# ----------------------------------------------------------------------
def process_image(image_path):
    if det_model is None or cls_model is None or full_cls_model is None or grounding_dino is None:
        raise RuntimeError('Сначала нужно вызвать load_models().')

    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        raise ValueError(f'Не удалось открыть изображение: {e}')

    # YOLO ищет платформу
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

    # ---------- Сценарий A: платформа найдена ----------
    if has_platform:
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

        print("Default")

        if old_conf >= new_conf:
            return old_class
        else:
            return "" if new_class == 'kgo_none' else new_class

    # ---------- Сценарий B: платформа НЕ найдена ----------
    input_full = full_cls_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        out_new = full_cls_model(input_full)
        prob_new = torch.softmax(out_new, dim=1)
        max_prob_new, pred_new = torch.max(prob_new, dim=1)
    new_class = FULL_CLASS_NAMES[pred_new.item()]

    if new_class == 'kgo_none':
        print("Default")
        return ""

    # ------------------ ЗАМЕНА: преобразуем PIL.Image -> torch.Tensor ------------------
    image_tensor, _ = dino_transform(image, None)   # теперь image_tensor - корректный тензор

    text_prompt = "filled bulky waste zone . empty bulky waste zone"
    detections = predict(
        model=grounding_dino,
        image=image_tensor,
        caption=text_prompt,
        box_threshold=GROUNDING_DINO_BOX_THRESHOLD,
        text_threshold=GROUNDING_DINO_TEXT_THRESHOLD
    )
    boxes, logits, phrases = detections

    if len(boxes) == 0:
        print("Default")
        return new_class

    scores = logits.max(dim=1)[0] if logits.dim() == 2 else logits
    best_idx = scores.argmax().item()
    best_score = scores[best_idx].item()
    best_phrase = phrases[best_idx].lower()

    if best_score < 0.3:
        print("Default")
        return new_class

    if "filled" in best_phrase:
        print("DINO")
        return "kgo_full"
    elif "empty" in best_phrase:
        print("DINO")
        return "kgo_empty"
    else:
        print("Default")
        return new_class