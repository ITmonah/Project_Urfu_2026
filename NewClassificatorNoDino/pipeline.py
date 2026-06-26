import os
import logging
import warnings

import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import efficientnet_v2_s
from ultralytics import YOLO

warnings.filterwarnings('ignore')

# Настройка уровня логирования библиотек 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # TensorFlow: показывать только ошибки
os.environ['TRANSFORMERS_VERBOSITY'] = 'error' # Transformers: только ошибки
os.environ['TOKENIZERS_PARALLELISM'] = 'false' # Отключаем параллелизм токенизаторов

logging.getLogger('ultralytics').setLevel(logging.CRITICAL) # YOLO - только критические ошибки

logging.getLogger('transformers').setLevel(logging.CRITICAL) # Transformers - только критические

logging.getLogger('torch').setLevel(logging.CRITICAL) # PyTorch - только критические

YOLO_PATH = "Yolo26s_kgo.pt"
CLS_PATH = "best_efficientnet_v2_s_kgo.pth"
FULL_CLS_PATH = "best_efficientnet_for_full.pth"

# Определяем устройство для вычислений
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Глобальные переменные для хранения загруженных моделей
det_model = None
cls_model = None
full_cls_model = None

# Списки названий классов для каждой модели
CLASS_NAMES = []
FULL_CLASS_NAMES = []

def load_models(yolo_path=YOLO_PATH, cls_path=CLS_PATH, full_cls_path=FULL_CLS_PATH):
    global det_model, cls_model, full_cls_model, CLASS_NAMES, FULL_CLASS_NAMES

    # Игнорируем предупреждения при загрузке моделей
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Загрузка YOLO модели для детекции контейнеров
        if not os.path.exists(yolo_path):
            raise FileNotFoundError(f'YOLO не найден: {yolo_path}')
        det_model = YOLO(yolo_path)

        # Загрузка классификатора на 2 класса
        if not os.path.exists(cls_path):
            raise FileNotFoundError(f'Классификатор (2 класса) не найден: {cls_path}')
        checkpoint_old = torch.load(cls_path, map_location=device)
        if 'class_names' in checkpoint_old:
            CLASS_NAMES = checkpoint_old['class_names']
            num_classes = len(CLASS_NAMES)
        else:
            raise ValueError("Не найдены class_names в старом классификаторе")
        
        # Извлекаем названия классов из чекпоинта
        cls_model = efficientnet_v2_s(weights=None, num_classes=num_classes)
        cls_model.load_state_dict(checkpoint_old['state_dict'])
        cls_model.to(device)
        cls_model.eval()

        # Загрузка классификатора на 3 класса
        if not os.path.exists(full_cls_path):
            raise FileNotFoundError(f'Классификатор (3 класса) не найден: {full_cls_path}')
        checkpoint_new = torch.load(full_cls_path, map_location=device)
        
        # Извлекаем названия классов из чекпоинта
        if 'class_names' in checkpoint_new:
            FULL_CLASS_NAMES = checkpoint_new['class_names']
            num_classes_full = len(FULL_CLASS_NAMES)
        else:
            raise ValueError("Не найдены class_names в новом классификаторе")
        
        # Создаем и загружаем модель классификатора
        full_cls_model = efficientnet_v2_s(weights=None, num_classes=num_classes_full)
        full_cls_model.load_state_dict(checkpoint_new['state_dict'])
        full_cls_model.to(device)
        full_cls_model.eval()

# Трансформации для полного изображения
full_cls_transform = transforms.Compose([
    transforms.Resize(256), # Изменяем размер до 256x256
    transforms.CenterCrop(224), # Центральный кроп 224x224
    transforms.ToTensor(), # Преобразуем в тензор
    transforms.Normalize(mean=[0.485, 0.456, 0.406], # Нормализуем по стандарту ImageNet
                         std=[0.229, 0.224, 0.225]),
])

def process_image(image_path):
    # Проверяем, что модели загружены
    if det_model is None or cls_model is None or full_cls_model is None:
        raise RuntimeError('Модели не загружены')

    # Загружаем изображение
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        raise ValueError(f'Не удалось открыть изображение: {e}')

    # Детекция контейнера с помощью YOLO
    results = det_model(image, verbose=False)
    yolo_boxes = results[0].boxes
    has_platform = False
    platform_box = None

    # Ищем bounding box для класса 'kgo_platform'
    if yolo_boxes is not None and len(yolo_boxes) > 0:
        names = results[0].names
        target_id = None
        for idx, name in names.items():
            if name == 'kgo_platform':
                target_id = idx
                break
        # Ищем первый bounding box с нужным классом
        if target_id is not None:
            for box in yolo_boxes:
                if int(box.cls[0]) == target_id:
                    platform_box = box
                    has_platform = True
                    break

    # Если контейнер не найден - возвращаем "пустой" результат
    if not has_platform:
        return ""

    # Вырезаем область контейнера для классификации
    x1, y1, x2, y2 = platform_box.xyxy[0].tolist()
    crop = image.crop((x1, y1, x2, y2))

    # Трансформации для кропа контейнера
    preprocess_crop = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    # Классификация кропа на 2 класса (empty/full)
    input_crop = preprocess_crop(crop).unsqueeze(0).to(device)
    with torch.no_grad(): 
        out_old = cls_model(input_crop) 
        prob_old = torch.softmax(out_old, dim=1)
        max_prob_old, pred_old = torch.max(prob_old, dim=1)
    old_class = CLASS_NAMES[pred_old.item()]
    old_conf = max_prob_old.item()

    # Классификация всего изображения на 3 класса (empty/full/none)
    input_full = full_cls_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        out_new = full_cls_model(input_full)
        prob_new = torch.softmax(out_new, dim=1)
        max_prob_new, pred_new = torch.max(prob_new, dim=1)
    new_class = FULL_CLASS_NAMES[pred_new.item()]
    new_conf = max_prob_new.item()

    # Выбор результата на основе уверенности моделей
    # Если 2-классовая модель увереннее - возвращаем её результат
    if old_conf >= new_conf:
        return old_class
    else:
        # Иначе возвращаем результат 3-классовой модели
        # Если она предсказала 'kgo_none' - возвращаем пустую строку
        return "" if new_class == 'kgo_none' else new_class
