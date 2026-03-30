# KGO Pipeline

Обнаружение и классификация объектов на изображениях через YOLOv8 и ResNet.

## Установка

```bash
python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

## Структура

```
datasetCreation/      - подготовка датасета и обучение
  train_classifiers.py
  resnetTrain.ipynb
pipeline/             - основной пайплайн
  pipeline.py
  classifier_models.py
  best_resnet_kgo.pth
  yolo_source.pt
```

## Команды

**Обучение классификаторов:**
```bash
python datasetCreation/train_classifiers.py \
    --dataset-root datasetCreation/cropped_datumaro \
    --output-dir artifacts/classification \
    --models resnet50 \
    --epochs 50 \
    --batch-size 32
```

**Подготовка датасета:**

Обработка и преобразование исходного датасета в формат, пригодный для обучения. Первый скрипт подготавливает данные через Datumaro, второй создает обрезанные изображения для классификации.

```bash
python datasetCreation/prepare_datumaro_dataset.py
python datasetCreation/createClassificationDataset.py
```

**Jupyter ноутбук:**

Интерактивное окружение для экспериментов с обучением. Удобно для отладки, визуализации результатов и пошагового выполнения.

```bash
jupyter notebook datasetCreation/resnetTrain.ipynb
```

**Использование в коде:**
```python
from pipeline.pipeline import process_images
from PIL import Image

img = Image.open("image.jpg")
results = process_images([img])
```
