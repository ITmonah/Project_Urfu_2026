## Запуск

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pipeline.debug
```

Перед запуском положите файлы весов `pipeline/yolo_source.pt` и `pipeline/best_resnet_kgo.pth` в папку `pipeline/`
или задайте пути через переменные окружения `KGO_YOLO_CHECKPOINT` и `KGO_CLASSIFIER_CHECKPOINT`.

## Подготовка датасета

Скрипт берёт `.zip`-архивы Datumaro из папки `datasets` и сохраняет кропы в `datasetCreation/cropped_datumaro`:

```powershell
python datasetCreation\prepare_datumaro_dataset.py --clear-output
```

## Обучение классификаторов

Обучение запускается на датасете из `datasetCreation/cropped_datumaro`, результаты сохраняются в `artifacts/classification`:

```powershell
python datasetCreation\train_classifiers.py
```
