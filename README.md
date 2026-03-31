## Запуск

Быстрая проверка, что пайплайн поднимается и модельные веса читаются:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pipeline.debug
```

Перед запуском положите файлы весов `pipeline/yolo_source.pt` и `pipeline/best_resnet_kgo.pth` в папку `pipeline/`
или задайте пути через переменные окружения `KGO_YOLO_CHECKPOINT` и `KGO_CLASSIFIER_CHECKPOINT`.

`pipeline.debug` сейчас запускает простой smoke-test пайплайна через `process_images(...)` и печатает результат в консоль.

## Подготовка датасета

Скрипт берёт `.zip`-архивы Datumaro из папки `datasets` и сохраняет кропы в `datasetCreation/cropped_datumaro`:

```powershell
python datasetCreation\prepare_datumaro_dataset.py --clear-output
```

Основные аргументы:
- `--datasets-dir` - папка с `.zip`-архивами Datumaro, по умолчанию `datasets`
- `--output-dir` - куда сохранять кропы и `summary.json`, по умолчанию `datasetCreation/cropped_datumaro`
- `--archives` - список конкретных архивов для обработки
- `--clear-output` - очистить старые `.jpg` в папках классов перед генерацией

Пример:

```powershell
python datasetCreation\prepare_datumaro_dataset.py --datasets-dir datasets --output-dir datasetCreation\cropped_datumaro --clear-output
```

## Обучение классификаторов

Обучение запускается на датасете из `datasetCreation/cropped_datumaro`, результаты сохраняются в `artifacts/classification`:

```powershell
python datasetCreation\train_classifiers.py
```

Основные аргументы:
- `--dataset-root` - папка с подготовленными изображениями классов, по умолчанию `datasetCreation/cropped_datumaro`
- `--output-dir` - куда сохранять чекпоинты и метрики, по умолчанию `artifacts/classification`
- `--models` - список моделей для обучения, по умолчанию `efficientnet_v2_s convnext_tiny`
- `--epochs` - число эпох, по умолчанию `30`
- `--batch-size` - размер батча, по умолчанию `16`
- `--learning-rate` - learning rate, по умолчанию `1e-4`
- `--weight-decay` - weight decay, по умолчанию `1e-5`
- `--num-workers` - число worker-процессов DataLoader, по умолчанию `0`
- `--early-stop-patience` - patience для early stopping, по умолчанию `4`
- `--seed` - random seed, по умолчанию `42`
- `--no-pretrained` - отключить предобученные веса backbone

Пример:

```powershell
python datasetCreation\train_classifiers.py --models resnet50 efficientnet_v2_s --epochs 20 --batch-size 8
```

Ключевые метрики классификатора сохраняются в `artifacts/classification/metrics_<model>.json` и `artifacts/classification/leaderboard.json`.
Там есть как минимум `accuracy`, `macro_f1`, `roc_auc`, номер лучшей эпохи и размеры train/val/test split.

## Проверка пайплайна и метрик

- Для быстрой проверки запуска всего пайплайна используйте `python -m pipeline.debug`
- Для расчёта численных метрик в текущем репозитории используется `python datasetCreation\train_classifiers.py`
- Отдельного скрипта для end-to-end метрик всего пайплайна `YOLO + classifier` сейчас в репозитории нет
