# KGO ML Pipeline

Web/API-приложение на FastAPI для запуска пайплайнов машинного обучения по изображению контейнерной площадки.

Доступные режимы:

- `new_classifier_nodino` - YOLO + классификатор кропа + классификатор полного изображения.
- `sam` - YOLO + SAM-based сегментация.
- `smp` - YOLO + Segmentation Models PyTorch.

## Быстрый запуск

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r fastapi_app\requirements.txt
uvicorn fastapi_app.main:app --reload
```

После запуска откройте `http://127.0.0.1:8000`.

API endpoint:

```text
POST /api/predict
```

Form-data:

- `image` - файл изображения.
- `pipeline` - один из `new_classifier_nodino`, `sam`, `smp`.

## Модельные веса

Файлы `.pt` и `.pth` не хранятся в основном репозитории. Приложение скачивает их при первом запуске выбранного пайплайна из GitHub Releases отдельного репозитория:

```text
https://github.com/likip3/AI_Models/releases/tag/v1
```

Ожидаемые release assets:

- `Yolo26s_kgo.pt`
- `best_efficientnet_v2_s_kgo.pth`
- `best_efficientnet_for_full.pth`
- `sam_model.pth`
- `best_model_SMP.pth`

Настройки по умолчанию:

```powershell
$env:KGO_MODEL_REPO = "likip3/AI_Models"
$env:KGO_MODEL_RELEASE_TAG = "v1"
$env:KGO_MODEL_CACHE_DIR = ".model_cache"
```

Для private repository можно передать токен:

```powershell
$env:KGO_MODEL_AUTH_TOKEN = "<github-token>"
```

Каждый checkpoint можно переопределить локальным путём или прямым URL:

- `KGO_NCD_YOLO_CHECKPOINT`, `KGO_NCD_YOLO_URL`
- `KGO_NCD_CROP_CLASSIFIER_CHECKPOINT`, `KGO_NCD_CROP_CLASSIFIER_URL`
- `KGO_NCD_FULL_CLASSIFIER_CHECKPOINT`, `KGO_NCD_FULL_CLASSIFIER_URL`
- `KGO_SAM_YOLO_CHECKPOINT`, `KGO_SAM_YOLO_URL`
- `KGO_SAM_CHECKPOINT`, `KGO_SAM_URL`
- `KGO_SMP_YOLO_CHECKPOINT`, `KGO_SMP_YOLO_URL`
- `KGO_SMP_CHECKPOINT`, `KGO_SMP_URL`

## Docker

Сборка образа:

```powershell
docker build -t project-urfu-2026 .
```

Запуск:

```powershell
docker run --rm -p 8000:8000 `
  -e KGO_MODEL_REPO=likip3/AI_Models `
  -e KGO_MODEL_RELEASE_TAG=v1 `
  project-urfu-2026
```

После запуска приложение доступно на `http://127.0.0.1:8000`.

## CI/CD

Workflow `.github/workflows/ci.yml` выполняет:

- unit tests через `pytest`;
- проверку PEP8/runtime-кода через `ruff`;
- сборку Docker image;
- публикацию Docker image в Docker Hub при наличии repository secrets:
  `DOCKERHUB_USERNAME` и `DOCKERHUB_TOKEN`.

Имя публикуемого образа:

```text
DOCKERHUB_USERNAME/project-urfu-2026
```

## Проверки локально

```powershell
python -m ruff check fastapi_app pipeline/classifier_models.py pipeline/pipeline.py predict_image.py NewClassificatorNoDino/pipeline.py SAM_model/pipeline.py SMP_model/pipeline.py tests
python -m pytest -q
```

## Подготовка датасета и обучение

Скрипты подготовки датасета и обучения сохранены в `datasetCreation`, `NewClassificatorNoDino`, `SAM_model` и `SMP_model`.
Они не требуются для запуска Web/API приложения, но оставлены в репозитории как исследовательская и training-часть проекта.
