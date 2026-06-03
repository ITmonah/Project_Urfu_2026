# FastAPI UI

Веб-приложение для запуска трех реальных пайплайнов:

- `new_classifier_nodino`: `NewClassificatorNoDino` (YOLO + классификатор)
- `sam`: `SAM_model` (foundation SAM segmentation)
- `smp`: `SMP_model` (Segmentation Models PyTorch)

## Запуск

```powershell
pip install -r fastapi_app\requirements.txt
uvicorn fastapi_app.main:app --reload
```

После запуска откройте `http://127.0.0.1:8000`.

## Checkpoints

Файлы `.pt` и `.pth` должны храниться в отдельном репозитории с GitHub Releases.
По умолчанию приложение скачивает недостающие веса из `likip3/AI-Models`, release tag `v1`, и кеширует их в `.model_cache`.

Общие настройки:

- `KGO_MODEL_REPO` - по умолчанию `likip3/AI-Models`
- `KGO_MODEL_RELEASE_TAG` - по умолчанию `v1`
- `KGO_MODEL_CACHE_DIR` - по умолчанию `.model_cache`
- `KGO_MODEL_AUTH_TOKEN` - optional token для private repository

Пути можно переопределить через переменные окружения:

- `KGO_NCD_YOLO_CHECKPOINT`
- `KGO_NCD_CROP_CLASSIFIER_CHECKPOINT`
- `KGO_NCD_FULL_CLASSIFIER_CHECKPOINT`
- `KGO_SAM_YOLO_CHECKPOINT`
- `KGO_SAM_CHECKPOINT`
- `KGO_SMP_YOLO_CHECKPOINT`
- `KGO_SMP_CHECKPOINT`

Прямые URL можно переопределить через:

- `KGO_NCD_YOLO_URL`
- `KGO_NCD_CROP_CLASSIFIER_URL`
- `KGO_NCD_FULL_CLASSIFIER_URL`
- `KGO_SAM_YOLO_URL`
- `KGO_SAM_URL`
- `KGO_SMP_YOLO_URL`
- `KGO_SMP_URL`

Если checkpoint не найден локально, приложение скачает его при запуске выбранного пайплайна.
Главная страница продолжит открываться даже без локально скачанных весов.

## API

`GET /api/pipelines`

Возвращает список пайплайнов, описания и доступность checkpoints.

`POST /api/predict`

Form-data:

- `image`: файл изображения
- `pipeline`: optional id пайплайна, по умолчанию `new_classifier_nodino`

Допустимые значения `pipeline`: `new_classifier_nodino`, `sam`, `smp`.

Поле `model` больше не используется, но остается допустимым для обратной совместимости старых клиентов.
