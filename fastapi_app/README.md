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

По умолчанию приложение ищет веса внутри папок пайплайнов. Пути можно переопределить через переменные окружения:

- `KGO_NCD_YOLO_CHECKPOINT`
- `KGO_NCD_CROP_CLASSIFIER_CHECKPOINT`
- `KGO_NCD_FULL_CLASSIFIER_CHECKPOINT`
- `KGO_SAM_YOLO_CHECKPOINT`
- `KGO_SAM_CHECKPOINT`
- `KGO_SMP_YOLO_CHECKPOINT`
- `KGO_SMP_CHECKPOINT`

Если checkpoint не найден, главная страница продолжит открываться, а запуск выбранного пайплайна вернет понятную ошибку.

## API

`GET /api/pipelines`

Возвращает список пайплайнов, описания и доступность checkpoints.

`POST /api/predict`

Form-data:

- `image`: файл изображения
- `pipeline`: optional id пайплайна, по умолчанию `new_classifier_nodino`

Допустимые значения `pipeline`: `new_classifier_nodino`, `sam`, `smp`.

Поле `model` больше не используется, но остается допустимым для обратной совместимости старых клиентов.
