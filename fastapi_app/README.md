# FastAPI UI

Минимальное приложение для запуска существующего пайплайна через браузер и API.

## Запуск

```powershell
pip install -r fastapi_app\requirements.txt
uvicorn fastapi_app.main:app --reload
```

После запуска откройте `http://127.0.0.1:8000`.

## Что нужно для работы

- `pipeline/yolo_source.pt`
- `artifacts/classification/best_resnet50_kgo.pth`
- `artifacts/classification/best_efficientnet_v2_s_kgo.pth`
- `artifacts/classification/best_convnext_tiny_kgo.pth`

Если какого-то checkpoint нет, UI покажет это рядом с моделью, а API вернёт ошибку.

## API

`POST /api/predict`

Form-data:
- `image`: файл изображения
- `model`: имя классификатора, например `convnext_tiny`
