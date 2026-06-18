# MLOps и CI/CD

Проект хранится в GitHub-репозитории `ITmonah/Project_Urfu_2026`. Разработка ведется в отдельных ветках: ветки приложения, экспериментов и интеграции данных не смешиваются с `main` до code review.

## Jenkins

Основной конвейер описан в `Jenkinsfile`.

Стадии:

1. Установка CI-зависимостей в Python 3.11 virtualenv.
2. `ruff check` для runtime-кода и тестов.
3. Unit tests: `pytest -q -m "not data_quality"`.
4. `dvc pull datasets.dvc model_weights.dvc`.
5. Data quality tests: `pytest -q -m data_quality`.
6. `docker build`.
7. `docker push` в Docker Hub для `main` и tag builds.

Обучение моделей в Jenkins не запускается. В репозитории сохранены training-скрипты и ноутбуки, но CI/CD проверяет только воспроизводимость данных, качество данных, код, тесты и сборку итогового Docker image.

## Jenkins credentials

Нужны два набора credentials:

- `dvc-s3`: username/password, где username соответствует `AWS_ACCESS_KEY_ID`, password соответствует `AWS_SECRET_ACCESS_KEY`.
- `dockerhub`: username/password для Docker Hub.

Для MinIO нужно добавить переменную окружения Jenkins job:

```text
DVC_S3_ENDPOINT_URL=http://minio.example.local:9000
```

Для обычного AWS S3 переменная `DVC_S3_ENDPOINT_URL` не нужна.

## DVC

DVC remote задан в `.dvc/config`:

```text
s3://project-urfu-2026-dvc
```

Версионируемые артефакты:

- `datasets.dvc`: архивы `datasets/39.zip`, `datasets/40.zip`, `datasets/42.zip`.
- `model_weights.dvc`: доступные локально веса моделей и экспериментальные checkpoint-файлы.

Перед первым запуском Jenkins или локальной проверкой данные нужно один раз отправить в remote:

```powershell
pip install "dvc[s3]"
dvc push datasets.dvc model_weights.dvc
```

На новой машине данные подтягиваются так:

```powershell
pip install "dvc[s3]"
dvc pull datasets.dvc model_weights.dvc
```

## Веса моделей

Приложение сначала использует локальные checkpoint-файлы, восстановленные через DVC. Если нужного файла нет локально, сохраняется существующий fallback: файл скачивается из GitHub Releases репозитория `likip3/AI-Models`.

Это позволяет:

- проверять версионирование весов через DVC;
- не ломать запуск приложения при отсутствии части локальных checkpoint-файлов;
- не запускать обучение в CI/CD.

## Data quality tests

Проверки находятся в `tests/test_data_quality.py` и помечены маркером `data_quality`.

Они проверяют, что:

- DVC-архивы датасетов доступны;
- каждый zip открывается;
- есть один файл `annotations/*.json`;
- аннотации соответствуют фактической Datumaro-структуре `info/categories/items`;
- категории содержат `kgo_platform`, `kgo_empty`, `kgo_full`;
- изображения, указанные в аннотациях, есть в архиве;
- bbox-аннотации имеют корректные размеры;
- sample изображений декодируется через PIL.
