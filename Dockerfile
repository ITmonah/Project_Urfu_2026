FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    KGO_MODEL_CACHE_DIR=/app/.model_cache \
    KGO_PRELOAD_MODEL_ASSETS=1 \
    NO_ALBUMENTATIONS_UPDATE=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libxcb1 \
    && rm -rf /var/lib/apt/lists/*

COPY fastapi_app/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p /app/.model_cache

EXPOSE 8000 8443

CMD ["python", "-m", "fastapi_app.run"]
