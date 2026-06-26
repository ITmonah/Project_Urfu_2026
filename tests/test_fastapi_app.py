from io import BytesIO

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from fastapi_app.https import RedirectToHTTPSMiddleware
from fastapi_app.main import app


client = TestClient(app)


def png_bytes() -> bytes:
    image = Image.new("RGB", (8, 8), color="white")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_index_page_opens():
    response = client.get("/")

    assert response.status_code == 200
    assert "Запуск пайплайна по изображению" in response.text


def test_pipelines_api_returns_configured_pipelines(monkeypatch, tmp_path):
    monkeypatch.setenv("KGO_MODEL_CACHE_DIR", str(tmp_path))
    response = client.get("/api/pipelines")

    assert response.status_code == 200
    payload = response.json()
    assert payload["default_pipeline"] == "new_classifier_nodino"
    assert {item["name"] for item in payload["pipelines"]} == {
        "new_classifier_nodino",
        "sam",
        "smp",
    }
    assert all(item["checkpoints"] for item in payload["pipelines"])


def test_models_api_keeps_legacy_response_shape():
    response = client.get("/api/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload["models"] == payload["pipelines"]


def test_predict_returns_400_for_unknown_pipeline():
    response = client.post(
        "/api/predict",
        data={"pipeline": "missing"},
        files={"image": ("input.png", png_bytes(), "image/png")},
    )

    assert response.status_code == 400
    assert "not configured" in response.json()["error"]


def test_https_redirect_preserves_path_query_and_uses_https_port():
    redirect_app = FastAPI()
    redirect_app.add_middleware(RedirectToHTTPSMiddleware, https_port=8443)

    @redirect_app.get("/api/pipelines")
    async def pipelines():
        return {"ok": True}

    redirect_client = TestClient(redirect_app, follow_redirects=False)
    response = redirect_client.get(
        "/api/pipelines?mode=sam",
        headers={"host": "example.test:8000"},
    )

    assert response.status_code == 308
    assert response.headers["location"] == "https://example.test:8443/api/pipelines?mode=sam"
