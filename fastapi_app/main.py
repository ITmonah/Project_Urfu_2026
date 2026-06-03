import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from fastapi_app.inference import (
    DEFAULT_PIPELINE,
    ensure_all_checkpoints,
    image_to_data_url,
    list_available_pipelines,
    run_inference,
)


BASE_DIR = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.getenv("KGO_PRELOAD_MODEL_ASSETS", "0").lower() not in {"1", "true", "yes"}:
        yield
        return

    print("[model-assets] Preloading all model assets before serving requests.", flush=True)
    resolved = ensure_all_checkpoints(log_progress=True)
    print(f"[model-assets] Preload complete. Resolved {len(resolved)} asset(s).", flush=True)
    yield


app = FastAPI(title="KGO Pipeline UI", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


def render_index(
    request: Request,
    *,
    result=None,
    error: str | None = None,
    selected_pipeline: str = DEFAULT_PIPELINE,
    image_preview: str | None = None,
):
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "pipeline_modes": list_available_pipelines(),
            "result": result,
            "error": error,
            "selected_pipeline": selected_pipeline,
            "image_preview": image_preview,
        },
    )


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return render_index(request)


@app.get("/api/pipelines")
async def get_pipelines():
    return {"pipelines": list_available_pipelines(), "default_pipeline": DEFAULT_PIPELINE}


@app.get("/api/models")
async def get_models():
    pipelines = list_available_pipelines()
    return {"models": pipelines, "pipelines": pipelines, "default_pipeline": DEFAULT_PIPELINE}


@app.post("/api/predict")
async def predict_api(
    image: UploadFile = File(...),
    pipeline: str = Form(DEFAULT_PIPELINE),
    model: str | None = Form(None),
):
    image_bytes = await image.read()
    try:
        return run_inference(image_bytes, pipeline)
    except (FileNotFoundError, RuntimeError, ValueError, ImportError, ModuleNotFoundError) as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})


@app.post("/run", response_class=HTMLResponse)
async def run_from_form(
    request: Request,
    image: UploadFile = File(...),
    pipeline: str = Form(DEFAULT_PIPELINE),
    model: str | None = Form(None),
):
    image_bytes = await image.read()
    content_type = image.content_type or "image/jpeg"
    image_preview = image_to_data_url(image_bytes, content_type)

    try:
        result = run_inference(image_bytes, pipeline)
        return render_index(
            request,
            result=result,
            selected_pipeline=pipeline,
            image_preview=result["annotated_image"],
        )
    except (FileNotFoundError, RuntimeError, ValueError, ImportError, ModuleNotFoundError) as exc:
        return render_index(
            request,
            error=str(exc),
            selected_pipeline=pipeline,
            image_preview=image_preview,
        )
