from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from fastapi_app.inference import image_to_data_url, list_available_models, run_inference


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PIPELINE = "yolo_classifier"
ACTUAL_PIPELINE_LABEL = "YOLO + классификатор"
PIPELINE_MODES = [
    {
        "name": "whole_image_classifier",
        "label": "Классификатор по всей картинке",
        "description": "Классификатор по всей картинке.",
    },
    {
        "name": DEFAULT_PIPELINE,
        "label": ACTUAL_PIPELINE_LABEL,
        "description": "Текущий рабочий сценарий: детекция объектов и классификация найденных кусков.",
    },
    {
        "name": "segmentation",
        "label": "Сегментация",
        "description": "Режим сегментации.",
    },
]

app = FastAPI(title="KGO Pipeline UI")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


def resolve_pipeline_mode(pipeline_name: str) -> dict[str, str]:
    for pipeline_mode in PIPELINE_MODES:
        if pipeline_mode["name"] == pipeline_name:
            return pipeline_mode

    available = ", ".join(item["name"] for item in PIPELINE_MODES)
    raise ValueError(f"Pipeline '{pipeline_name}' is not configured. Available: {available}")


def add_pipeline_context(result: dict, pipeline_name: str) -> dict:
    pipeline_mode = resolve_pipeline_mode(pipeline_name)
    result["selected_pipeline"] = pipeline_mode["name"]
    result["selected_pipeline_label"] = pipeline_mode["label"]
    result["actual_pipeline"] = DEFAULT_PIPELINE
    result["actual_pipeline_label"] = ACTUAL_PIPELINE_LABEL
    result["pipeline_notice"] = None

    if pipeline_mode["name"] != DEFAULT_PIPELINE:
        result["pipeline_notice"] = (
            "Выбор режима подготовлен на фронте. "
            "Текущий запуск использует существующий пайплайн YOLO + классификатор."
        )

    return result


def render_index(
    request: Request,
    *,
    result=None,
    error: str | None = None,
    selected_model: str = "convnext_tiny",
    selected_pipeline: str = DEFAULT_PIPELINE,
    image_preview: str | None = None,
):
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "models": list_available_models(),
            "pipeline_modes": PIPELINE_MODES,
            "result": result,
            "error": error,
            "selected_model": selected_model,
            "selected_pipeline": selected_pipeline,
            "image_preview": image_preview,
        },
    )


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return render_index(request)


@app.get("/api/models")
async def get_models():
    return {"models": list_available_models()}


@app.post("/api/predict")
async def predict_api(
    image: UploadFile = File(...),
    model: str = Form(...),
    pipeline: str = Form(DEFAULT_PIPELINE),
):
    image_bytes = await image.read()
    try:
        resolve_pipeline_mode(pipeline)
        result = run_inference(image_bytes, model)
        add_pipeline_context(result, pipeline)
    except (FileNotFoundError, ValueError) as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    return result


@app.post("/run", response_class=HTMLResponse)
async def run_from_form(
    request: Request,
    image: UploadFile = File(...),
    model: str = Form(...),
    pipeline: str = Form(DEFAULT_PIPELINE),
):
    image_bytes = await image.read()
    content_type = image.content_type or "image/jpeg"
    image_preview = image_to_data_url(image_bytes, content_type)

    try:
        resolve_pipeline_mode(pipeline)
        result = run_inference(image_bytes, model)
        add_pipeline_context(result, pipeline)
        return render_index(
            request,
            result=result,
            selected_model=model,
            selected_pipeline=pipeline,
            image_preview=result["annotated_image"],
        )
    except (FileNotFoundError, ValueError) as exc:
        return render_index(
            request,
            error=str(exc),
            selected_model=model,
            selected_pipeline=pipeline,
            image_preview=image_preview,
        )
