from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from fastapi_app.inference import image_to_data_url, list_available_models, run_inference


BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="KGO Pipeline UI")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


def render_index(
    request: Request,
    *,
    result=None,
    error: str | None = None,
    selected_model: str = "convnext_tiny",
    image_preview: str | None = None,
):
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "models": list_available_models(),
            "result": result,
            "error": error,
            "selected_model": selected_model,
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
async def predict_api(image: UploadFile = File(...), model: str = Form(...)):
    image_bytes = await image.read()
    try:
        result = run_inference(image_bytes, model)
    except (FileNotFoundError, ValueError) as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    return result


@app.post("/run", response_class=HTMLResponse)
async def run_from_form(request: Request, image: UploadFile = File(...), model: str = Form(...)):
    image_bytes = await image.read()
    content_type = image.content_type or "image/jpeg"
    image_preview = image_to_data_url(image_bytes, content_type)

    try:
        result = run_inference(image_bytes, model)
        return render_index(
            request,
            result=result,
            selected_model=model,
            image_preview=image_preview,
        )
    except (FileNotFoundError, ValueError) as exc:
        return render_index(
            request,
            error=str(exc),
            selected_model=model,
            image_preview=image_preview,
        )
