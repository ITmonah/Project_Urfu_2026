import base64
import os
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from fastapi_app.model_assets import (
    ModelAssetSpec,
    ensure_model_asset,
    model_asset_status,
    resolve_model_asset_path,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
ULTRALYTICS_CONFIG_DIR = ROOT_DIR / ".tmp" / "ultralytics"
ULTRALYTICS_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_CONFIG_DIR))


@dataclass(frozen=True)
class CheckpointSpec:
    label: str
    env_var: str
    default_path: Path
    asset_name: str
    url_env_var: str

    def to_model_asset(self) -> ModelAssetSpec:
        return ModelAssetSpec(
            label=self.label,
            path_env_var=self.env_var,
            default_path=self.default_path,
            asset_name=self.asset_name,
            url_env_var=self.url_env_var,
        )


@dataclass(frozen=True)
class PipelineSpec:
    name: str
    label: str
    description: str
    checkpoints: tuple[CheckpointSpec, ...]


PIPELINE_SPECS: tuple[PipelineSpec, ...] = (
    PipelineSpec(
        name="new_classifier_nodino",
        label="NewClassificatorNoDino",
        description="YOLO + классификатор, включая классификатор по полному изображению.",
        checkpoints=(
            CheckpointSpec(
                "YOLO",
                "KGO_NCD_YOLO_CHECKPOINT",
                ROOT_DIR / "NewClassificatorNoDino" / "Yolo26s_kgo.pt",
                "Yolo26s_kgo.pt",
                "KGO_NCD_YOLO_URL",
            ),
            CheckpointSpec(
                "Crop classifier",
                "KGO_NCD_CROP_CLASSIFIER_CHECKPOINT",
                ROOT_DIR / "NewClassificatorNoDino" / "best_efficientnet_v2_s_kgo.pth",
                "best_efficientnet_v2_s_kgo.pth",
                "KGO_NCD_CROP_CLASSIFIER_URL",
            ),
            CheckpointSpec(
                "Full image classifier",
                "KGO_NCD_FULL_CLASSIFIER_CHECKPOINT",
                ROOT_DIR / "NewClassificatorNoDino" / "best_efficientnet_for_full.pth",
                "best_efficientnet_for_full.pth",
                "KGO_NCD_FULL_CLASSIFIER_URL",
            ),
        ),
    ),
    PipelineSpec(
        name="sam",
        label="SAM_model",
        description="Foundation SAM segmentation pipeline.",
        checkpoints=(
            CheckpointSpec(
                "YOLO",
                "KGO_SAM_YOLO_CHECKPOINT",
                ROOT_DIR / "SAM_model" / "Yolo26s_kgo.pt",
                "Yolo26s_kgo.pt",
                "KGO_SAM_YOLO_URL",
            ),
            CheckpointSpec(
                "SAM",
                "KGO_SAM_CHECKPOINT",
                ROOT_DIR / "SAM_model" / "sam_model.pth",
                "sam_model.pth",
                "KGO_SAM_URL",
            ),
        ),
    ),
    PipelineSpec(
        name="smp",
        label="SMP_model",
        description="Segmentation Models PyTorch pipeline.",
        checkpoints=(
            CheckpointSpec(
                "YOLO",
                "KGO_SMP_YOLO_CHECKPOINT",
                ROOT_DIR / "SMP_model" / "Yolo26s_kgo.pt",
                "Yolo26s_kgo.pt",
                "KGO_SMP_YOLO_URL",
            ),
            CheckpointSpec(
                "SMP",
                "KGO_SMP_CHECKPOINT",
                ROOT_DIR / "SMP_model" / "best_model_SMP.pth",
                "best_model_SMP.pth",
                "KGO_SMP_URL",
            ),
        ),
    ),
)
DEFAULT_PIPELINE = "new_classifier_nodino"


def image_to_data_url(image_bytes: bytes, content_type: str) -> str:
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{content_type};base64,{encoded}"


def pil_image_to_data_url(image: Image.Image, image_format: str = "PNG") -> str:
    buffer = BytesIO()
    image.save(buffer, format=image_format)
    return image_to_data_url(buffer.getvalue(), "image/png")


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT_DIR / path


def resolve_checkpoint(spec: CheckpointSpec) -> Path:
    return resolve_model_asset_path(spec.to_model_asset())


def get_pipeline_spec(pipeline_name: str) -> PipelineSpec:
    for spec in PIPELINE_SPECS:
        if spec.name == pipeline_name:
            return spec

    available = ", ".join(item.name for item in PIPELINE_SPECS)
    raise ValueError(f"Pipeline '{pipeline_name}' is not configured. Available: {available}")


def pipeline_checkpoints(spec: PipelineSpec) -> list[dict[str, Any]]:
    checkpoints = []
    for checkpoint in spec.checkpoints:
        checkpoints.append(model_asset_status(checkpoint.to_model_asset()))
    return checkpoints


def ensure_checkpoints(spec: PipelineSpec, *, log_progress: bool = False) -> dict[str, Path]:
    return {
        checkpoint.env_var: ensure_model_asset(checkpoint.to_model_asset(), log_progress=log_progress)
        for checkpoint in spec.checkpoints
    }


def ensure_all_checkpoints(*, log_progress: bool = False) -> list[dict[str, str]]:
    resolved = []
    for pipeline_spec in PIPELINE_SPECS:
        for checkpoint in pipeline_spec.checkpoints:
            path = ensure_model_asset(checkpoint.to_model_asset(), log_progress=log_progress)
            resolved.append(
                {
                    "pipeline": pipeline_spec.name,
                    "label": checkpoint.label,
                    "asset_name": checkpoint.asset_name,
                    "path": str(path),
                }
            )
    return resolved


def list_available_pipelines() -> list[dict[str, Any]]:
    pipelines = []
    for spec in PIPELINE_SPECS:
        checkpoints = pipeline_checkpoints(spec)
        pipelines.append(
            {
                "name": spec.name,
                "label": spec.label,
                "description": spec.description,
                "available": all(item["available"] for item in checkpoints),
                "checkpoints": checkpoints,
            }
        )
    return pipelines


def list_available_models() -> list[dict[str, Any]]:
    return list_available_pipelines()


def draw_label_overlay(image: Image.Image, label: str, title: str | None = None) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()
    display_label = label or "no kgo_platform"
    text = f"{title}: {display_label}" if title else display_label
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    padding = 8
    x, y = 12, 12
    draw.rectangle((x, y, x + text_width + padding * 2, y + text_height + padding * 2), fill="black")
    draw.text((x + padding, y + padding), text, fill="white", font=font)
    return annotated


def normalize_result(
    *,
    spec: PipelineSpec,
    label: str,
    annotated_image: Image.Image,
    checkpoints: dict[str, Path],
    bbox: tuple[int, int, int, int] | None = None,
    fill_ratio: float | None = None,
) -> dict[str, Any]:
    prediction: dict[str, Any] | None = None
    if label:
        prediction = {"label": label, "bbox": None, "fill_percentage": None}
        if bbox is not None:
            prediction["bbox"] = [round(value, 2) for value in bbox]
        if fill_ratio is not None:
            prediction["fill_percentage"] = round(fill_ratio * 100, 2)

    return {
        "pipeline": spec.name,
        "pipeline_label": spec.label,
        "selected_pipeline": spec.name,
        "selected_pipeline_label": spec.label,
        "actual_pipeline": spec.name,
        "actual_pipeline_label": spec.label,
        "label": label,
        "bbox": [round(value, 2) for value in bbox] if bbox is not None else None,
        "fill_percentage": round(fill_ratio * 100, 2) if fill_ratio is not None else None,
        "checkpoints": [
            {
                "label": checkpoint.label,
                "env_var": checkpoint.env_var,
                "url_env_var": checkpoint.url_env_var,
                "path": str(checkpoints[checkpoint.env_var]),
                "available": checkpoints[checkpoint.env_var].exists(),
                "source_url": model_asset_status(checkpoint.to_model_asset())["source_url"],
            }
            for checkpoint in spec.checkpoints
        ],
        "predictions": [prediction] if prediction else [],
        "annotated_image": pil_image_to_data_url(annotated_image),
    }


def save_temp_image(image: Image.Image, directory: str | Path) -> Path:
    image_path = Path(directory) / "input.png"
    image.save(image_path)
    return image_path


@lru_cache(maxsize=1)
def load_new_classifier_nodino(yolo_path: str, cls_path: str, full_cls_path: str):
    from NewClassificatorNoDino import pipeline as ncd_pipeline

    ncd_pipeline.load_models(yolo_path=yolo_path, cls_path=cls_path, full_cls_path=full_cls_path)
    return ncd_pipeline


def run_new_classifier_nodino(spec: PipelineSpec, image: Image.Image, checkpoints: dict[str, Path]) -> dict[str, Any]:
    ncd_pipeline = load_new_classifier_nodino(
        yolo_path=str(checkpoints["KGO_NCD_YOLO_CHECKPOINT"]),
        cls_path=str(checkpoints["KGO_NCD_CROP_CLASSIFIER_CHECKPOINT"]),
        full_cls_path=str(checkpoints["KGO_NCD_FULL_CLASSIFIER_CHECKPOINT"]),
    )
    with tempfile.TemporaryDirectory(prefix="kgo_ncd_") as temp_dir:
        image_path = save_temp_image(image, temp_dir)
        label = ncd_pipeline.process_image(str(image_path))

    annotated = draw_label_overlay(image, label, spec.label)
    return normalize_result(spec=spec, label=label, annotated_image=annotated, checkpoints=checkpoints)


@lru_cache(maxsize=1)
def load_sam_pipeline(yolo_path: str, sam_path: str):
    from SAM_model import pipeline as sam_pipeline

    sam_pipeline.load_models(yolo_path=yolo_path, sam_path=sam_path)
    return sam_pipeline


def run_sam(spec: PipelineSpec, image: Image.Image, checkpoints: dict[str, Path]) -> dict[str, Any]:
    sam_pipeline = load_sam_pipeline(
        yolo_path=str(checkpoints["KGO_SAM_YOLO_CHECKPOINT"]),
        sam_path=str(checkpoints["KGO_SAM_CHECKPOINT"]),
    )
    with tempfile.TemporaryDirectory(prefix="kgo_sam_") as temp_dir:
        image_path = save_temp_image(image, temp_dir)
        try:
            result = sam_pipeline.process_and_visualize(
                image_path=str(image_path),
                output_dir=temp_dir,
                show_full=False,
                show_crop=False,
            )
            label = result["label"]
            fill_ratio = result["percentage"] / 100.0
            full_img_path = result.get("full_img_path")
            annotated = Image.open(full_img_path).convert("RGB") if full_img_path else draw_label_overlay(image, label, spec.label)
        except ValueError:
            label = ""
            fill_ratio = None
            annotated = draw_label_overlay(image, label, spec.label)

    return normalize_result(
        spec=spec,
        label=label,
        annotated_image=annotated,
        checkpoints=checkpoints,
        fill_ratio=fill_ratio,
    )


def smp_overlay_image(image: Image.Image, bbox, mask, fill_ratio: float | None) -> Image.Image:
    import cv2
    import numpy as np

    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    vis = image_bgr.copy()

    if bbox is None:
        return draw_label_overlay(image, "", "SMP_model")

    x1, y1, x2, y2 = bbox
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
    crop = image_bgr[y1:y2, x1:x2].copy()
    if mask is not None and crop.size > 0:
        colors = {
            1: (255, 0, 0),
            2: (0, 255, 0),
            3: (0, 0, 255),
        }
        mask_resized = cv2.resize(
            mask.astype(np.uint8),
            (crop.shape[1], crop.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        overlay = crop.copy()
        for class_id, color in colors.items():
            overlay[mask_resized == class_id] = color
        vis[y1:y2, x1:x2] = cv2.addWeighted(crop, 0.5, overlay, 0.5, 0)

    text = f"Fill: {fill_ratio:.1%}" if fill_ratio is not None else "Fill: N/A"
    cv2.putText(vis, text, (15, max(35, vis.shape[0] - 20)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    return Image.fromarray(vis_rgb)


@lru_cache(maxsize=1)
def load_smp_pipeline(yolo_path: str, segm_path: str):
    from SMP_model.pipeline import KGOFillPipeline

    return KGOFillPipeline(yolo_path=yolo_path, segm_path=segm_path)


def run_smp(spec: PipelineSpec, image: Image.Image, checkpoints: dict[str, Path]) -> dict[str, Any]:
    import cv2
    import numpy as np

    pipeline = load_smp_pipeline(
        yolo_path=str(checkpoints["KGO_SMP_YOLO_CHECKPOINT"]),
        segm_path=str(checkpoints["KGO_SMP_CHECKPOINT"]),
    )
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    label, mask, bbox, fill_ratio = pipeline.predict_with_mask(image_bgr)
    annotated = smp_overlay_image(image, bbox, mask, fill_ratio)
    return normalize_result(
        spec=spec,
        label=label,
        annotated_image=annotated,
        checkpoints=checkpoints,
        bbox=bbox,
        fill_ratio=fill_ratio,
    )


@lru_cache(maxsize=3)
def get_runner(pipeline_name: str):
    runners = {
        "new_classifier_nodino": run_new_classifier_nodino,
        "sam": run_sam,
        "smp": run_smp,
    }
    if pipeline_name not in runners:
        get_pipeline_spec(pipeline_name)
    return runners[pipeline_name]


def run_inference(image_bytes: bytes, pipeline_name: str = DEFAULT_PIPELINE) -> dict[str, Any]:
    spec = get_pipeline_spec(pipeline_name)
    checkpoints = ensure_checkpoints(spec)
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    runner = get_runner(spec.name)
    return runner(spec, image, checkpoints)
