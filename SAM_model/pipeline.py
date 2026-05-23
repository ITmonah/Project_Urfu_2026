import os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from segment_anything import sam_model_registry


YOLO_PATH = "Yolo26s_kgo.pt"
SAM_FULL_MODEL_PATH = "sam_model.pth"

BACKGROUND = 0
FLOOR = 1
WALL = 2
GARBAGE = 3

DEFAULT_THRESHOLD = 0.70


class SAMSemanticMulticlass(nn.Module):

    def __init__(self, sam_base, num_classes=4):
        super().__init__()
        self.image_encoder = sam_base.image_encoder
        for param in self.image_encoder.parameters():
            param.requires_grad = False

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, num_classes, kernel_size=1)
        )

    def forward(self, x):
        features = self.image_encoder(x)
        logits = self.decoder(features)
        logits = torch.nn.functional.interpolate(
            logits, size=(1024, 1024), mode='bilinear', align_corners=False
        )
        return logits



det_model = None
segment_model = None
device = None
transform = None


def _check_models_loaded():
    if det_model is None or segment_model is None:
        raise RuntimeError(
            "Models not loaded. Call load_models() first."
        )



def load_models(yolo_path=YOLO_PATH, sam_path=SAM_FULL_MODEL_PATH):
    global det_model, segment_model, device, transform

    if not os.path.exists(yolo_path):
        raise FileNotFoundError(f"YOLO model not found: {yolo_path}")
    if not os.path.exists(sam_path):
        raise FileNotFoundError(f"SAM model not found: {sam_path}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"DEVICE: {device}")

    # YOLO
    det_model = YOLO(yolo_path)
    det_model.to(device)

    # SAM
    sam_base = sam_model_registry['vit_b'](checkpoint=None)
    segment_model = SAMSemanticMulticlass(sam_base, num_classes=4).to(device)
    state_dict = torch.load(sam_path, map_location=device)
    segment_model.load_state_dict(state_dict)
    segment_model.eval()

    transform = A.Compose([
        A.LongestMaxSize(max_size=1024),
        A.PadIfNeeded(min_height=1024, min_width=1024,
                      border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0)),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    print("MODELS LOADED")


def predict_with_tta(input_tensor):
    predictions = []

    def _infer(t):
        with torch.no_grad():
            return segment_model(t)

    predictions.append(_infer(input_tensor))

    flipped_h = torch.flip(input_tensor, dims=[3])
    pred_h = _infer(flipped_h)
    predictions.append(torch.flip(pred_h, dims=[3]))

    flipped_v = torch.flip(input_tensor, dims=[2])
    pred_v = _infer(flipped_v)
    predictions.append(torch.flip(pred_v, dims=[2]))

    rotated_90 = torch.rot90(input_tensor, k=1, dims=[2, 3])
    pred_r90 = _infer(rotated_90)
    predictions.append(torch.rot90(pred_r90, k=-1, dims=[2, 3]))

    return torch.mean(torch.stack(predictions), dim=0)



def segment_kgp(image_bgr):
    h, w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    transformed = transform(image=image_rgb)
    input_tensor = transformed['image'].unsqueeze(0).to(device)

    logits = predict_with_tta(input_tensor)
    pred_map = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

    pred_resized = cv2.resize(
        pred_map.astype(np.uint8), (w, h),
        interpolation=cv2.INTER_NEAREST
    )
    waste_mask = (pred_resized == GARBAGE).astype(np.uint8) * 255
    return pred_resized, waste_mask



def clean_garbage_mask(garbage_mask, platform_area, min_component_ratio=0.0005):
    h, w = garbage_mask.shape[:2]
    k = max(3, min(15, int(min(h, w) * 0.005) | 1))
    kernel = np.ones((k, k), np.uint8)

    closed = cv2.morphologyEx(garbage_mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        closed, connectivity=8
    )
    cleaned_mask = np.zeros_like(closed)

    min_area = max(1, platform_area * min_component_ratio)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned_mask[labels == i] = 1

    return cleaned_mask.astype(np.uint8)


def calculate_fill_percentage(image_bgr, threshold=DEFAULT_THRESHOLD,
                               min_component_ratio=0.0005):

    pred_map, _ = segment_kgp(image_bgr)

    platform_mask = ((pred_map == FLOOR) | (pred_map == WALL)).astype(np.uint8)
    platform_area = int(np.sum(platform_mask))

    if platform_area == 0:
        return {
            'percentage': 0.0,
            'label': 'kgo_empty',
            'mask': np.zeros(pred_map.shape, dtype=np.uint8),
            'pred_map': pred_map
        }

    garbage_mask_raw = (pred_map == GARBAGE).astype(np.uint8)
    garbage_mask = clean_garbage_mask(
        garbage_mask_raw, platform_area, min_component_ratio
    )

    garbage_area = int(np.sum(garbage_mask))
    percentage = min(100.0, (garbage_area / platform_area) * 100.0)

    label = 'kgo_full' if percentage >= threshold * 100.0 else 'kgo_empty'

    return {
        'percentage': percentage,
        'label': label,
        'mask': (garbage_mask * 255).astype(np.uint8),
        'pred_map': pred_map
    }


def detect_platform(pil_img):
    _check_models_loaded()

    results = det_model(pil_img, verbose=False)
    if results[0].boxes is None or len(results[0].boxes) == 0:
        return None

    names = results[0].names
    target_id = next((i for i, n in names.items() if n == 'kgo_platform'), None)
    if target_id is None:
        return None

    best_box = max(
        (b for b in results[0].boxes if int(b.cls[0]) == target_id),
        key=lambda b: float(b.conf[0]),
        default=None
    )
    if best_box is None:
        return None

    x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())

    pad = 0.08
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - bw * pad))
    y1 = max(0, int(y1 - bh * pad))
    x2 = min(pil_img.width, int(x2 + bw * pad))
    y2 = min(pil_img.height, int(y2 + bh * pad))

    return x1, y1, x2, y2


def process_image(image_path, threshold=DEFAULT_THRESHOLD):
    _check_models_loaded()

    pil_img = Image.open(image_path).convert('RGB')
    box = detect_platform(pil_img)
    if box is None:
        return ""

    x1, y1, x2, y2 = box
    crop = pil_img.crop((x1, y1, x2, y2))
    crop_bgr = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2BGR)

    result = calculate_fill_percentage(crop_bgr, threshold=threshold)
    return result['label']


def process_and_visualize(image_path, output_dir=None, show_full=True,
                          show_crop=True, threshold=DEFAULT_THRESHOLD):
    _check_models_loaded()

    pil_img = Image.open(image_path).convert('RGB')
    box = detect_platform(pil_img)
    if box is None:
        raise ValueError(f"Platform not found in: {image_path}")

    x1, y1, x2, y2 = box
    crop_pil = pil_img.crop((x1, y1, x2, y2))
    crop_bgr = cv2.cvtColor(np.array(crop_pil), cv2.COLOR_RGB2BGR)

    result = calculate_fill_percentage(crop_bgr, threshold=threshold)
    percentage = result['percentage']
    label = result['label']
    waste_mask = result['mask']

    full_img = pil_img.copy()
    draw = ImageDraw.Draw(full_img)
    draw.rectangle([x1, y1, x2, y2], outline='red', width=5)

    text = f"{percentage:.1f}% - {label}"
    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except (IOError, OSError):
        font = ImageFont.load_default()

    tw, th = draw.textbbox((0, 0), text, font=font)[2:]
    tx, ty = x1, max(0, y1 - th - 15)
    draw.rectangle([tx, ty, tx + tw + 20, ty + th + 10], fill='black')
    draw.text((tx + 10, ty + 5), text, fill='white', font=font)

    crop_vis = crop_bgr.copy()
    red_overlay = np.zeros_like(crop_vis)
    red_overlay[waste_mask > 0] = [0, 0, 255]
    crop_vis = cv2.addWeighted(crop_vis, 1.0, red_overlay, 0.45, 0)
    cv2.putText(crop_vis, f"{percentage:.1f}% - {label}",
                (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    crop_rgb = cv2.cvtColor(crop_vis, cv2.COLOR_BGR2RGB)
    crop_pil_vis = Image.fromarray(crop_rgb)

    full_img_path = None
    crop_img_path = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        full_img_path = os.path.join(output_dir, 'full_result.jpg')
        crop_img_path = os.path.join(output_dir, 'crop_result.jpg')
        full_img.save(full_img_path)
        crop_pil_vis.save(crop_img_path)

    if show_full:
        full_img.show()
    if show_crop:
        crop_pil_vis.show()

    return {
        'label': label,
        'percentage': percentage,
        'mask': waste_mask,
        'full_img_path': full_img_path,
        'crop_img_path': crop_img_path
    }