import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
from PIL import Image
import base64
import io
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEBUG_IMAGE_PATH = ""
USE_DEBUG_IMAGE = False

current_dir = Path(__file__).parent
yolo_model = YOLO(str(current_dir / "yolo_source.pt"))

resnet_model = models.resnet50()
num_features = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(num_features, 2)
resnet_model.load_state_dict(torch.load(str(current_dir / "best_resnet_kgo.pth"), map_location=device))
resnet_model = resnet_model.to(device)
resnet_model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class_names = ['kgo_empty', 'kgo_full']

def process_images(images):
    if USE_DEBUG_IMAGE:
        if DEBUG_IMAGE_PATH:
            img = Image.open(DEBUG_IMAGE_PATH).convert("RGB")
            images = [img]
        else:
            return []
    results = []
    for img_input in images:
        if isinstance(img_input, str):
            img_data = base64.b64decode(img_input)
            img = Image.open(io.BytesIO(img_data)).convert("RGB")
        else:
            img = img_input
        yolo_results = yolo_model(img)
        if USE_DEBUG_IMAGE:
            yolo_results[0].save(str(current_dir / "debug.jpg"))
        detections = []
        for result in yolo_results:
            for box in result.boxes:
                cls = int(box.cls)
                if cls == 2:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    crop = img.crop((x1, y1, x2, y2))
                    detections.append(crop)
        if not detections:
            results.append([])
            continue
        img_results = []
        for crop in detections:
            input_tensor = transform(crop).unsqueeze(0).to(device)
            with torch.no_grad():
                output = resnet_model(input_tensor)
                pred = torch.argmax(output, dim=1).item()
                img_results.append(class_names[pred])
        results.append(img_results)
    if USE_DEBUG_IMAGE:
        print(results)
    return results