"""
pipeline.py – итоговый пайплайн оценки заполненности зоны КГО.
Использует YOLO для детекции зоны (только класс 'kgo_platform') и DeepLabV3+ (SMP) для сегментации мусора.
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp


class KGOFillPipeline:
    def __init__(
        self,
        yolo_path: str = "Yolo26s_kgo.pt",
        segm_path: str = "best_model.pth",
        segm_encoder: str = "se_resnext50_32x4d",
        segm_classes: int = 4,
        fill_threshold: float = 0.5,
        segm_input_size: int = 512,
        use_floor_only: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.fill_threshold = fill_threshold
        self.segm_input_size = segm_input_size
        self.use_floor_only = use_floor_only
        self.segm_classes = segm_classes

        self.yolo_model = YOLO(yolo_path)
        self.yolo_model.to(self.device)
        self.yolo_names = self.yolo_model.names

        self.segm_model = smp.DeepLabV3Plus(
            encoder_name=segm_encoder,
            encoder_weights=None,
            in_channels=3,
            classes=segm_classes,
            activation=None,
        )
        checkpoint = torch.load(segm_path, map_location=self.device)
        self.segm_model.load_state_dict(checkpoint["model_state_dict"])
        self.segm_model.to(self.device)
        self.segm_model.eval()

        self.segm_transform = A.Compose([
            A.Resize(segm_input_size, segm_input_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    def predict(self, image: np.ndarray) -> str:
        """Только статус (без маски)."""
        status, _, _, _ = self._predict_internal(image)
        return status

    def predict_with_mask(self, image: np.ndarray):
        """
        Возвращает (status, pred_mask, bbox, fill_ratio).
        status: 'kgo_full', 'kgo_empty' или ''
        pred_mask: np.ndarray (H,W) с классами 0..3, или None
        bbox: (x1, y1, x2, y2) или None
        fill_ratio: float, доля мусора в зоне КГО (0..1), или None если зона не найдена
        """
        return self._predict_internal(image)

    def _predict_internal(self, image: np.ndarray):
        results = self.yolo_model(image, verbose=False)
        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            return "", None, None, None

        # Фильтрация по классу 'kgo_platform'
        kgo_class_id = None
        for class_id, name in self.yolo_names.items():
            if name == "kgo_platform":
                kgo_class_id = class_id
                break

        if kgo_class_id is None:
            print("Предупреждение: класс 'kgo_platform' не найден в модели YOLO, будут использованы все детекции.")
            filtered_boxes = boxes
        else:
            class_ids = boxes.cls.cpu().numpy() if boxes.cls is not None else []
            mask = np.array([cid == kgo_class_id for cid in class_ids])
            if not mask.any():
                return "", None, None, None
            filtered_boxes = boxes[mask]

        best_idx = filtered_boxes.conf.argmax()
        x1, y1, x2, y2 = filtered_boxes.xyxy[best_idx].int().tolist()
        bbox = (x1, y1, x2, y2)

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return "", None, bbox, None

        mask_pred = self._segment(crop)
        ratio = self._calc_fill_ratio(mask_pred)
        status = "kgo_full" if ratio >= self.fill_threshold else "kgo_empty"
        return status, mask_pred, bbox, ratio

    def _segment(self, crop: np.ndarray) -> np.ndarray:
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        transformed = self.segm_transform(image=crop_rgb)
        input_tensor = transformed["image"].unsqueeze(0).to(self.device, dtype=torch.float32)
        with torch.no_grad():
            logits = self.segm_model(input_tensor)
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
        return pred

    def _calc_fill_ratio(self, mask: np.ndarray) -> float:
        garbage = np.sum(mask == 3)
        if self.use_floor_only:
            total_zone = np.sum(mask == 1)
        else:
            total_zone = np.sum((mask == 1) | (mask == 2) | (mask == 3))
        if total_zone == 0:
            return 0.0
        return garbage / total_zone 
    
    