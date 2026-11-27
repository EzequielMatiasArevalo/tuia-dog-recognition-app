from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import numpy as np
import onnxruntime
import torch

from lib.preprocessing import normalize_breed_name, preprocess_bgr_image
from lib.schemas import ClassifyResult, DetectResult, DogDetection
from lib.services.classifier_service import ClassifierService

logger = logging.getLogger(__name__)


class DetectionService:
    """Etapa 3: pipeline de deteccion y clasificacion.

    Funciones implementadas:
      - detect_dogs(image)
      - classify_detected_dog(crop)
    """

    def __init__(
        self,
        classifier: ClassifierService,
        yolo_model: str,
        conf_threshold: float,
        dog_class_id: int,
    ) -> None:
        self.classifier = classifier
        self.yolo_model_name = yolo_model
        self.conf_threshold = conf_threshold
        self.dog_class_id = dog_class_id
        self._detector: Any = None
        self._class_names: list[str] | None = None

    @staticmethod
    def _clip_xyxy(
        x1: int, y1: int, x2: int, y2: int, height: int, width: int
    ) -> tuple[int, int, int, int]:
        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height))
        if x2 <= x1:
            x2 = min(x1 + 1, width)
        if y2 <= y1:
            y2 = min(y1 + 1, height)
        return x1, y1, x2, y2

    def _load_image(self, source_path: str) -> np.ndarray:
        image = cv2.imread(str(source_path))
        if image is None:
            raise ValueError(f"Could not read image: {source_path}")
        return image

    def _get_detector(self) -> Any:
        """Carga y cachea el detector YOLO pre-entrenado una sola vez."""
        if self._detector is None:
            from ultralytics import YOLO

            model_path = Path(self.yolo_model_name)
            if model_path.parent != Path("."):
                model_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(
                "Loading YOLO detector '%s' (conf=%.2f, dog_class_id=%d)",
                self.yolo_model_name,
                self.conf_threshold,
                self.dog_class_id,
            )
            self._detector = YOLO(self.yolo_model_name)
        return self._detector

    def _resolve_class_names(self, model: Any) -> list[str]:
        """Obtiene la lista ordenada de razas desde el checkpoint o classes.json."""
        ckpt_classes = getattr(model, "class_names", None)
        if ckpt_classes:
            return [str(c) for c in ckpt_classes]

        if self._class_names is not None:
            return self._class_names

        classes_json = self.classifier.dataset_path.parent / "classes.json"
        if classes_json.is_file():
            payload = json.loads(classes_json.read_text(encoding="utf-8"))
            self._class_names = [str(c) for c in payload["classes"]]
            return self._class_names

        train_dir = self.classifier.dataset_path / "train"
        if train_dir.is_dir():
            self._class_names = sorted(
                normalize_breed_name(p.name) for p in train_dir.iterdir() if p.is_dir()
            )
            return self._class_names

        raise ValueError("No se pudo resolver la lista de clases del clasificador.")

    # ------------------------------------------------------------------
    # Etapa 3: funciones implementadas
    # ------------------------------------------------------------------

    def detect_dogs(self, image: np.ndarray) -> list[tuple[tuple[int, int, int, int], float]]:
        """Detecta todos los perros presentes en una imagen BGR uint8 usando YOLO."""
        if image is None or image.size == 0:
            raise ValueError("Input image is empty.")

        height, width = image.shape[:2]
        detector = self._get_detector()
        results = detector.predict(
            source=image,
            conf=self.conf_threshold,
            classes=[self.dog_class_id],
            verbose=False,
        )

        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return []

        boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()

        detections: list[tuple[tuple[int, int, int, int], float]] = []
        for box, conf in zip(boxes_xyxy, confs):
            x1, y1, x2, y2 = self._clip_xyxy(
                int(round(float(box[0]))),
                int(round(float(box[1]))),
                int(round(float(box[2]))),
                int(round(float(box[3]))),
                height,
                width,
            )
            detections.append(((x1, y1, x2, y2), float(conf)))

        detections.sort(key=lambda item: item[1], reverse=True)
        return detections

    def classify_detected_dog(self, crop: np.ndarray) -> tuple[str, float]:
        """Clasifica la raza de un recorte BGR uint8 usando el clasificador activo."""
        if crop is None or crop.size == 0:
            return ("unknown", 0.0)

        tensor = preprocess_bgr_image(crop, image_size=self.classifier.image_size)
        batch = tensor.unsqueeze(0)
        model = self.classifier.load_model()

        if isinstance(model, onnxruntime.InferenceSession):
            input_name = model.get_inputs()[0].name
            logits = model.run(None, {input_name: batch.numpy()})[0][0]
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)
        else:
            device = next(model.parameters()).device
            model.eval()
            with torch.no_grad():
                logits = model(batch.to(device))
                probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        pred_idx = int(np.argmax(probs))
        score = float(probs[pred_idx])
        class_names = self._resolve_class_names(model)
        breed = class_names[pred_idx] if 0 <= pred_idx < len(class_names) else "unknown"
        return (breed, score)

    # ------------------------------------------------------------------
    # Orquestacion provista
    # ------------------------------------------------------------------

    def classify_image(
        self, source_path: str, output_path: Path, model_name: str | None = None
    ) -> str:
        """Clasifica la imagen completa con el modelo entrenado (pestaña Etapa 2)."""
        image = self._load_image(source_path)
        previous = self.classifier.active_model_name
        if model_name:
            self.classifier.set_active_model(model_name)
        try:
            breed, score = self.classify_detected_dog(image)
        finally:
            self.classifier.set_active_model(previous)
        payload = ClassifyResult(
            source_path=source_path,
            model=model_name or self.classifier.active_model_name,
            breed=breed,
            score=round(float(score), 4),
        )
        output_path.mkdir(parents=True, exist_ok=True)
        result_file = output_path / f"result-{uuid4()}.json"
        result_file.write_text(
            json.dumps(payload.model_dump(), ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        return str(result_file)

    def predict(self, source_path: str, output_path: Path) -> str:
        """Flujo completo: deteccion -> bounding boxes -> recortes -> clasificacion."""
        image = self._load_image(source_path)
        height, width = image.shape[:2]

        detections: list[DogDetection] = []
        for (box, det_score) in self.detect_dogs(image):
            x1, y1, x2, y2 = self._clip_xyxy(*[int(v) for v in box], height, width)
            crop = image[y1:y2, x1:x2]
            breed, breed_score = self.classify_detected_dog(crop)
            detections.append(
                DogDetection(
                    bbox=[x1, y1, x2, y2],
                    det_score=round(float(det_score), 4),
                    breed=breed,
                    breed_score=round(float(breed_score), 4),
                )
            )

        detected_breeds = sorted({item.breed for item in detections if item.breed != "unknown"})
        payload = DetectResult(
            source_path=source_path,
            detections=detections,
            detected_breeds=detected_breeds,
        )
        output_path.mkdir(parents=True, exist_ok=True)
        result_file = output_path / f"result-{uuid4()}.json"
        result_file.write_text(
            json.dumps(payload.model_dump(), ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        return str(result_file)