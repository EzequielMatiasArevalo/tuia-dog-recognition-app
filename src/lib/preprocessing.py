"""Preprocesamiento compartidos entre notebooks y servicios.

Convenciones:
  - Las imagenes en memoria dentro de los servicios llegan en BGR uint8 (OpenCV).
  - Antes de aplicar normalizacion ImageNet se convierten explicitamente a RGB.
  - Misma resolucion (IMAGE_SIZE x IMAGE_SIZE) y estadisticas ImageNet en
    entrenamiento, validacion, evaluacion e inferencia.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


def normalize_breed_name(raw_name: str) -> str:
    """Normaliza espacios multiples en nombres de carpetas (ej. 'American  Spaniel')."""
    return " ".join(raw_name.strip().split())


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """Convierte una imagen BGR uint8 (OpenCV) a RGB uint8 validando dimensiones."""
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("La imagen recibida esta vacia o es None.")
    if image_bgr.ndim == 2:
        return cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2RGB)
    if image_bgr.shape[2] == 4:
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def get_eval_transforms(image_size: int = 224) -> A.Compose:
    """Pipeline deterministico para validacion, test e inferencia en produccion.

    Usa Resize directo a (image_size, image_size) en lugar de CenterCrop para
    no recortar orejas o hocicos en los bounding boxes ajustados que entrega YOLO.
    """
    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size, interpolation=cv2.INTER_LINEAR),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def get_train_transforms(image_size: int = 224) -> A.Compose:
    """Pipeline de data augmentation + preprocesamiento exclusivo para train."""
    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size, interpolation=cv2.INTER_LINEAR),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=20, border_mode=cv2.BORDER_REFLECT_101, p=0.6),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
            A.HueSaturationValue(
                hue_shift_limit=8, sat_shift_limit=15, val_shift_limit=15, p=0.4
            ),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MotionBlur(blur_limit=(3, 5), p=1.0),
                ],
                p=0.25,
            ),
            A.GaussNoise(p=0.25),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def get_visual_augmentation_pipeline(image_size: int = 224) -> A.Compose:
    """Mismas augmentations de train sin Normalize/ToTensorV2 para inspeccion visual."""
    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size, interpolation=cv2.INTER_LINEAR),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=20, border_mode=cv2.BORDER_REFLECT_101, p=0.6),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
            A.HueSaturationValue(
                hue_shift_limit=8, sat_shift_limit=15, val_shift_limit=15, p=0.4
            ),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MotionBlur(blur_limit=(3, 5), p=1.0),
                ],
                p=0.25,
            ),
            A.GaussNoise(p=0.25),
        ]
    )


def preprocess_bgr_image(image_bgr: np.ndarray, image_size: int = 224) -> torch.Tensor:
    """Transforma una imagen BGR (OpenCV) en tensor RGB normalizado (3, H, W)."""
    image_rgb = bgr_to_rgb(image_bgr)
    transform = get_eval_transforms(image_size=image_size)
    return transform(image=image_rgb)["image"]