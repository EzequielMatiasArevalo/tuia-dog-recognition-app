from __future__ import annotations

import copy
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import onnxruntime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from lib.evaluation.metrics import precision_recall_f1, specificity
from lib.models.cnn_custom import CustomCNN, ResNet18BreedClassifier
from lib.preprocessing import (
    bgr_to_rgb,
    get_eval_transforms,
    get_train_transforms,
    normalize_breed_name,
    preprocess_bgr_image,
)

logger = logging.getLogger(__name__)


class DogBreedSplitDataset(Dataset):
    """Dataset PyTorch sobre las particiones generadas en 1-pre-processing.ipynb."""

    def __init__(self, records: list[dict[str, Any]], data_dir: Path, transform: Any) -> None:
        self.records = records
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        rec = self.records[idx]
        img_path = self.data_dir / rec["rel_path"]
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise ValueError(f"Could not read dataset image: {img_path}")
        img_rgb = bgr_to_rgb(img_bgr)
        tensor = self.transform(image=img_rgb)["image"]
        return tensor, int(rec["label_idx"])


class ClassifierService:
    """Entrenamiento y comparacion de modelos de clasificacion.

    Funciones implementadas:
      - train_classifier()
      - evaluate_classifier()
      - extract_custom_embedding(image)
    """

    def __init__(
        self,
        checkpoints: dict[str, Path],
        image_size: int,
        dataset_path: Path,
        output_path: Path,
        active_model: str = "resnet18_finetuned",
        embedding_dim: int = 512,
    ) -> None:
        self.checkpoints = checkpoints
        self.image_size = image_size
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.active_model_name = active_model
        self.embedding_dim = embedding_dim
        self._loaded: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Infraestructura provista
    # ------------------------------------------------------------------

    def set_active_model(self, name: str) -> None:
        """Define que checkpoint usan extract_custom_embedding y la clasificacion."""
        if name not in self.checkpoints:
            raise ValueError(f"Unknown model '{name}'. Expected one of: {sorted(self.checkpoints)}")
        self.active_model_name = name

    @property
    def active_checkpoint(self) -> Path:
        return self.checkpoints[self.active_model_name]

    def load_model(self, name: str | None = None) -> Any:
        """Carga (con cache) el checkpoint del modelo indicado o del activo."""
        key = name or self.active_model_name
        if key in self._loaded:
            return self._loaded[key]
        path = self.checkpoints[key]
        if not path.exists():
            raise ValueError(
                f"Checkpoint not found: {path}. Entrena el modelo (Etapa 2) y guardalo en esa ruta."
            )
        suf = path.suffix.lower()
        if suf == ".pth":
            model = torch.load(path, map_location="cpu", weights_only=False)
        elif suf == ".onnx":
            model = onnxruntime.InferenceSession(str(path))
        else:
            raise ValueError(f"Unsupported model format (expected .pth or .onnx): {path}")
        self._loaded[key] = model
        return model

    # ------------------------------------------------------------------
    # Helpers internos de datos y semillas
    # ------------------------------------------------------------------

    @staticmethod
    def _seed_everything(seed: int = 42) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _load_splits_and_classes(self) -> tuple[pd.DataFrame, list[str]]:
        """Carga `data/splits.csv` y `data/classes.json` generados en Parte 1."""
        data_dir = self.dataset_path.parent
        splits_csv = data_dir / "splits.csv"
        classes_json = data_dir / "classes.json"

        if splits_csv.is_file() and classes_json.is_file():
            df = pd.read_csv(splits_csv)
            meta = json.loads(classes_json.read_text(encoding="utf-8"))
            class_names: list[str] = meta["classes"]
            logger.info("Loaded splits from %s (%d rows, %d classes)", splits_csv, len(df), len(class_names))
            return df, class_names

        # Fallback reproducible si splits.csv no existe aun
        logger.warning("splits.csv not found at %s; scanning %s directly", splits_csv, self.dataset_path)
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        rows: list[dict[str, Any]] = []
        for split in ("train", "valid", "test"):
            split_dir = self.dataset_path / split
            if not split_dir.is_dir():
                continue
            for breed_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
                breed = normalize_breed_name(breed_dir.name)
                for img_path in sorted(breed_dir.iterdir()):
                    if img_path.is_file() and img_path.suffix.lower() in exts:
                        rows.append(
                            {
                                "rel_path": img_path.relative_to(data_dir).as_posix(),
                                "breed": breed,
                                "split": split,
                            }
                        )
        df = pd.DataFrame(rows)
        class_names = sorted(df["breed"].unique())
        breed_to_idx = {b: i for i, b in enumerate(class_names)}
        df["label_idx"] = df["breed"].map(breed_to_idx)
        return df, class_names

    # ------------------------------------------------------------------
    # Funciones implementadas
    # ------------------------------------------------------------------

    def train_classifier(self) -> None:
        """Entrena el modelo activo (resnet18_finetuned o cnn_custom) y guarda el mejor checkpoint."""
        self._seed_everything(42)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        df, class_names = self._load_splits_and_classes()
        num_classes = len(class_names)
        data_dir = self.dataset_path.parent

        train_records = df[df["split"] == "train"].to_dict(orient="records")
        valid_records = df[df["split"] == "valid"].to_dict(orient="records")

        train_ds = DogBreedSplitDataset(
            train_records, data_dir, transform=get_train_transforms(self.image_size)
        )
        valid_ds = DogBreedSplitDataset(
            valid_records, data_dir, transform=get_eval_transforms(self.image_size)
        )

        
        num_workers = 0 if os.name == "nt" else 2
        pin_mem = torch.cuda.is_available()

        if self.active_model_name == "resnet18_finetuned":
            batch_size = 32
            epochs = 15
            patience = 5
            weight_decay = 1e-4
            label_smoothing = 0.1

            model: nn.Module = ResNet18BreedClassifier(
                num_classes=num_classes,
                embedding_dim=self.embedding_dim,
                dropout=0.3,
                pretrained=True,
                class_names=class_names,
            ).to(device)

            # Fine-tuning con learning rates diferenciados: backbone (1e-4) vs cabeza nueva (1e-3)
            optimizer = torch.optim.AdamW(
                [
                    {"params": model.backbone.parameters(), "lr": 1e-4},
                    {"params": model.embedding_proj.parameters(), "lr": 1e-3},
                    {"params": model.head.parameters(), "lr": 1e-3},
                ],
                weight_decay=weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
            hparams = {
                "architecture": "ResNet18BreedClassifier (ImageNet pretrained)",
                "strategy": "differential_lr_finetuning",
                "lr_backbone": 1e-4,
                "lr_head": 1e-3,
                "batch_size": batch_size,
                "epochs": epochs,
                "optimizer": "AdamW",
                "weight_decay": weight_decay,
                "scheduler": "CosineAnnealingLR",
                "label_smoothing": label_smoothing,
                "dropout": 0.3,
                "early_stopping_patience": patience,
            }
        elif self.active_model_name == "cnn_custom":
            batch_size = 32
            epochs = 35
            patience = 8
            lr = 1e-3
            weight_decay = 1e-4
            label_smoothing = 0.1

            model = CustomCNN(
                num_classes=num_classes,
                embedding_dim=self.embedding_dim,
                dropout=0.4,
                class_names=class_names,
            ).to(device)

            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
            hparams = {
                "architecture": "CustomCNN (5 ConvBlocks + GAP + 512d Embedding)",
                "strategy": "from_scratch",
                "lr": lr,
                "batch_size": batch_size,
                "epochs": epochs,
                "optimizer": "AdamW",
                "weight_decay": weight_decay,
                "scheduler": "CosineAnnealingLR",
                "label_smoothing": label_smoothing,
                "dropout": 0.4,
                "early_stopping_patience": patience,
            }
        else:
            raise ValueError(f"Unsupported model for training: {self.active_model_name}")

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_mem,
            drop_last=True,
        )
        valid_loader = DataLoader(
            valid_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_mem,
        )

        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        best_val_acc = -1.0
        best_val_loss = float("inf")
        best_epoch = 0
        epochs_no_improve = 0
        best_state = copy.deepcopy(model.state_dict())

        history: dict[str, list[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "lr": [],
        }

        logger.info(
            "Starting training for '%s' on %s (train=%d, valid=%d, epochs=%d)",
            self.active_model_name,
            device,
            len(train_ds),
            len(valid_ds),
            epochs,
        )
        t0 = time.perf_counter()

        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            running_correct = 0
            total_train = 0

            for inputs, targets in train_loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                optimizer.zero_grad()
                logits = model(inputs)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()

                running_loss += float(loss.item()) * inputs.size(0)
                preds = logits.argmax(dim=1)
                running_correct += int((preds == targets).sum().item())
                total_train += inputs.size(0)

            train_loss = running_loss / max(total_train, 1)
            train_acc = running_correct / max(total_train, 1)

            model.eval()
            val_loss_sum = 0.0
            val_correct = 0
            total_val = 0
            with torch.no_grad():
                for inputs, targets in valid_loader:
                    inputs = inputs.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    logits = model(inputs)
                    loss = criterion(logits, targets)
                    val_loss_sum += float(loss.item()) * inputs.size(0)
                    preds = logits.argmax(dim=1)
                    val_correct += int((preds == targets).sum().item())
                    total_val += inputs.size(0)

            val_loss = val_loss_sum / max(total_val, 1)
            val_acc = val_correct / max(total_val, 1)
            current_lr = float(optimizer.param_groups[-1]["lr"])
            scheduler.step()

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["lr"].append(current_lr)

            logger.info(
                "[%s] Epoch %02d/%02d | train_loss=%.4f train_acc=%.4f | val_loss=%.4f val_acc=%.4f",
                self.active_model_name,
                epoch,
                epochs,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
            )

            improved = (val_acc > best_val_acc) or (
                np.isclose(val_acc, best_val_acc) and val_loss < best_val_loss
            )
            if improved:
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logger.info(
                        "Early stopping triggered at epoch %d (best epoch=%d, val_acc=%.4f)",
                        epoch,
                        best_epoch,
                        best_val_acc,
                    )
                    break

        elapsed = time.perf_counter() - t0
        model.load_state_dict(best_state)
        model.cpu()
        model.eval()
        model.class_names = class_names  # type: ignore[attr-defined]

        self.active_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model, self.active_checkpoint)
        self._loaded.pop(self.active_model_name, None)

        self.output_path.mkdir(parents=True, exist_ok=True)
        history_payload = {
            "model_name": self.active_model_name,
            "best_epoch": best_epoch,
            "best_val_acc": round(float(best_val_acc), 6),
            "best_val_loss": round(float(best_val_loss), 6),
            "train_time_sec": round(float(elapsed), 2),
            "num_classes": num_classes,
            "class_names": class_names,
            "hyperparameters": hparams,
            "history": history,
        }
        history_file = self.output_path / f"history_{self.active_model_name}.json"
        history_file.write_text(json.dumps(history_payload, indent=2, ensure_ascii=True), encoding="utf-8")
        logger.info("Saved best checkpoint to %s and history to %s", self.active_checkpoint, history_file)

    def evaluate_classifier(self) -> dict[str, float]:
        """Evalua el modelo activo sobre el conjunto de prueba (`test`)."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        df, class_names = self._load_splits_and_classes()
        num_classes = len(class_names)
        data_dir = self.dataset_path.parent

        test_df = df[df["split"] == "test"].reset_index(drop=True)
        test_records = test_df.to_dict(orient="records")
        test_ds = DogBreedSplitDataset(
            test_records, data_dir, transform=get_eval_transforms(self.image_size)
        )
        num_workers = 0 if os.name == "nt" else 2
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=num_workers)

        model = self.load_model()
        is_onnx = isinstance(model, onnxruntime.InferenceSession)

        all_targets: list[int] = []
        all_preds: list[int] = []
        all_probs: list[np.ndarray] = []

        try:
            if not is_onnx:
                model.to(device)
                model.eval()

            with torch.no_grad():
                for inputs, targets in test_loader:
                    if is_onnx:
                        input_name = model.get_inputs()[0].name
                        logits_np = model.run(None, {input_name: inputs.numpy()})[0]
                        exp_l = np.exp(logits_np - np.max(logits_np, axis=1, keepdims=True))
                        probs_np = exp_l / np.sum(exp_l, axis=1, keepdims=True)
                    else:
                        logits = model(inputs.to(device))
                        probs_np = torch.softmax(logits, dim=1).cpu().numpy()

                    preds_np = np.argmax(probs_np, axis=1)
                    all_targets.extend(int(t) for t in targets.numpy())
                    all_preds.extend(int(p) for p in preds_np)
                    all_probs.append(probs_np)
        finally:
            if not is_onnx:
                model.to("cpu")

        y_true = np.asarray(all_targets, dtype=np.int64)
        y_pred = np.asarray(all_preds, dtype=np.int64)
        probs_mat = np.concatenate(all_probs, axis=0) if all_probs else np.empty((0, num_classes))

        accuracy = float(np.mean(y_true == y_pred)) if len(y_true) > 0 else 0.0

        precisions, recalls, f1s, specificities = [], [], [], []
        for cls_idx in range(num_classes):
            tp = int(np.sum((y_true == cls_idx) & (y_pred == cls_idx)))
            fp = int(np.sum((y_true != cls_idx) & (y_pred == cls_idx)))
            fn = int(np.sum((y_true == cls_idx) & (y_pred != cls_idx)))
            tn = int(np.sum((y_true != cls_idx) & (y_pred != cls_idx)))

            p, r, f1 = precision_recall_f1(tp, fp, fn)
            spec = specificity(tn, fp)
            precisions.append(p)
            recalls.append(r)
            f1s.append(f1)
            specificities.append(spec)

        metrics = {
            "accuracy": round(accuracy, 4),
            "precision": round(float(np.mean(precisions)), 4),
            "recall": round(float(np.mean(recalls)), 4),
            "specificity": round(float(np.mean(specificities)), 4),
            "f1": round(float(np.mean(f1s)), 4),
        }

        # Persistir predicciones detalladas para matrices de confusion y analisis de errores en 2-train.ipynb
        self.output_path.mkdir(parents=True, exist_ok=True)
        eval_file = self.output_path / f"eval_test_{self.active_model_name}.npz"
        np.savez_compressed(
            eval_file,
            y_true=y_true,
            y_pred=y_pred,
            probs=probs_mat,
            rel_paths=test_df["rel_path"].to_numpy(),
            class_names=np.asarray(class_names),
        )
        logger.info("Test metrics (%s): %s", self.active_model_name, metrics)
        return metrics

    def extract_custom_embedding(self, image: np.ndarray) -> list[float]:
        """Extrae el embedding L2-normalizado de 512 dims de la penultima capa del modelo activo."""
        tensor = preprocess_bgr_image(image, image_size=self.image_size)
        batch = tensor.unsqueeze(0)
        model = self.load_model()

        if isinstance(model, onnxruntime.InferenceSession):
            raise ValueError(
                "extract_custom_embedding requiere el checkpoint PyTorch (.pth) para acceder a forward_features."
            )

        model.eval()
        with torch.no_grad():
            if hasattr(model, "forward_features"):
                feats = model.forward_features(batch)
            else:
                feats = model(batch)
            vec = feats.squeeze(0).detach().cpu().numpy().astype(np.float32)

        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec = vec / norm
        return [float(x) for x in vec]