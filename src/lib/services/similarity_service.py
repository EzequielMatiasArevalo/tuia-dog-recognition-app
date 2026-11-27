from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional
from uuid import uuid4

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

from lib.preprocessing import preprocess_bgr_image
from lib.schemas import EmbeddingRecord, Neighbor, SearchResult
from lib.storage.base import EmbeddingStoreProtocol

logger = logging.getLogger(__name__)


class SimilarityService:
    """Etapa 1: buscador de imagenes por similitud.

    Funciones implementadas:
      - extract_embedding(image)
      - search_similar_images(embedding, top_k)
      - predict_breed_from_neighbors(results)
    """

    def __init__(
        self,
        store: EmbeddingStoreProtocol,
        similarity_metric: str,
        similarity_threshold: float,
        top_k: int,
        image_size: int,
        model_name: str,
        url_resolver: Optional[Callable[[Path], Optional[str]]] = None,
        data_root: Optional[Path] = None,
    ) -> None:
        self.store = store
        self.similarity_metric = similarity_metric
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        self.image_size = image_size
        self.model_name = model_name
        self.url_resolver = url_resolver
        self.data_root = data_root
        self._backbone: Optional[nn.Module] = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_image(self, source_path: str) -> np.ndarray:
        image = cv2.imread(str(source_path))
        if image is None:
            raise ValueError(f"Could not read image: {source_path}")
        return image

    def _get_backbone(self) -> nn.Module:
        """Carga y cachea ResNet18 pre-entrenada en ImageNet sin la capa fc (512 dims)."""
        if self._backbone is None:
            logger.info("Loading baseline feature extractor (ResNet18 ImageNet) on %s", self._device)
            backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT)
            backbone.fc = nn.Identity()
            backbone.eval()
            backbone.to(self._device)
            self._backbone = backbone
        return self._backbone

    # ------------------------------------------------------------------
    # Etapa 1: funciones implementadas
    # ------------------------------------------------------------------

    def extract_embedding(self, image: np.ndarray) -> list[float]:
        """Genera un embedding L2-normalizado de 512 dimensiones usando ResNet18 ImageNet."""
        tensor = preprocess_bgr_image(image, image_size=self.image_size)
        batch = tensor.unsqueeze(0).to(self._device)
        backbone = self._get_backbone()
        with torch.no_grad():
            features = backbone(batch).squeeze(0).detach().cpu().numpy().astype(np.float32)

        norm = float(np.linalg.norm(features))
        if norm > 0.0:
            features = features / norm
        return [float(x) for x in features]

    def search_similar_images(self, embedding: list[float], top_k: int) -> list[Neighbor]:
        """Recupera de la base vectorial las top_k imagenes mas similares ordenadas por score."""
        k = max(1, int(top_k))
        metric = self.similarity_metric.lower()

        # Si el store soporta busqueda nativa por distancia coseno (pgvector <=>), delegar el top-k
        search_fn = getattr(self.store, "search", None)
        if metric == "cosine" and callable(search_fn):
            records = search_fn(embedding, k)
            scored = [
                Neighbor(
                    path=rec.path,
                    breed=rec.breed,
                    score=round(float(self.similarity(embedding, rec.embedding)), 6),
                )
                for rec in records
            ]
            scored.sort(key=lambda n: n.score, reverse=True)
            return scored[:k]

        records = self.store.all()
        if not records:
            return []

        scored_all = [
            Neighbor(
                path=rec.path,
                breed=rec.breed,
                score=round(float(self.similarity(embedding, rec.embedding)), 6),
            )
            for rec in records
        ]
        scored_all.sort(key=lambda n: n.score, reverse=True)
        return scored_all[:k]

    def predict_breed_from_neighbors(self, results: list[Neighbor]) -> tuple[str, float]:
        """Predice la raza mediante voto mayoritario ponderado por similitud."""
        if not results:
            return ("unknown", 0.0)

        best_neighbor_score = max(float(r.score) for r in results)
        if best_neighbor_score < self.similarity_threshold:
            return ("unknown", best_neighbor_score)

        breed_weights: dict[str, float] = defaultdict(float)
        breed_scores: dict[str, list[float]] = defaultdict(list)

        for neighbor in results:
            score = float(neighbor.score)
            breed_weights[neighbor.breed] += max(0.0, score)
            breed_scores[neighbor.breed].append(score)

        winning_breed = max(
            breed_weights.keys(),
            key=lambda b: (breed_weights[b], float(np.mean(breed_scores[b]))),
        )
        mean_breed_score = float(np.mean(breed_scores[winning_breed]))

        if mean_breed_score < self.similarity_threshold:
            return ("unknown", mean_breed_score)
        return (winning_breed, mean_breed_score)

    # ------------------------------------------------------------------
    # Helpers de similitud provistos
    # ------------------------------------------------------------------

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _l2_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        dist = float(np.linalg.norm(a - b))
        return 1.0 / (1.0 + dist)

    def similarity(self, query: list[float], ref: list[float]) -> float:
        a = np.asarray(query, dtype=np.float32)
        b = np.asarray(ref, dtype=np.float32)
        if self.similarity_metric.lower() == "l2":
            return self._l2_similarity(a, b)
        return self._cosine(a, b)

    # ------------------------------------------------------------------
    # Orquestacion provista
    # ------------------------------------------------------------------

    def _to_stored_path(self, image_path: str) -> str:
        if self.data_root is None:
            return str(image_path)
        try:
            return Path(image_path).resolve().relative_to(self.data_root.resolve()).as_posix()
        except ValueError:
            return str(image_path)

    def resolve_path(self, stored_path: str) -> Path:
        path = Path(stored_path)
        if self.data_root is None or path.is_absolute():
            return path
        candidate = self.data_root / path
        if not candidate.exists() and path.exists():
            return path
        return candidate

    def index_image(
        self, image_path: str, breed: str, metadata: dict[str, object] | None = None
    ) -> EmbeddingRecord:
        image = self._load_image(image_path)
        embedding = self.extract_embedding(image)
        record = EmbeddingRecord(
            id_imagen=str(uuid4()),
            embedding=embedding,
            path=self._to_stored_path(image_path),
            breed=breed,
            metadata=metadata or {},
        )
        self.store.append(record)
        return record

    def _with_url(self, neighbor: Neighbor) -> Neighbor:
        if self.url_resolver is not None and not neighbor.url:
            neighbor.url = self.url_resolver(self.resolve_path(neighbor.path))
        return neighbor

    def search(
        self,
        source_path: str,
        output_path: Path,
        embedding_fn: Optional[Callable[[np.ndarray], list[float]]] = None,
        model_name: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> str:
        image = self._load_image(source_path)
        extractor = embedding_fn or self.extract_embedding
        embedding = extractor(image)

        k = int(top_k) if top_k else self.top_k
        neighbors = [self._with_url(n) for n in self.search_similar_images(embedding, k)]
        breed, score = self.predict_breed_from_neighbors(neighbors)
        logger.info("Predicted breed: %s (score=%.4f) for %s", breed, score, source_path)

        payload = SearchResult(
            source_path=source_path,
            model=model_name or self.model_name,
            predicted_breed=breed,
            score=round(float(score), 4),
            neighbors=neighbors,
        )
        output_path.mkdir(parents=True, exist_ok=True)
        result_file = output_path / f"result-{uuid4()}.json"
        result_file.write_text(
            json.dumps(payload.model_dump(), ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        return str(result_file)