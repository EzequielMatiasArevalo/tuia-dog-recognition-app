"""Arquitecturas de clasificacion (Modelo A: ResNet18 fine-tuned, Modelo B: CNN propia).

Definidas en un modulo importable de `src/lib/models/` para que `torch.save(model, path)`
y `torch.load(path, weights_only=False)` funcionen tanto en notebooks como en el backend.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


class ResNet18BreedClassifier(nn.Module):
    """Modelo A: ResNet18 pre-entrenada en ImageNet con cabeza adaptada a 70 razas."""

    def __init__(
        self,
        num_classes: int = 70,
        embedding_dim: int = 512,
        dropout: float = 0.3,
        pretrained: bool = True,
        class_names: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.class_names: list[str] = list(class_names) if class_names else []

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        in_features = backbone.fc.in_features  # 512
        backbone.fc = nn.Identity()
        self.backbone = backbone

        if in_features == embedding_dim:
            self.embedding_proj: nn.Module = nn.Identity()
        else:
            self.embedding_proj = nn.Sequential(
                nn.Linear(in_features, embedding_dim, bias=False),
                nn.BatchNorm1d(embedding_dim),
                nn.SiLU(inplace=True),
            )

        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(embedding_dim, num_classes),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Devuelve el vector de caracteristicas de dimension `embedding_dim` (512)."""
        feats = self.backbone(x)
        return self.embedding_proj(feats)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.forward_features(x)
        return self.head(emb)


class ConvBlock(nn.Module):
    """Bloque convolucional doble con BatchNorm y SiLU + MaxPool2d."""

    def __init__(self, in_channels: int, out_channels: int, pool: bool = True) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CustomCNN(nn.Module):
    """Modelo B: CNN disenada y entrenada desde cero con capa de embedding de 512 dims.

    Entrada: (B, 3, 224, 224)
    Etapas convolucionales:
      - Stage 1: 3   -> 32  (224 -> 112)
      - Stage 2: 32  -> 64  (112 -> 56)
      - Stage 3: 64  -> 128 (56  -> 28)
      - Stage 4: 128 -> 256 (28  -> 14)
      - Stage 5: 256 -> 512 (14  -> 7)
    Global Average Pooling -> Linear(512, 512) + BN + SiLU (embedding) -> Dropout -> Linear(512, 70)
    """

    def __init__(
        self,
        num_classes: int = 70,
        embedding_dim: int = 512,
        dropout: float = 0.4,
        class_names: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.class_names: list[str] = list(class_names) if class_names else []

        self.features = nn.Sequential(
            ConvBlock(3, 32, pool=True),
            ConvBlock(32, 64, pool=True),
            ConvBlock(64, 128, pool=True),
            ConvBlock(128, 256, pool=True),
            ConvBlock(256, 512, pool=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim),
            nn.SiLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(embedding_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Devuelve el embedding de dimension `embedding_dim` (512)."""
        x = self.features(x)
        x = self.pool(x)
        return self.embedding_layer(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.forward_features(x)
        return self.classifier(emb)