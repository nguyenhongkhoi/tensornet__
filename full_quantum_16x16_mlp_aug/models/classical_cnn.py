from __future__ import annotations

from typing import Dict

import torch
from torch import nn


class ClassicalCNN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 10, parameter_matched: bool = False):
        super().__init__()
        widths = (8, 16) if parameter_matched else (32, 64)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, widths[0], 3, padding=1),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(widths[0], widths[1], 3, padding=1),
            nn.BatchNorm2d(widths[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(widths[1] * 4 * 4, widths[1]),
            nn.ReLU(inplace=True),
            nn.Linear(widths[1], num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"classical_total": total, "total": total}
