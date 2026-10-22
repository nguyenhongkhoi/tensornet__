from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


def rgb_to_grayscale(images: torch.Tensor) -> torch.Tensor:
    if images.shape[1] == 1:
        return images
    weights = images.new_tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1)
    return (images * weights).sum(dim=1, keepdim=True)


def downsample_images(images: torch.Tensor, size: Sequence[int]) -> torch.Tensor:
    return F.interpolate(images, size=tuple(size), mode="bilinear", align_corners=False)


def flatten_quantum_sites(images: torch.Tensor) -> torch.Tensor:
    return images.flatten(start_dim=1).clamp(0.0, 1.0)


def preprocess_full_quantum(
    images: torch.Tensor,
    target_size: Sequence[int] = (8, 8),
    grayscale: bool = False,
    mode: str = "resize",
) -> torch.Tensor:
    if grayscale:
        images = rgb_to_grayscale(images)
    target_height, target_width = tuple(target_size)
    if mode == "resize":
        images = downsample_images(images, target_size)
    elif mode == "pad":
        height, width = images.shape[-2:]
        if target_height < height or target_width < width:
            raise ValueError(f"pad mode cannot shrink {height}x{width} to {target_height}x{target_width}")
        pad_height, pad_width = target_height - height, target_width - width
        top, left = pad_height // 2, pad_width // 2
        images = F.pad(images, (left, pad_width - left, top, pad_height - top))
    else:
        raise ValueError(f"Unknown full-quantum preprocessing mode: {mode!r}")
    return flatten_quantum_sites(images)
