"""Shared image preprocessing for TripoSR inference.

TripoSR expects RGB images with:
  - Gray (0.5) background
  - Foreground centered and occupying ~85% of the frame
  - 512x512 resolution (handled by the model's ImagePreprocessor)

This module provides the standard preprocessing pipeline that matches
the official run.py behavior.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from tsr.utils import resize_foreground


def prepare_image(
    image_path: Path | str,
    foreground_ratio: float = 0.85,
) -> Image.Image:
    """Load and preprocess an image for TripoSR inference.

    Handles both RGBA (bg-removed) and RGB (with background) images.
    For RGBA: composites foreground onto gray (0.5) background.
    For RGB with --no-remove-bg: returns as-is (assumes pre-processed).
    """
    image = Image.open(image_path)

    if image.mode == "RGBA":
        image = resize_foreground(image, foreground_ratio)
        arr = np.array(image).astype(np.float32) / 255.0
        rgb = arr[:, :, :3] * arr[:, :, 3:4] + (1 - arr[:, :, 3:4]) * 0.5
        return Image.fromarray((rgb * 255.0).astype(np.uint8))
    else:
        return image.convert("RGB")
