# image_activation.py

"""
Image activation utilities extracted from brain.py's activate_with_image.
"""

import numpy as np
from typing import Tuple

try:
    from ..core.backend import get_xp, to_xp, to_cpu
except ImportError:
    from core.backend import get_xp, to_xp, to_cpu

# Optional torch import for tensor support
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False


class ImageActivationEngine:
    def __init__(self):
        pass

    def preprocess_image(self, image, target_n: int):
        """
        Flatten, crop or pad the image to match target_n.
        """
        xp = get_xp()
        image_flat = xp.asarray(image).flatten()
        size = image_flat.size
        if size > target_n:
            image_flat = image_flat[:target_n]
        elif size < target_n:
            padding = xp.zeros(target_n - size, dtype=image_flat.dtype)
            image_flat = xp.concatenate((image_flat, padding))
        return image_flat

    def normalize_and_select_topk(self, vec, k: int):
        """
        Normalize vec with min-max and L2, then return (winners, normalized_vec).
        """
        xp = get_xp()
        vec = xp.asarray(vec)
        ptp = float(vec.max() - vec.min())
        norm_vec = (vec - vec.min()) / (ptp + 1e-6)
        norm_vec = norm_vec / (float(xp.linalg.norm(norm_vec)) + 1e-6)
        winners = xp.argsort(-norm_vec)[:k]
        return winners.astype(xp.uint32), norm_vec


# Standalone functions for direct use in brain.py
def preprocess_image(image, target_n):
    """Flatten, crop or pad the image to match target_n."""
    if isinstance(image, np.ndarray):
        image_flat = image.flatten()
        image_size = image_flat.size
    elif HAS_TORCH and isinstance(image, torch.Tensor):
        image_flat = image.flatten().cpu().numpy()
        image_size = image_flat.size
    else:
        # Try to convert to numpy array
        try:
            image_flat = np.asarray(image).flatten()
            image_size = image_flat.size
        except Exception:
            raise TypeError(f"Unsupported image type: {type(image)}")

    if image_size > target_n:
        image_flat = image_flat[:target_n]
    elif image_size < target_n:
        padding = np.zeros(target_n - image_size, dtype=image_flat.dtype)
        image_flat = np.concatenate((image_flat, padding))

    return to_xp(image_flat)


def normalize_and_select_topk(image_flat, k, n):
    """Normalize and select top-k winners."""
    xp = get_xp()
    image_flat = xp.asarray(image_flat)
    # Normalize the image data to [0, 1]
    ptp = float(image_flat.max() - image_flat.min())
    normalized_image = (image_flat - image_flat.min()) / (ptp + 1e-6)
    normalized_image = normalized_image / (float(xp.linalg.norm(normalized_image)) + 1e-6)

    # Select the top-k pixels with the highest values
    top_k_indices = xp.argsort(-normalized_image)[:k]

    # Set the winners in the area
    valid = top_k_indices[top_k_indices < n]
    return valid.astype(xp.uint32)
