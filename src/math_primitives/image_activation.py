# image_activation.py

"""
Image activation utilities extracted from brain.py's activate_with_image.
"""

import numpy as np
import torch
from typing import Tuple


class ImageActivationEngine:
    def __init__(self):
        pass

    def preprocess_image(self, image: np.ndarray, target_n: int) -> np.ndarray:
        """
        Flatten, crop or pad the image to match target_n.
        """
        image_flat = image.flatten()
        size = image_flat.size
        if size > target_n:
            image_flat = image_flat[:target_n]
        elif size < target_n:
            padding = np.zeros(target_n - size, dtype=image_flat.dtype)
            image_flat = np.concatenate((image_flat, padding))
        return image_flat

    def normalize_and_select_topk(self, vec: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalize vec with min-max and L2, then return (winners, normalized_vec).
        """
        norm_vec = (vec - vec.min()) / (np.ptp(vec) + 1e-6)
        norm_vec = norm_vec / (np.linalg.norm(norm_vec) + 1e-6)
        winners = np.argsort(-norm_vec)[:k]
        return winners.astype(np.uint32), norm_vec


# Standalone functions for direct use in brain.py
def preprocess_image(image, target_n):
    """Flatten, crop or pad the image to match target_n."""
    if isinstance(image, np.ndarray):
        image_flat = image.flatten()
        image_size = image_flat.size
    elif isinstance(image, torch.Tensor):
        image_flat = image.flatten().cpu().numpy()
        image_size = image_flat.size
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")
    
    if image_size > target_n:
        image_flat = image_flat[:target_n]
    elif image_size < target_n:
        padding = np.zeros(target_n - image_size, dtype=image_flat.dtype)
        image_flat = np.concatenate((image_flat, padding))
    
    return image_flat


def normalize_and_select_topk(image_flat, k, n):
    """Normalize and select top-k winners."""
    # Normalize the image data to [0, 1]
    normalized_image = (image_flat - image_flat.min()) / (np.ptp(image_flat) + 1e-6)
    normalized_image = normalized_image / (np.linalg.norm(normalized_image) + 1e-6)

    # Select the top-k pixels with the highest values
    top_k_indices = np.argsort(-normalized_image)[:k]

    # Set the winners in the area
    return np.array(top_k_indices[top_k_indices < n], dtype=np.uint32)


