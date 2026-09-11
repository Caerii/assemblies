# image_activation.py

"""
Image activation utilities extracted from brain.py's activate_with_image.
"""

import sys
import numpy as np

try:
    from ..core.backend import get_xp, to_xp
except ImportError:
    from core.backend import get_xp, to_xp


def _torch_tensor_type():
    """``torch.Tensor`` if torch is ALREADY imported, else ``None``.

    Torch is only needed here to recognise a tensor argument, but importing it
    eagerly costs ~11s and this module is on the ``import neural_assemblies``
    path (via ``core.brain``), so every process paid that.

    Checking ``sys.modules`` instead is EQUIVALENT, not an approximation: a
    ``torch.Tensor`` instance cannot exist unless torch has already been
    imported by whoever constructed it. If torch is absent from ``sys.modules``
    the isinstance check could not have matched anyway.
    """
    mod = sys.modules.get("torch")
    return getattr(mod, "Tensor", None) if mod is not None else None


def __getattr__(name):
    # Preserve the previous module-level ``torch`` / ``HAS_TORCH`` names
    # without forcing the import (PEP 562).
    if name == "HAS_TORCH":
        return _torch_tensor_type() is not None
    if name == "torch":
        return sys.modules.get("torch")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
    elif (_TensorT := _torch_tensor_type()) is not None and isinstance(
        image, _TensorT,
    ):
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
