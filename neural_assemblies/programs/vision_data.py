"""
Unified vision dataset loading — MNIST, Fashion-MNIST, synthetic fallback.

Fashion-MNIST uses the same 28×28 grayscale layout as MNIST so patch graphs
and spatial connectomes transfer without reshape changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Tuple

import numpy as np

from neural_assemblies.programs.colt_mnist_data import (
    find_mnist_dir,
    load_mnist_arrays,
    synthetic_mnist_arrays,
)
from neural_assemblies.programs.colt_mnist_protocol import preprocess_mnist_examples


class VisionDataset(str, Enum):
    MNIST = "mnist"
    FASHION_MNIST = "fashion_mnist"


FASHION_MNIST_CLASS_NAMES: tuple[str, ...] = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)

# Apparel confusions analogous to MNIST stroke pairs.
FASHION_CONFUSED_PAIRS: tuple[tuple[int, int], ...] = (
    (0, 2), (0, 6), (2, 4), (4, 6), (5, 7), (7, 9), (1, 3),
)

FASHION_CONFUSED_LABELS: frozenset[int] = frozenset(
    d for pair in FASHION_CONFUSED_PAIRS for d in pair
)


@dataclass(frozen=True)
class VisionDataInfo:
    dataset: VisionDataset
    n_classes: int
    class_names: tuple[str, ...]
    confused_pairs: tuple[tuple[int, int], ...]
    data_source: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def find_fashion_mnist_dir() -> Path | None:
    root = _repo_root()
    candidates = [
        root / "data" / "fashion_mnist",
        root / ".reference" / "mdabagia-learning-with-assemblies" / "data" / "fashion_mnist",
    ]
    for d in candidates:
        train = d / "fashion_mnist_train.csv"
        test = d / "fashion_mnist_test.csv"
        if train.is_file() and test.is_file():
            return d
    return None


def synthetic_fashion_mnist_arrays(
    n_examples: int,
    *,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Separable random patterns (same protocol as MNIST synthetic)."""
    return synthetic_mnist_arrays(n_examples, seed=seed + 17)


def load_fashion_mnist_arrays(
    n_examples: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ensure_fashion_mnist_csv()
    fdir = find_fashion_mnist_dir()
    if fdir is None:
        return synthetic_fashion_mnist_arrays(n_examples)

    train = np.loadtxt(fdir / "fashion_mnist_train.csv", delimiter=",")
    test = np.loadtxt(fdir / "fashion_mnist_test.csv", delimiter=",")
    train_labels = train[:, 0].astype(int)
    test_labels = test[:, 0].astype(int)
    train_imgs = train[:, 1:]
    test_imgs = test[:, 1:]
    return train_imgs, train_labels, test_imgs, test_labels


def _read_idx_images(path: Path) -> np.ndarray:
    import gzip
    import struct

    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows * cols).astype(np.float64)


def _read_idx_labels(path: Path) -> np.ndarray:
    import gzip
    import struct

    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8).astype(int)


def ensure_fashion_mnist_csv(*, force: bool = False) -> Path | None:
    """
    Download Fashion-MNIST IDX files and write CSV mirrors of MNIST layout.

    Returns data directory when CSV exists or was created; None on network failure.
    """
    import urllib.request

    out_dir = _repo_root() / "data" / "fashion_mnist"
    train_csv = out_dir / "fashion_mnist_train.csv"
    test_csv = out_dir / "fashion_mnist_test.csv"
    if not force and train_csv.is_file() and test_csv.is_file():
        return out_dir

    base = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion"
    pairs = (
        ("train-images-idx3-ubyte.gz", "train-images-idx3-ubyte.gz"),
        ("train-labels-idx1-ubyte.gz", "train-labels-idx1-ubyte.gz"),
        ("t10k-images-idx3-ubyte.gz", "t10k-images-idx3-ubyte.gz"),
        ("t10k-labels-idx1-ubyte.gz", "t10k-labels-idx1-ubyte.gz"),
    )
    tmp = out_dir / "_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        for name, _ in pairs:
            dest = tmp / name
            if force or not dest.is_file():
                urllib.request.urlretrieve(f"{base}/{name}", dest)

        train_x = _read_idx_images(tmp / "train-images-idx3-ubyte.gz")
        train_y = _read_idx_labels(tmp / "train-labels-idx1-ubyte.gz")
        test_x = _read_idx_images(tmp / "t10k-images-idx3-ubyte.gz")
        test_y = _read_idx_labels(tmp / "t10k-labels-idx1-ubyte.gz")

        np.savetxt(train_csv, np.column_stack([train_y, train_x]), delimiter=",", fmt="%.0f")
        np.savetxt(test_csv, np.column_stack([test_y, test_x]), delimiter=",", fmt="%.0f")
        return out_dir
    except (OSError, Exception):
        return None


def load_vision_arrays(
    dataset: VisionDataset | str,
    n_examples: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ds = VisionDataset(dataset)
    if ds == VisionDataset.MNIST:
        return load_mnist_arrays(n_examples)
    return load_fashion_mnist_arrays(n_examples)


def preprocess_vision_examples(
    train_imgs: np.ndarray,
    train_labels: np.ndarray,
    *,
    n_examples: int,
    cap_size: int = 200,
) -> np.ndarray:
    return preprocess_mnist_examples(
        train_imgs, train_labels, n_examples=n_examples, cap_size=cap_size,
    )


def vision_data_info(dataset: VisionDataset | str) -> VisionDataInfo:
    ds = VisionDataset(dataset)
    if ds == VisionDataset.MNIST:
        from neural_assemblies.programs.colt_mnist_tier_util import CONFUSED_PAIRS

        return VisionDataInfo(
            dataset=ds,
            n_classes=10,
            class_names=tuple(str(i) for i in range(10)),
            confused_pairs=CONFUSED_PAIRS,
            data_source="mnist_csv" if find_mnist_dir() else "synthetic_fallback",
        )
    return VisionDataInfo(
        dataset=ds,
        n_classes=10,
        class_names=FASHION_MNIST_CLASS_NAMES,
        confused_pairs=FASHION_CONFUSED_PAIRS,
        data_source="fashion_mnist_csv" if find_fashion_mnist_dir() else "synthetic_fallback",
    )


def load_vision_examples(
    dataset: VisionDataset | str,
    *,
    n_examples: int = 50,
    cap_size: int = 200,
) -> tuple[np.ndarray, VisionDataInfo]:
    """Return ``(examples[digit, j], info)`` shaped ``(10, n_examples, 784)``."""
    train_imgs, train_labels, _, _ = load_vision_arrays(dataset, n_examples)
    examples = preprocess_vision_examples(
        train_imgs, train_labels, n_examples=n_examples, cap_size=cap_size,
    )
    return examples, vision_data_info(dataset)
