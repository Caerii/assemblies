"""MNIST data loading and preprocessing helpers for COLT 2022 protocols."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


class DatasetUnavailable(RuntimeError):
    """A requested real dataset is absent; a synthetic substitute is not evidence."""


def require_mnist_dir() -> Path:
    directory = find_mnist_dir()
    if directory is None:
        raise DatasetUnavailable(
            'MNIST golden requires mnist_train.csv and mnist_test.csv in '
            'data/mnist (or the reference data directory). Real data is absent; '
            'synthetic fallback cannot reproduce this golden.')
    return directory


def k_cap(input_arr: np.ndarray, cap_size: int) -> np.ndarray:
    output = np.zeros_like(input_arr)
    if input_arr.ndim == 1:
        idx = np.argsort(input_arr)[-cap_size:]
        output[idx] = 1
    else:
        idx = np.argsort(input_arr, axis=-1)[:, -cap_size:]
        np.put_along_axis(output, idx, 1, axis=-1)
    return output


def find_mnist_dir() -> Path | None:
    root = Path(__file__).resolve().parents[2]
    candidates = [
        root / "data" / "mnist",
        root / ".reference" / "mdabagia-learning-with-assemblies" / "data" / "mnist",
    ]
    for d in candidates:
        train = d / "mnist_train.csv"
        test = d / "mnist_test.csv"
        if train.is_file() and test.is_file():
            return d
    return None


def synthetic_mnist_arrays(
    n_examples: int,
    *,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Separable random binary patterns (Stimulus-Classes style)."""
    rng = np.random.default_rng(seed)
    n_in = 784
    cap_size = 200
    train_imgs = np.zeros((10 * n_examples, n_in))
    train_labels = np.repeat(np.arange(10), n_examples)
    test_imgs = np.zeros((10 * n_examples, n_in))
    test_labels = np.repeat(np.arange(10), n_examples)

    for digit in range(10):
        hot = rng.choice(n_in, size=cap_size, replace=False)
        sl = slice(digit * n_examples, (digit + 1) * n_examples)
        train_imgs[sl][:, hot] = 1.0
        test_imgs[sl][:, hot] = 1.0
        train_imgs[sl] += (rng.random((n_examples, n_in)) < 0.02)
        test_imgs[sl] += (rng.random((n_examples, n_in)) < 0.02)

    return train_imgs, train_labels, test_imgs, test_labels


def load_mnist_arrays(
    n_examples: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mnist_dir = find_mnist_dir()
    if mnist_dir is None:
        return synthetic_mnist_arrays(n_examples)

    train = np.loadtxt(mnist_dir / "mnist_train.csv", delimiter=",")
    test = np.loadtxt(mnist_dir / "mnist_test.csv", delimiter=",")
    train_labels = train[:, 0].astype(int)
    test_labels = test[:, 0].astype(int)
    train_imgs = train[:, 1:]
    test_imgs = test[:, 1:]
    return train_imgs, train_labels, test_imgs, test_labels
