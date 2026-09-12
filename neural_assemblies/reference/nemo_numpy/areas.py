"""
Numpy area models from mdabagia/nemo ``brain.py``.

Faithful FF / recurrent / scaffold dynamics for literature parity when the
Brain assembly-calculus engine differs (e.g. non-LRI sequence recall).
"""

from __future__ import annotations

import numpy as np


def k_cap(input_arr: np.ndarray, cap_size: int) -> np.ndarray:
    """Top-cap_size indices by total input (reference ``brain.k_cap``)."""
    if np.all(input_arr == 0):
        return np.array([], dtype=int)
    return input_arr.argsort(axis=-1)[..., -cap_size:]


def positional_overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Positional winner overlap (nemo-demo ``(ref == read).sum() / k``)."""
    if len(a) == 0 or len(b) == 0:
        return 0.0
    if len(a) == len(b):
        return float((a == b).sum()) / len(b)
    return len(np.intersect1d(a, b)) / len(b)


class FFArea:
    """Feed-forward area (reference ``brain.FFArea``)."""

    def __init__(
        self,
        n_inputs,
        n_neurons: int,
        cap_size: int,
        density: float,
        plasticity: float,
        rng: np.random.Generator,
        norm_init: bool = False,
    ):
        if isinstance(n_inputs, int):
            self.n_input_areas = 1
            n_inputs = [n_inputs]
        else:
            self.n_input_areas = len(n_inputs)
        self.n_inputs = list(n_inputs)
        self.n_neurons = n_neurons
        self.cap_size = cap_size
        self.density = density
        self.plasticity = plasticity
        self.rng = rng
        self.norm_init = norm_init
        self.reset()

    def reset(self) -> None:
        self.input_weights = [
            (self.rng.random((n, self.n_neurons)) < self.density).astype(float)
            for n in self.n_inputs
        ]
        self.inhibit()
        if self.norm_init:
            self.normalize()

    def inhibit(self) -> None:
        self.inputs = [[] for _ in self.n_inputs]
        self.activations: np.ndarray = np.array([], dtype=int)

    def set_input(self, inputs, input_area: int = 0) -> None:
        if isinstance(input_area, int):
            if len(inputs) == self.n_input_areas:
                self.inputs = list(inputs)
            else:
                self.inputs[input_area] = inputs
        else:
            for i in input_area:
                self.inputs[i] = inputs[i]

    def clear_input(self, input_area: int = -1) -> None:
        if isinstance(input_area, int):
            if input_area < 0:
                self.inputs = [[] for _ in self.n_inputs]
            else:
                self.inputs[input_area] = []

    def get_total_input(self) -> np.ndarray:
        return sum(
            w[inp].sum(axis=0) if len(inp) else np.zeros(self.n_neurons)
            for w, inp in zip(self.input_weights, self.inputs, strict=True)
        )

    def step(self, update: bool = True) -> None:
        new_activations = k_cap(self.get_total_input(), self.cap_size)
        if update:
            self.update(new_activations)
        self.activations = new_activations
        self.clear_input()

    def fire(self, activations: np.ndarray, update: bool = True) -> None:
        """Set winners directly (reference ``brain.FFArea.fire``)."""
        if update:
            self.update(activations)
        self.activations = activations
        self.clear_input()

    def forward(self, inputs, input_area: int = 0, update: bool = True) -> None:
        self.set_input(inputs, input_area=input_area)
        self.step(update=update)

    def update(self, new_activations: np.ndarray) -> None:
        for w, inp in zip(self.input_weights, self.inputs, strict=True):
            if len(inp):
                w[np.ix_(inp, new_activations)] *= 1 + self.plasticity

    def normalize(self) -> None:
        for w in self.input_weights:
            w /= np.maximum(w.sum(axis=0, keepdims=True), 1e-12)

    def read(self) -> np.ndarray:
        return self.activations


class RecurrentArea(FFArea):
    """Recurrent area (reference ``brain.RecurrentArea``)."""

    def reset(self) -> None:
        self.recurrent_weights = (
            self.rng.random((self.n_neurons, self.n_neurons)) < self.density
        ).astype(float)
        super().reset()

    def get_total_input(self) -> np.ndarray:
        ff = super().get_total_input()
        if len(self.activations):
            ff = ff + self.recurrent_weights[self.activations].sum(axis=0)
        return ff

    def update(self, new_activations: np.ndarray) -> None:
        super().update(new_activations)
        if len(self.activations):
            self.recurrent_weights[np.ix_(self.activations, new_activations)] *= (
                1 + self.plasticity
            )

    def normalize(self) -> None:
        super().normalize()
        self.recurrent_weights /= np.maximum(
            self.recurrent_weights.sum(axis=0, keepdims=True), 1e-12,
        )


class RefractedArea(FFArea):
    """Refracted arc area (reference ``brain.RefractedArea``)."""

    def reset(self) -> None:
        self.bias = np.zeros(self.n_neurons)
        super().reset()

    def get_total_input(self) -> np.ndarray:
        return super().get_total_input() - self.bias

    def update(self, new_activations: np.ndarray) -> None:
        self.bias[new_activations] += super().get_total_input()[new_activations] * self.plasticity
        super().update(new_activations)


class ScaffoldNetwork:
    """Bidirectional main + auxiliary recurrent areas."""

    def __init__(
        self,
        n_inputs: int,
        n_neurons: int,
        cap_size: int,
        density: float,
        plasticity: float,
        rng: np.random.Generator,
    ):
        self.areas = [
            RecurrentArea(
                [n_inputs, n_neurons], n_neurons, cap_size, density, plasticity, rng,
            ),
            RecurrentArea(n_neurons, n_neurons, cap_size, density, plasticity, rng),
        ]

    def inhibit(self) -> None:
        for area in self.areas:
            area.inhibit()

    def forward(self, inputs, update: bool = True) -> None:
        self.areas[0].set_input(inputs)
        self.step(update=update)

    def step(self, update: bool = True) -> None:
        self.areas[0].set_input(self.areas[1].read(), input_area=1)
        self.areas[1].set_input(self.areas[0].read())
        for area in self.areas:
            area.step(update=update)

    def read(self) -> np.ndarray:
        return self.areas[0].read()
