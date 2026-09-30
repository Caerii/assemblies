"""

COLT 2022 MNIST classification — numpy port of learning-with-assemblies/MNIST.ipynb.



Delegates to ``colt_mnist_protocol`` for the faithful notebook algorithm.

"""



from __future__ import annotations



from dataclasses import dataclass



import numpy as np



from neural_assemblies.programs.colt_mnist_data import (

    find_mnist_dir as _find_mnist_dir,

    k_cap as _k_cap,

    load_mnist_arrays as _load_mnist_arrays,

)

from neural_assemblies.programs.colt_mnist_protocol import run_colt_mnist_protocol



# Re-export for backward compatibility

__all__ = [

    "ColtMnistResult",

    "run_colt_mnist_numpy",

    "_find_mnist_dir",

    "_k_cap",

    "_load_mnist_arrays",

]





@dataclass

class ColtMnistResult:

    per_class_accuracy: np.ndarray

    mean_accuracy: float

    data_source: str

    parameters: dict





def run_colt_mnist_numpy(

    *,

    seed: int = 42,

    n_in: int = 784,

    n_neurons: int = 2000,

    cap_size: int = 200,

    sparsity: float = 0.1,

    beta: float = 1.0,

    n_rounds: int = 5,

    n_examples: int = 500,

) -> ColtMnistResult:

    """Train + classify per COLT MNIST notebook protocol."""

    result = run_colt_mnist_protocol(

        seed=seed,

        n_in=n_in,

        n_neurons=n_neurons,

        cap_size=cap_size,

        sparsity=sparsity,

        beta=beta,

        n_rounds=n_rounds,

        n_examples=n_examples,

    )

    return ColtMnistResult(

        per_class_accuracy=result.per_class_accuracy,

        mean_accuracy=result.mean_accuracy,

        data_source=result.data_source,

        parameters=result.parameters,

    )


