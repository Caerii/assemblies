"""Prediction-error metrics — N400 analogue kernels.

The N400 is a negative-going ERP component whose amplitude scales with how
UNEXPECTED a word is in its context -- larger for a word the preceding
context did not predict, smaller for one it did.  Under a predictive-coding
reading, it is prediction error.

NEMO makes that error directly measurable.  A context assembly is a literal
prediction: it is the set of neurons the accumulated context drives, before
the actual word arrives.  The stored lexicon assembly for the word that does
arrive is the target.  How far apart they are IS the prediction error, so no
regression or fitting step is needed to get from the model to the component.

Contrast with the P600 kernels in ``instability``: N400 compares a prediction
to a target and is a ONE-SHOT quantity; P600 measures how far the
representation has to travel to settle and is a quantity OVER TIME.  They
answer different questions and are deliberately not the same measure.

Pure functions on assembly winner arrays. No parser or Brain state.
"""

from __future__ import annotations

from typing import Union

import numpy as np

from ..assembly import overlap
from ...core.index_spaces import NeuronIds


def measure_n400(
    predicted: NeuronIds,
    lexicon_entry: NeuronIds,
) -> float:
    """N400 = 1 - overlap(context prediction, lexicon entry).

    BOTH OPERANDS ARE NEURON IDS. A stored lexicon entry is in neuron-ID space
    by construction (it outlives the projection that made it), so the live
    prediction must be read through ``ops._snap`` / ``diagnostics.read_assembly``
    and not off ``area.winners``. Passing compact indices here compares
    unrelated integer sets and returns a plausible N400 -- see
    ``core/index_spaces``.

    Returns value in [0, 1]: 0 = perfectly predicted, 1 = fully unexpected.

    Uses the min-normalised :func:`~..assembly.overlap`, which is the right
    choice here (unlike in the P600 kernels): a context assembly that contains
    the whole target word assembly plus extra neurons has predicted the word,
    and should score 0.  Union normalisation would penalise it for the extra
    activity, which is not what "was this word predicted" asks.

    The floor is not 0 in practice.  Two unrelated assemblies still share
    ``k/n`` neurons by chance (:func:`~..assembly.chance_overlap`), so a fully
    unexpected word scores about ``1 - k/n``, not 1.0.  Report differences
    between conditions rather than absolute values.
    """
    return 1.0 - overlap(predicted, lexicon_entry)
