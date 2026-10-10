"""Composable pieces for an area whose connectome is GENERATED, not stored.

WHY THIS SHAPE. The algebra composes, so the code should:

    drive[j] = SUM over afferent FIBERS f of   norm_f(j) * SUM_{i in rows_f} w_f[i,j]

and the engine already works this way -- `_norm_scale` is called per fiber,
`_scale_columns_now` iterates the fibers of a target. So a fiber owns three
things and an area owns none of them:

    its BASE        a hash, `w[i,j] != 0 iff float24(fmix32(...)) < p`
    its DEVIATIONS  what training changed, [[HEBB-OUTER-PRODUCT]]
    its PRICE       the norm divisor, and the column scale if any

Adding an input is then adding a fiber to a list, not adding a flag to a
projection -- which is what `batched_project_hashed` had started to become.

TWO KINDS OF FIBER, and the difference is not cosmetic. An area->area fiber is
2-D and its deviations are keyed `(i, j)`. A stimulus fiber stores PRE-SUMMED
input -- one weight per target neuron, equal to how many stimulus neurons wire
to it -- so its deviations are keyed by `j` alone, its `w_max` cap is scaled by
the initial magnitude `stim_size * p`, and its norm divisor is priced at the
TARGET's n rather than its own size. Getting either wrong is silent and has a
direction; see `AreaFiber.contribute` and `StimulusFiber.contribute`.

THE DEVIATION STORE. Within one episode the co-firing record is a single
64-bit round mask per neuron. At the end of the episode it is folded into a
sorted (key, count) store via a GEMM and never walked again:

    query cost   O(sum_{i in S} nnz_i)      -- the store, read BY ROW
    versus       O(k n W), W = rounds / 64  -- a mask, which cannot say WHICH
                                               cells are nonzero

a ratio `n^2 T / (64 k^2)` that is independent of M ([[DRIVE-SPLIT]]) -- 8894x
at n=16000, k=60, T=8. Keeping the mask across episodes made W grow with
training length and a study's cost quadratic in it.

THE MODULES. This file is the facade; the pieces live in

    _hashed_common.py     per-brain values, gain tables, clip_count, local indices
    _hashed_store.py      RunStore and the STORE fiber (AreaFiber)
    _hashed_present.py    the PRESENT-ONLY fiber (PresentFiber)
    _hashed_organ.py      the DENSE ORGAN fiber (DenseOrganFiber)
    _hashed_stimulus.py   the STIMULUS fiber (StimulusFiber)
    _hashed_area.py       the AREA (HashedArea)

and every name below is re-exported, so `from ._hashed import X` is unchanged.
"""
from __future__ import annotations

from . import _fused_cuda
from .._pricing import (chain_table as _chain_table,
                        count_saturation_is_exact,
                        gain_table as _gain_table,
                        relative_table as _rel_table)
from ._hashed_common import (_GAIN_TABLES, _device_gain, _local_index, _raise_overflow,
                             clip_count, per_brain)
from ._hashed_store import AreaFiber, RunStore
from ._hashed_present import PresentFiber
from ._hashed_organ import DenseOrganFiber
from ._hashed_stimulus import StimulusFiber
from ._hashed_area import HashedArea

__all__ = ["AreaFiber", "DenseOrganFiber", "HashedArea", "PresentFiber", "RunStore", "StimulusFiber",
           "clip_count", "per_brain", "_GAIN_TABLES", "_device_gain", "_local_index", "_raise_overflow",
           "_fused_cuda", "_chain_table", "count_saturation_is_exact", "_gain_table", "_rel_table"]
