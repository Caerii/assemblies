"""DEGREE accounting and norm_init's read-time scale.

Split out of `_sparse.py` 2026-08-10 as a PURE MOVE (see `_growth.py` for the
protocol). The concern: per-column nonzero counts maintained incrementally
(`_deg_counts` and its dirty/rewind bookkeeping) and the two consumers that
divide by them -- `_norm_scale`, the reference's one-time 1/d_j applied at
read time, and the candidate-divisor bridge. The fiber-p fix (79fba4f) lives
here: unmaterialized rows are priced at the FIBER's density, never the
brain's.
"""

from __future__ import annotations

import os

import numpy as np

from ..backend import to_cpu
from .._pricing import candidate_divisor, inverse_indegree
from ._csr_weights import CSRWeights
from ._virtual_weights import VirtualWeights


class DegreeNormMixin:
    """Degree/norm methods of `NumpySparseEngine`; see module docstring."""

    #: Set NEURAL_ASSEMBLIES_VERIFY_NNZ=1 to assert the incrementally
    #: maintained column counts against a full recount on every read. Slow, and
    #: the point: it converts "did I find every write site?" from an argument
    #: into a measurement. Run it over the suite before trusting the fast path.
    _VERIFY_NNZ = bool(int(os.environ.get("NEURAL_ASSEMBLIES_VERIFY_NNZ", "0")))



    @staticmethod
    def mark_region_refilled(conn, log_rows: int, log_cols: int) -> None:
        """Initialised content is about to be written below/right of the
        logical extent, into cells the degree counter may already have tallied.

        The counter reads up to the PHYSICAL array shape, which can run ahead of
        the LOGICAL content -- allocated-but-uninitialised cells read as zero
        and are counted as zero. When `_expand_connectomes` fills them the
        tally becomes wrong, which is what the verifier caught (252 mutations of
        already-counted blocks in one trial).

        The repair is exact rather than a blanket invalidation: the affected
        rows contributed EXACTLY ZERO to every column total, so rewinding the
        row watermark to `log_rows` and letting the incremental path re-add rows
        [log_rows, rows) reproduces the true count. Columns at or beyond
        `log_cols` were tallied as zero for the same reason and are simply
        recomputed.
        """
        have = int(getattr(conn, "_deg_rows", 0))
        if have > log_rows:
            conn._deg_rows = int(log_rows)
        counts = getattr(conn, "_deg_counts_arr", None)
        if counts is not None and len(counts) > log_cols:
            d = getattr(conn, "_deg_dirty", None)
            stale = set(range(int(log_cols), len(counts)))
            conn._deg_dirty = stale if d is None else (d | stale)

    @staticmethod
    def mark_column_dirty(conn, col_idx: int) -> None:
        """Record that *col_idx*'s nonzero pattern changed inside OLD rows.

        Growth into fresh rows or fresh columns is handled by the incremental
        arithmetic in `_deg_counts`. This exists for the one thing that
        arithmetic cannot see: a write that lands in a region already counted.
        `sample_new_winner_inputs` does exactly that -- it writes
        ``weights[chosen, col_idx] = 1.0`` where `chosen` are EXISTING source
        rows, and `_expansion_col` can map a first-time winner onto a column
        that was already materialised (index reuse).

        Measured before this was handled: 56.6% of previously counted blocks had
        changed by the next read, by up to 7 synapses. An incremental count that
        ignores them is not an optimisation, it is a different divisor -- which
        showed up as margins of 11.995 against 12.697 while accuracy stayed
        identical at 64/64/64.
        """
        d = getattr(conn, "_deg_dirty", None)
        if d is None:
            conn._deg_dirty = {int(col_idx)}
        else:
            d.add(int(col_idx))

    def _deg_counts(self, conn, w, rows: int, cols: int):
        """Per-column nonzero counts over ``w[:rows, :cols]``, maintained.

        WHY THIS IS NOT A RECOUNT. The previous implementation cached on
        ``(rows, cols)`` and recomputed the whole block whenever either changed.
        Rows materialise ONE AT A TIME during training, so the key changed on
        80% of calls and each miss cost O(rows*cols). Profiled over a single
        ladder cell (n=4000, M=64, depth 3): 763 recounts touching 3.4 BILLION
        elements, which is the entire 28.1s that `_norm_scale` contributed to a
        64.2s trial -- 44% of the run, spent recomputing a quantity that changes
        by a handful of synapses at a time.

        Here the counts persist and are extended:
          * new COLUMNS are counted over the rows already accounted for;
          * new ROWS add their contribution to every tracked column;
          * columns flagged by `mark_column_dirty` are recomputed alone.
        Work therefore scales with what actually changed, O(dr*cols + rows*dc),
        rather than with the size of the matrix. `np.count_nonzero` replaces
        ``(w != 0).sum(axis=0)`` as well, which avoids materialising a boolean
        temporary the size of the block.

        Exactness is asserted rather than argued -- see `_VERIFY_NNZ`.
        """
        xp = self._xp
        if isinstance(w, VirtualWeights):
            # Degree of a virtual fiber is base-plus-overrides, recounted
            # from the hash when the cache is cold. Only norm_init consults
            # this, so norm-off organs never pay the regeneration.
            counts = getattr(conn, "_deg_counts_arr", None)
            if (counts is None or int(getattr(conn, "_deg_rows", 0)) != rows
                    or len(counts) < cols):
                counts = xp.asarray(
                    w.column_nnz(rows_known=rows), dtype=xp.float32)
                conn._deg_counts_arr = counts
                conn._deg_rows = rows
                conn._deg_dirty = None
            return counts[:cols]
        counts = getattr(conn, "_deg_counts_arr", None)
        have_rows = int(getattr(conn, "_deg_rows", 0))

        # Full (re)build: first use, or the block shrank, which the incremental
        # arithmetic is not defined for.
        if counts is None or have_rows > rows or len(counts) > w.shape[1]:
            if isinstance(w, CSRWeights) and rows >= w.shape[0]:
                # CSR already stores the column index of every nonzero, so this
                # is a bincount over nnz rather than a scan of n^2 cells. Only
                # valid for the whole block -- a row-limited count would need
                # the indptr walk, and a materialised block is always whole.
                counts = xp.asarray(w.column_nnz()[:cols], dtype=xp.float32)
            else:
                counts = xp.asarray(
                    np.count_nonzero(np.asarray(to_cpu(w))[:rows, :cols],
                                     axis=0), dtype=xp.float32)
            conn._deg_counts_arr = counts
            conn._deg_rows = rows
            conn._deg_dirty = None
        else:
            have_cols = len(counts)
            # NOTHING-TO-DO FAST PATH. Densifying `w` before checking whether
            # any of the three updates below actually applies cost a FULL
            # `csr_todense` of the block on every settle round -- 99 of them in
            # 100 rounds, 58% of the flip. A materialised block never grows and
            # is only ever written multiplicatively, so this is the common case
            # and it must not touch the matrix at all.
            if (cols <= have_cols and rows <= have_rows
                    and not getattr(conn, "_deg_dirty", None)):
                return conn._deg_counts_arr
            w_cpu = np.asarray(to_cpu(w))
            if cols > have_cols:
                add = xp.asarray(
                    np.count_nonzero(w_cpu[:have_rows, have_cols:cols], axis=0),
                    dtype=xp.float32)
                counts = xp.concatenate([counts, add])
                conn._deg_counts_arr = counts
            if rows > have_rows:
                tracked = len(counts)
                counts[:tracked] += xp.asarray(
                    np.count_nonzero(w_cpu[have_rows:rows, :tracked], axis=0),
                    dtype=xp.float32)
                conn._deg_rows = rows
                have_rows = rows
            dirty = getattr(conn, "_deg_dirty", None)
            if dirty:
                idx = [c for c in sorted(dirty) if c < len(counts)]
                if idx:
                    counts[idx] = xp.asarray(
                        np.count_nonzero(w_cpu[:have_rows, idx], axis=0),
                        dtype=xp.float32)
                conn._deg_dirty = None

        if self._VERIFY_NNZ:
            truth = np.count_nonzero(np.asarray(to_cpu(w))[:rows, :cols], axis=0)
            got = np.asarray(to_cpu(conn._deg_counts_arr))[:cols]
            if not np.array_equal(got, truth):
                bad = int(np.argmax(np.abs(got - truth)))
                raise AssertionError(
                    f"maintained column counts diverged at rows={rows} "
                    f"cols={cols}: column {bad} has {got[bad]} but the matrix "
                    f"has {truth[bad]}. A write site changed an already-counted "
                    f"region without calling mark_column_dirty()."
                )
        return conn._deg_counts_arr

    def _norm_scale(self, conn, n_pre: int, rows_known: int, needed: int,
                    p: float | None = None):
        """Per-postsynaptic-neuron read-time scale ``1/d_j`` for one fiber.

        Reproduces the reference implementation's ``norm_init``
        (``.reference/mdabagia-nemo/brain.py``: ``FFArea.normalize`` /
        ``RecurrentArea.normalize``, which are called ONLY from ``reset()`` --
        a ONE-TIME initialization, not ongoing homeostasis).  There, each
        fiber's weight matrix is divided by its own column sums exactly once at
        init, when every present weight is 1.  The column sum is therefore the
        postsynaptic neuron's IN-DEGREE ``d_j``, and normalization is precisely
        "initialize every incoming synapse of neuron j to ``1/d_j``".

        WHY A READ-TIME SCALE RATHER THAN SCALED WEIGHTS.  This engine
        materializes neurons lazily, so neuron j's full incoming column does
        not exist when it would need to be divided.  But plasticity here is
        purely MULTIPLICATIVE (``w *= 1 + beta``), so

            (w_0 / d_j) * prod_t (1 + beta_t) == (w_0 * prod_t (1 + beta_t)) / d_j

        i.e. dividing a neuron's *summed drive* by a per-neuron constant is
        algebraically identical to having initialized its incoming weights at
        ``1/d_j``.  Storage therefore stays on the unit scale -- which also
        preserves ``w_max``'s "multiples of the initial weight" semantics and
        keeps the lazily *sampled* candidate drive commensurable -- and the
        division happens at read time.

        HOW ``d_j`` IS OBTAINED.  Sampling ``d_j ~ Binomial(n_pre, p)``
        independently of the neuron's realized wiring was tried first and does
        nothing (measured: winner in-degree z 1.66 -> 1.53, overlap
        0.940 -> 0.916).  It cannot work: an independent draw does not cancel
        the neuron's ACTUAL degree advantage, it only adds noise.  The
        reference divides by the neuron's OWN column sum, so the divisor must
        track realized wiring.  Here that is

            d_j = (present synapses over the rows that exist so far)
                  + p * (rows that do not exist yet)

        which is an unbiased running estimate of the full-population in-degree:
        each row that later materializes contributes a present synapse with
        probability p, exactly the rate the second term assumes, so ``d_j``
        stays centred on ``n_pre * p`` while tracking each neuron's own
        realized excess.  Present synapses are COUNTED, not summed, so the
        divisor is potentiation-invariant, matching the reference's
        take-it-once-at-init semantics.

        For STIMULUS fibers ``n_pre`` is the TARGET area's ``n``, not the
        stimulus size.  The reference's inputs are areas of size n with only a
        cap of k neurons active, so every fiber delivers drive of order k/n.
        This engine's stimuli instead fire in full and are stored pre-summed,
        so dividing by their own in-degree would set every neuron's stimulus
        drive to exactly 1.0 -- a constant, which erases the stimulus
        representation entirely (constants do not affect top-k).  Treating a
        stimulus of size s as the active cap of an implicit input population of
        size n restores the reference's geometry: stimulus drive ~ s/n,
        recurrent drive ~ k/n, so the fibers compete on equal terms and the
        stimulus keeps its per-neuron identity.

        Returns ``None`` when norm_init is off or there is nothing to scale.
        """
        if not self.norm_init:
            return None
        xp = self._xp
        w = conn.weights
        if w is None or getattr(w, "size", 0) == 0:
            return None

        if getattr(w, "ndim", 1) == 2:
            rows = int(min(rows_known, w.shape[0]))
            cols = int(min(needed, w.shape[1]))
            if cols <= 0:
                return None
            deg = self._deg_counts(conn, w, rows, cols)[:cols]
            unknown = max(int(n_pre) - rows, 0)
        else:
            cols = int(min(needed, len(w)))
            if cols <= 0:
                return None
            # 1-D stimulus fiber: the stored value IS the observed in-degree
            # from the stimulus population -- but only until plasticity scales
            # it, so snapshot each column the first time it is seen (a fresh
            # column is always read before it is ever potentiated).
            base = getattr(conn, "_norm_deg_base", None)
            have = 0 if base is None else len(base)
            if have < cols:
                add = xp.asarray(w[have:cols], dtype=xp.float32)
                base = add if base is None or have == 0 else xp.concatenate(
                    [base, add])
                conn._norm_deg_base = base
            deg = base[:cols]
            unknown = max(int(n_pre) - int(rows_known), 0)

        # `rows_known` has already been folded into `unknown` above, because the
        # 1-D and 2-D branches count "rows that exist" differently. Pass the
        # residual directly by declaring zero known rows.
        #
        # THE FIBER'S p, NOT THE BRAIN'S. Unmaterialized rows contribute
        # `unknown * p` expected synapses, and they arrive at the fiber's own
        # density: pricing them at the global p under-estimated d_j 6.15x on
        # a p=0.4 fiber in a p=0.05 brain (130.0 vs 799.9), over-scaling the
        # drive by a factor that DRIFTS toward 1x as the source materializes
        # -- so training potentiated under a moving mis-scale. Same defect
        # class as the per-fiber w_max clamp: a scale derived from p while
        # the weights were drawn at a different p. Callers pass their fiber's
        # p; None means homogeneous and falls back to the brain's.
        return inverse_indegree(deg, unknown, 0,
                                self.p if p is None else float(p), xp=xp)

    def _norm_candidate_divisor(self, tgt, input_sizes=None,
                                src_pops=None, input_ps=None) -> float:
        """Scale applied to sampled candidate drive; see `core._pricing`.

        The law -- an activity-weighted harmonic mean of the source populations
        -- lives in `_pricing.candidate_divisor` so that this engine and the
        torch engine cannot drift apart again.  They did: this fix (54e6c00)
        never reached the torch mirror, and the two engines priced k-WTA
        differently until the law was unified.
        """
        return candidate_divisor(self.p if input_ps is None else input_ps,
                                 tgt.n, input_sizes, src_pops)

    # -- Registration -------------------------------------------------------
