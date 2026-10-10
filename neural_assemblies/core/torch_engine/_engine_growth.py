"""Connectome growth for the torch engine: newly recruited neurons get their synapses
(hash-derived, content-addressed), fibers that keep being rebuilt are densified, and an
area can be materialized whole.

A mixin of TorchSparseEngine (_engine.py), which owns the state these methods read;
the methods were moved out of _engine.py unchanged."""
import numpy as np

from ._torch_ops import torch_ops

from ._hash import (
    WEIGHT_DTYPE, hash_stim_counts,
    hash_bernoulli_coo,
)
from ._csr import (
    CSRConn, DENSIFY_MAX_BYTES, DENSIFY_MIN_NNZ,
    densify,
)
from ._state import LAZY_ID_THRESHOLD


class GrowthMixin:
    """Synapses for recruited neurons; densification; materialization."""

    # -- Connectome expansion -----------------------------------------------

    def _maybe_densify(self, src_name, target):
        """Swap a rebuild-bound CSR fiber for dense storage; return the conn.

        Call at every GROWTH site before touching the fiber. Density picks
        the representation at `add_connectivity` time, but a low-density
        fiber that grows every recruitment step (an area self-fiber during
        training) is just as rebuild-bound once it is large -- see
        DENSIFY_MIN_NNZ. Budgeted against the fiber's FINAL dense footprint
        (n_src x n_tgt), not its current extent, so a fiber that will not
        fit never starts migrating.
        """
        conn = self._area_conns[src_name][target]
        if (isinstance(conn, CSRConn)
                and conn.nnz >= DENSIFY_MIN_NNZ):
            final_bytes = (int(self._areas[src_name].n)
                           * int(self._areas[target].n) * 2)
            if final_bytes <= DENSIFY_MAX_BYTES:
                conn = densify(conn, device=self._device,
                               max_rows=int(self._areas[src_name].n),
                               max_cols=int(self._areas[target].n))
                self._area_conns[src_name][target] = conn
        return conn

    def _hash_grow_parts(self, csr, pair_seed, fiber_p,
                         needed_rows, needed_cols):
        """L-shaped hash init of a CSR fiber's uncovered region, as COO parts.

        The one canonical implementation of the grow-to-extent step (it was
        inline in `_expand_connectomes`; `materialize_area` needs the same
        mechanics, and two copies of an init path is how the numpy engine
        got its self-fiber masking defects). Content-addressed: entries
        depend only on (pair_seed, absolute position, fiber_p), so growing
        in any order yields the same fiber. Updates the logical extents;
        the caller merges the returned parts via ``csr.expand``.
        """
        log_rows = csr._log_rows
        log_cols = csr._log_cols
        coo_r, coo_c, coo_v = [], [], []
        if needed_rows > log_rows or needed_cols > log_cols:
            regions = []
            # Block A: new rows x existing cols
            if needed_rows > log_rows and log_cols > 0:
                regions.append((log_rows, needed_rows, 0, log_cols))
            # Block B: existing rows x new cols
            if needed_cols > log_cols and log_rows > 0:
                regions.append((0, log_rows, log_cols, needed_cols))
            # Block C: new rows x new cols
            if needed_rows > log_rows and needed_cols > log_cols:
                regions.append((log_rows, needed_rows, log_cols, needed_cols))
            for r0, r1, c0, c1 in regions:
                r, c, v = hash_bernoulli_coo(
                    r0, r1, c0, c1, pair_seed, fiber_p, device=self._device)
                if len(r) > 0:
                    coo_r.append(r)
                    coo_c.append(c)
                    coo_v.append(v)
            csr._log_rows = max(log_rows, needed_rows)
            csr._log_cols = max(log_cols, needed_cols)
        return coo_r, coo_c, coo_v

    def materialize_area(self, area: str, storage: str = "csr") -> int:
        """Bring ALL ``n`` of an area's neurons into existence at once.

        The torch port of `NumpySparseEngine.materialize_area` (see its
        docstring for WHY this exists: any protocol that drives an area from
        an arbitrary subset of ``n`` -- assigned blocks, uniform seeds --
        silently reads zeros from neurons lazy materialization has not
        created yet). Init is content-addressed by absolute position, so a
        materialized-all-at-once area has exactly the weights it would have
        had if the same neurons had been recruited one at a time.

        This engine's native representation is CSR.  A dense request is
        rejected instead of being silently accepted and stored as CSR; the
        representation is part of a benchmark's protocol and must be real.

        Returns the number of neurons newly materialized.
        """
        if storage != "csr":
            raise ValueError(
                "torch_sparse materialize_area supports storage='csr' only; "
                f"got {storage!r}"
            )
        tgt = self._areas[area]
        prior_w = int(tgt.w)
        n = int(tgt.n)
        if prior_w >= n:
            return 0
        if tgt._lazy_ids:
            raise NotImplementedError(
                f"materialize_area({area!r}): lazy neuron-id mode "
                f"(n > {LAZY_ID_THRESHOLD}) has no full-permutation pool to "
                f"draw identities from.")

        # 1. Compact ids for every remaining neuron, drawn from the same
        #    shuffled pool the incremental path consumes, so identities match.
        if tgt.neuron_id_pool is not None:
            pool = np.asarray(tgt.neuron_id_pool)
            need = n - len(tgt.compact_to_neuron_id)
            ptr = int(tgt.neuron_id_pool_ptr)
            take = pool[ptr:ptr + need]
            tgt.compact_to_neuron_id.extend(int(x) for x in take)
            tgt.neuron_id_pool_ptr = ptr + len(take)
        while len(tgt.compact_to_neuron_id) < n:
            tgt.compact_to_neuron_id.append(len(tgt.compact_to_neuron_id))

        # 2. Stim -> area vectors out to n.
        for stim_name, conns in self._stim_conns.items():
            conn = conns.get(area)
            if conn is None or not conn.sparse or conn.weights is None:
                continue
            old = int(conn.weights.numel())
            if old < n:
                add = hash_stim_counts(
                    self._stimuli[stim_name].size, old, n,
                    self._get_pair_seed(stim_name, area),
                    self._p_for(stim_name, area), device=self._device)
                conn.weights = torch_ops.cat([conn.weights, add])

        # 3. Area fibers: hash-grow every block touching this area to full
        #    extent. IN-fibers gain columns; OUT-fibers gain rows; the self
        #    fiber gains both (covered by the first loop, then skipped).
        def _grow(csr, src_name, tgt_name, needed_rows, needed_cols):
            r, c, v = self._hash_grow_parts(
                csr, self._get_pair_seed(src_name, tgt_name),
                self._p_for(src_name, tgt_name), needed_rows, needed_cols)
            if r:
                csr.expand(csr._log_rows, csr._log_cols,
                           torch_ops.cat(r), torch_ops.cat(c), torch_ops.cat(v))

        for src_name, conns in self._area_conns.items():
            csr = conns.get(area)
            if csr is None:
                continue
            src_rows = (n if src_name == area
                        else max(int(self._areas[src_name].w),
                                 csr._log_rows))
            _grow(csr, src_name, area, src_rows, n)
        for tgt_name, csr in self._area_conns.get(area, {}).items():
            if tgt_name == area:
                continue  # self fiber handled above
            tgt_cols = max(int(self._areas[tgt_name].w), csr._log_cols)
            _grow(csr, area, tgt_name, n, tgt_cols)

        tgt.w = n
        return n - prior_w

    def _expand_connectomes(self, target, from_stimuli, from_areas,
                            input_sizes, winners, first_winner_inputs,
                            new_w):
        """Expand connectivity for first-time winners using hash-based init."""
        tgt = self._areas[target]
        inputs_names = list(from_stimuli) + list(from_areas)

        splits_per_new = self._sparse_sim.compute_input_splits(
            input_sizes, first_winner_inputs)

        prior_w = tgt.w
        new_indices = [int(w) for w in winners if int(w) >= prior_w]

        stim_names = [n for n in inputs_names if n in self._stimuli]
        area_names = [n for n in inputs_names if n in self._areas]

        # --- Expand stim->area 1-D vectors ---
        for stim_name in self._stimuli.keys():
            conn = self._stim_conns[stim_name][target]
            if conn.sparse:
                old = len(conn.weights)
                if new_w > old:
                    add_len = new_w - old
                    if stim_name not in stim_names:
                        pair_seed = self._get_pair_seed(stim_name, target)
                        add = hash_stim_counts(
                            self._stimuli[stim_name].size, old, new_w,
                            pair_seed, self._p_for(stim_name, target),
                            device=self._device)
                    else:
                        add = torch_ops.zeros(add_len, dtype=WEIGHT_DTYPE,
                                          device=self._device)
                    conn.weights = torch_ops.cat([conn.weights, add])

        # Write allocations for firing stimuli
        for idx, win in enumerate(new_indices):
            if win >= new_w:
                continue
            split = (splits_per_new[idx]
                     if idx < len(splits_per_new) else None)
            if split is None:
                continue
            for j, name in enumerate(inputs_names):
                alloc = int(split[j])
                if name in self._stimuli:
                    conn = self._stim_conns[name][target]
                    if conn.sparse and win < len(conn.weights):
                        conn.weights[win] = alloc

        # --- Expand area->area CSR matrices ---
        for src_name in area_names:
            csr = self._maybe_densify(src_name, target)
            src = self._areas[src_name]
            pair_seed = self._get_pair_seed(src_name, target)

            src_w_arr = src.winners.long()
            max_src_idx = (
                (int(src_w_arr.max()) + 1)
                if src_w_arr.numel() > 0 else 0)
            # Never shrink below pregrown CSR extent (CONTEXT w can reset to 0
            # while connectome topology is preserved — matches numpy dense path).
            needed_rows = max(
                max_src_idx,
                new_w if src_name == target else src.w,
                csr._nrows,
            )
            needed_cols = max(new_w, csr._ncols)

            coo_r_parts, coo_c_parts, coo_v_parts = self._hash_grow_parts(
                csr, pair_seed, self._p_for(src_name, target),
                needed_rows, needed_cols)

            # Explicit entries from first-timer allocations
            from_index = inputs_names.index(src_name)
            local_rng = np.random.default_rng(
                self._rng.integers(0, 2**32))
            src_winners_cpu = src.winners.cpu().numpy().astype(np.int64)

            # VECTORISED PER NEW WINNER. The draws stay one-per-winner in the
            # same order (they consume `local_rng`, so any reordering would
            # change the connectome), but the chosen rows are kept as ARRAYS
            # instead of appended element by element: the inner Python loop
            # ran once per synapse -- 1.78M list appends in a 4-presentation
            # build -- and then paid again handing a multi-million-element
            # Python list to `torch_ops.tensor`. Same values, same order, one
            # host->device copy.
            exp_rows_parts, exp_cols_parts = [], []
            for idx, win in enumerate(new_indices):
                alloc = (
                    int(splits_per_new[idx][from_index])
                    if idx < len(splits_per_new) else 0)
                if alloc <= 0 or src.w == 0:
                    continue
                sample_size = min(alloc, len(src.winners))
                if sample_size <= 0:
                    continue
                chosen = local_rng.choice(
                    src_winners_cpu, size=sample_size, replace=False)
                col_idx = win - prior_w
                if col_idx < 0 or col_idx >= needed_cols:
                    continue
                chosen = chosen[(chosen >= 0) & (chosen < needed_rows)]
                if chosen.size == 0:
                    continue
                exp_rows_parts.append(chosen)
                exp_cols_parts.append(
                    np.full(chosen.size, col_idx, dtype=np.int64))

            if exp_rows_parts:
                exp_rows = np.concatenate(exp_rows_parts).astype(
                    np.int32, copy=False)
                exp_cols = np.concatenate(exp_cols_parts).astype(
                    np.int32, copy=False)
                coo_r_parts.append(
                    torch_ops.from_numpy(exp_rows).to(self._device))
                coo_c_parts.append(
                    torch_ops.from_numpy(exp_cols).to(self._device))
                coo_v_parts.append(torch_ops.ones(
                    exp_rows.size, dtype=WEIGHT_DTYPE,
                    device=self._device))

            # Merge into CSR
            if coo_r_parts:
                new_r = torch_ops.cat(coo_r_parts)
                new_c = torch_ops.cat(coo_c_parts)
                new_v = torch_ops.cat(coo_v_parts)
                csr.expand(needed_rows, needed_cols, new_r, new_c, new_v)
            elif needed_rows > csr._nrows or needed_cols > csr._ncols:
                e = torch_ops.empty(0, dtype=torch_ops.int32, device=self._device)
                csr.expand(needed_rows, needed_cols,
                           e, e.clone(),
                           torch_ops.empty(0, dtype=WEIGHT_DTYPE,
                                       device=self._device))
