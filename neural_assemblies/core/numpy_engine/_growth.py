"""Connectome GROWTH: recruitment expansion, coverage, stim-vector growth.

Split out of `_sparse.py` 2026-08-10 as a PURE MOVE -- method bodies are
verbatim, called through the same `self`, guarded by
`test_connectome_representation_fingerprint` (byte identity) and the engine
surface. The concern boundary: everything here decides how a fiber's stored
extent GROWS -- amortised reallocation, the L-shaped content-addressed fill,
first-winner edge allocations, stim-vector capacity. Nothing here computes
drive or applies plasticity.

This is where a virtual (never-stored) representation plugs in: growth is the
concern that pays the 2x reallocation peak, and the one that becomes two
integer updates when the base is recomputable.
"""

from __future__ import annotations

import os

import numpy as np

from ..backend import to_cpu
from ._csr_weights import CSRWeights
from ._seeding import stable_seed

def _self_fiber_deferred_init() -> bool:
    """Whether a SELF fiber (``A -> A``) gets deferred block initialization.

    **OFF by default, and that is a staging decision, not a verdict on the
    defect.**  Excluding self fibers leaves a whole class of recurrence
    silently inert, and that IS a bug.  But turning it on repairs ~4,900 dead
    deliveries at once, and the repository is calibrated on the inert
    behaviour: measured over the full ``not slow`` suite, ON takes it from
    **5 failures to 21** -- ERP calibration, noise robustness, cross-repo
    parity, engine E2 overlap, simulation integration and three torch-parity
    cells all move, because each was reading "preserve the current assembly"
    and now reads what its fiber actually delivers.

    ``ops.project``'s docstring already prices exactly this class of change:
    *"the default is kept WRONG on purpose ... flipping it is a migration with
    its own re-baseline, not a bug fix."*  The same applies here.  Set
    ``ASSEMBLIES_SELF_FIBER_INIT=1`` to run with the defect repaired; the
    migration is tracked as #72 and #70.

    Note the two PNAS protocol corrections that came out of this investigation
    are INDEPENDENT of this flag and are already live: they pass
    ``recurrent=True``, so ``A -> A`` is exercised while the area is still
    recruiting and ``_expand_connectomes`` sizes the block the ordinary way.
    The flag only matters for a self fiber first driven AFTER its area stops
    growing.

    THE DEFECT.  ``project_into`` marks an empty area->area block for deferred
    sizing so it can carry drive on the NEXT round -- but the guard read
    ``src_name != target``, so a self fiber was never marked.  Self blocks are
    otherwise sized only as a side effect of the target RECRUITING, via
    ``_expand_connectomes``.  So a self fiber first driven AFTER its area has
    stopped growing is **permanently** dead: it delivers exactly zero, and
    ``project_into`` then takes the "zero signal -- preserve current assembly"
    branch and hands back the incumbent winners.  The projection looks like it
    worked.  It returns ``k`` winners.  They are simply the ones already there.

    WHAT IT COST, measured by ``research/experiments/dormant_mechanism_sweep.py``
    over a full suite slice (275,834 ``project()`` calls): 27 fibers dead on
    100% of their uses, 4,910 dead deliveries, and 23 of the 27 were self
    fibers -- ``VP->VP`` 84/84, ``PREDICTION->PREDICTION`` 55/55, and every coin
    and PFA settle loop.

    The sharpest case is ``RandomChoiceArea._flip_k_split``, which builds a
    k-split mix with ``rng.choice`` and then "settles" it with
    ``project({}, {area: [area]})``.  With the self block at ``(0, 0)`` that
    loop does nothing at all::

        rounds = 0 / 1 / 10   ->  200/200 per-flip agreement, identical heads
        COIN->COIN block      ->  shape (0, 0), nnz 0
        winners moved         ->  0 of 10 rounds
        cross-fiber control   ->  moves winners 2/5 rounds

    i.e. the neural coin's outcome was decided entirely by the numpy RNG that
    built the mix, and the substrate contributed nothing -- under ``test_pfa``,
    ``test_nemo_fsm``, ``test_computation_value``, four ``TestCoin2024*Golden``
    classes and the ``coin2024_*`` parity protocols.

    ``NumpySparseEngine.ensure_area_conn`` is the explicit repair for exactly
    this situation and ``binding.py`` calls it with that reasoning in a comment.
    Across the same slice it was invoked ZERO times in 631,925 projections: the
    disease occurred 27 times over and the cure was never reached.

    Set ``ASSEMBLIES_SELF_FIBER_INIT=0`` to restore the old behaviour, which is
    required to reproduce any number recorded before this fix -- and every coin
    and PFA figure in the repo is such a number.
    """
    return os.environ.get(
        "ASSEMBLIES_SELF_FIBER_INIT", "0",
    ).strip().lower() in ("1", "true", "yes", "on")



class GrowthMixin:
    """Growth methods of `NumpySparseEngine`; see module docstring."""


    def _init_deferred_area_srcs(self, target, src_names, new_w) -> None:
        """Repair the area->area blocks marked as undelivering this projection.

        Two repair cases, same next-round semantics:

        EMPTY. A source area whose connectome into *target* has never been
        sized contributes nothing on the round it is first used; this gives it
        a Bernoulli(p) block over the target's currently materialised neurons
        so it can contribute on the NEXT round. The stimulus path has carried
        the same fix since `add_stimulus` ("without this the empty weight
        vector produces zero input and the projection short-circuits"); this
        is its area->area counterpart.

        STALE (#151 dead fiber). A block that EXISTS but whose initialised
        rows/cols no longer cover (src.w, tgt.w). Row/col growth used to
        happen only inside `_expand_connectomes`, which returns early when the
        target recruits no first-time winner -- so one no-recruitment episode
        after the source had grown froze the fiber forever: out-of-range rows
        are silently dropped from drive, zero drive recruits nobody, and no
        recruitment means no expansion. MEASURED (dead_fiber_hunt, Brown at
        n=1e5): seed 45's NOUN_CORE->NUMBER_PL froze at 180 rows while the
        source grew to 18641, reading own-drive exactly 0.0 on 45/46 trained
        words. Content-addressed init makes this repair value-identical to
        the expansion the recruit path would have done.

        Called from BOTH exits of `project_into` -- the normal one and the
        zero-signal early return. Only reaching it from the normal exit made it
        unreachable whenever the new source was the only source, which is
        exactly when it is needed.
        """
        if not src_names:
            return
        for src_name in src_names:
            conn = self._area_conns[src_name][target]
            nr, nc = int(self._areas[src_name].w), int(new_w)
            if getattr(conn.weights, "shape", (0, 0))[1] > 0:
                self._ensure_area_block_coverage(src_name, target, conn,
                                                 nr, nc)
                continue
            if nr > 0 and nc > 0:
                conn.weights = self._init_area_block(
                    src_name, target, 0, nr, 0, nc)

    def _ensure_area_block_coverage(self, src_name, target, conn,
                                    needed_rows, needed_cols) -> None:
        """Grow an EXISTING block so its INITIALISED region covers
        [0, needed_rows) x [0, needed_cols).

        The counterpart of `_expand_connectomes`' growth step for the
        no-recruitment case (see `_init_deferred_area_srcs`). Mirrors its
        mechanics exactly: amortised physical realloc, L-shaped
        content-addressed fill, degree-counter rewind, watermark update. In
        deterministic mode it grows EXACTLY (no doubling), because that
        branch of `_expand_connectomes` treats the physical shape as the
        logical extent and would never fill padding this method left behind.

        No-op on CSR-stored, dense/explicit, and 1-D blocks -- only the lazy
        2-D ndarray representation has the recruitment-gated growth defect.
        """
        xp = self._xp
        w = conn.weights
        if (not conn.sparse or isinstance(w, CSRWeights)
                or getattr(w, "ndim", 0) != 2):
            return
        phys_rows, phys_cols = w.shape
        if phys_cols == 0:
            return  # empty block: the EMPTY repair path owns this case
        log_rows = min(int(getattr(conn, "_log_rows", phys_rows)), phys_rows)
        log_cols = min(int(getattr(conn, "_log_cols", phys_cols)), phys_cols)
        needed_rows = max(int(needed_rows), log_rows)
        needed_cols = max(int(needed_cols), log_cols)
        if needed_rows <= log_rows and needed_cols <= log_cols:
            return
        src = self._areas[src_name]
        tgt = self._areas[target]
        new_pr, new_pc = phys_rows, phys_cols
        if needed_rows > phys_rows:
            new_pr = (needed_rows if self._deterministic else
                      min(max(needed_rows, phys_rows * 2, 2 * src.k),
                          max(int(src.n), needed_rows)))
        if needed_cols > phys_cols:
            new_pc = (needed_cols if self._deterministic else
                      min(max(needed_cols, phys_cols * 2, 2 * tgt.k),
                          max(int(tgt.n), needed_cols)))
        if (new_pr, new_pc) != (phys_rows, phys_cols):
            buf = xp.zeros((new_pr, new_pc), dtype=xp.float32)
            if phys_rows > 0 and phys_cols > 0:
                buf[:phys_rows, :phys_cols] = w
            conn.weights = buf
        nr = needed_rows - log_rows
        nc = needed_cols - log_cols
        self.mark_region_refilled(conn, log_rows, log_cols)
        if nr > 0 and log_cols > 0:
            conn.weights[log_rows:needed_rows, :log_cols] = (
                self._init_area_block(src_name, target, log_rows,
                                      needed_rows, 0, log_cols))
        if nc > 0 and log_rows > 0:
            conn.weights[:log_rows, log_cols:needed_cols] = (
                self._init_area_block(src_name, target, 0, log_rows,
                                      log_cols, needed_cols))
        if nr > 0 and nc > 0:
            conn.weights[log_rows:needed_rows, log_cols:needed_cols] = (
                self._init_area_block(src_name, target, log_rows,
                                      needed_rows, log_cols, needed_cols))
        conn._log_rows = max(int(getattr(conn, "_log_rows", 0)), needed_rows)
        conn._log_cols = max(int(getattr(conn, "_log_cols", 0)), needed_cols)

    def _expand_connectomes(self, target, from_stimuli, from_areas,
                            input_sizes, winners, first_winner_inputs, new_w):
        """Expand connectivity for first-time winners.

        Uses amortised buffer growth for 2-D area->area matrices: physical
        capacity doubles when exceeded, avoiding repeated vstack/hstack
        reallocation on every step.
        """
        xp = self._xp
        tgt = self._areas[target]
        inputs_names = list(from_stimuli) + list(from_areas)

        prior_w = tgt.w
        new_indices = [int(w) for w in winners if int(w) >= prior_w]
        if not new_indices:
            return

        splits_per_new = self._sparse_sim.compute_input_splits(
            input_sizes, first_winner_inputs,
        )

        if getattr(tgt, '_freeze_connectome_growth', False):
            area_names = [name for name in inputs_names if name in self._areas]
            can_freeze = True
            for src_name in area_names:
                conn = self._area_conns[src_name][target]
                if not conn.sparse or conn.weights.ndim != 2:
                    continue
                phys_rows, phys_cols = conn.weights.shape
                src = self._areas[src_name]
                src_w_arr = xp.asarray(src.winners)
                max_src_idx = (
                    int(xp.max(src_w_arr)) + 1 if src_w_arr.size > 0 else 0
                )
                needed_rows = max(
                    max_src_idx,
                    new_w if src_name == target else src.w,
                )
                needed_cols = new_w
                if needed_rows > phys_rows or needed_cols > phys_cols:
                    can_freeze = False
                    break
            if can_freeze:
                for src_name in area_names:
                    conn = self._area_conns[src_name][target]
                    if not conn.sparse or conn.weights.ndim != 2:
                        continue
                    _, phys_cols = conn.weights.shape
                    self._write_area_expansion_edges(
                        src_name, target, conn, inputs_names,
                        new_indices, splits_per_new, prior_w, phys_cols,
                    )
                return

        # --- Expand stim->area 1-D vectors ---
        firing_stimuli = [name for name in inputs_names if name in self._stimuli]
        area_names = [name for name in inputs_names if name in self._areas]

        if new_w > prior_w:
            # The fast path batches background draws by stimulus SIZE, which
            # assumes one density for all of them. With per-fiber `p` set, take
            # the per-fiber-correct path instead; the two are otherwise
            # bit-identical, so homogeneous brains keep the fast one.
            if self._stim_fastpath and not self.heterogeneous():
                self._expand_stim_vectors_fast(target, tgt, firing_stimuli, new_w)
            else:
                self._expand_stim_vectors_legacy(target, firing_stimuli, new_w)

        # Write allocations for firing stimuli
        for idx, win in enumerate(new_indices):
            if win >= new_w:
                continue
            split = splits_per_new[idx] if idx < len(splits_per_new) else None
            if split is None:
                continue
            for j, name in enumerate(inputs_names):
                alloc = int(split[j])
                if name in self._stimuli:
                    conn = self._stim_conns[name][target]
                    if conn.sparse and win < len(conn.weights):
                        conn.weights[win] = alloc

        # --- Expand area->area 2-D matrices ---
        for src_name in area_names:
            conn = self._area_conns[src_name][target]
            if not conn.sparse:
                continue
            src = self._areas[src_name]
            if conn.weights.ndim != 2:
                conn.weights = xp.empty((0, 0), dtype=xp.float32)

            phys_rows, phys_cols = conn.weights.shape
            src_w_arr = xp.asarray(src.winners)
            max_src_idx = (int(xp.max(src_w_arr)) + 1) if src_w_arr.size > 0 else 0
            needed_rows = max(max_src_idx, new_w if src_name == target else src.w)
            needed_cols = new_w

            if self._deterministic:
                if needed_rows > phys_rows:
                    new_rows = self._init_area_block(
                        src_name, target, phys_rows, needed_rows, 0, phys_cols)
                    conn.weights = xp.vstack([conn.weights, new_rows]) if phys_cols > 0 else xp.zeros((needed_rows, 0), dtype=xp.float32)
                    phys_rows = needed_rows
                if needed_cols > phys_cols:
                    new_cols = self._init_area_block(
                        src_name, target, 0, phys_rows, phys_cols, needed_cols)
                    conn.weights = xp.hstack([conn.weights, new_cols]) if phys_rows > 0 else xp.zeros((0, needed_cols), dtype=xp.float32)
                    phys_cols = needed_cols
            else:
                log_rows = getattr(conn, '_log_rows', phys_rows)
                log_cols = getattr(conn, '_log_cols', phys_cols)
                if log_rows > phys_rows or log_cols > phys_cols:
                    log_rows = min(log_rows, phys_rows)
                    log_cols = min(log_cols, phys_cols)

                new_pr, new_pc = phys_rows, phys_cols
                need_realloc = False
                # CLAMPED TO n, like `_stim_capacity` does for the 1-D case.
                # Rows index presynaptic neurons of `src` and columns
                # postsynaptic neurons of `tgt`, so neither can logically
                # exceed that area's n -- `needed_*` is bounded by `w <= n`.
                # Unclamped doubling overshot it: a LEX with n=10000, w=7918
                # held a (15682, 15682) matrix, 984 MB where 400 MB is the
                # most the fiber can ever need, and up to ~4x n^2 in the
                # worst case. Consumers slice by the LOGICAL w (see
                # `_norm_*`), so the padding never changed a result -- it
                # only wasted memory and crashed `materialize_area`, which
                # reasonably assumed no block exceeds n.
                if needed_rows > phys_rows:
                    new_pr = min(max(needed_rows, phys_rows * 2, 2 * src.k),
                                 max(int(src.n), int(needed_rows)))
                    need_realloc = True
                if needed_cols > phys_cols:
                    new_pc = min(max(needed_cols, phys_cols * 2, 2 * tgt.k),
                                 max(int(tgt.n), int(needed_cols)))
                    need_realloc = True

                if need_realloc:
                    buf = xp.zeros((new_pr, new_pc), dtype=xp.float32)
                    if phys_rows > 0 and phys_cols > 0:
                        buf[:phys_rows, :phys_cols] = conn.weights
                    conn.weights = buf
                    phys_rows, phys_cols = new_pr, new_pc

                # The three pieces of the L-shaped new region. Under
                # content addressing each is written at its ABSOLUTE position,
                # so splitting the region this way is invisible: the same cell
                # gets the same value whichever piece happens to cover it.
                nr = needed_rows - log_rows
                nc = needed_cols - log_cols
                # The degree counter reads up to the PHYSICAL shape, which can
                # run ahead of the LOGICAL content: rows/cols that are
                # allocated but not yet initialised read as zero and get
                # counted as zero. Filling them below therefore changes a
                # region the counter already tallied. Rewind precisely -- those
                # rows contributed exactly 0, so re-adding them is exact -- and
                # flag the columns about to be initialised.
                self.mark_region_refilled(conn, log_rows, log_cols)
                if nr > 0 and log_cols > 0:
                    conn.weights[log_rows:needed_rows, :log_cols] = (
                        self._init_area_block(src_name, target, log_rows,
                                            needed_rows, 0, log_cols))
                if nc > 0 and log_rows > 0:
                    conn.weights[:log_rows, log_cols:needed_cols] = (
                        self._init_area_block(src_name, target, 0, log_rows,
                                            log_cols, needed_cols))
                if nr > 0 and nc > 0:
                    conn.weights[log_rows:needed_rows, log_cols:needed_cols] = (
                        self._init_area_block(src_name, target, log_rows,
                                            needed_rows, log_cols, needed_cols))

                conn._log_rows = max(getattr(conn, '_log_rows', 0), needed_rows)
                conn._log_cols = max(getattr(conn, '_log_cols', 0), needed_cols)

            # -- Write specific allocations for first-time winners --
            from_index = inputs_names.index(src_name)
            # Which presynaptic winners a first-time winner attaches to is part
            # of INITIALISATION, so it is keyed on the fiber and the growth
            # point rather than drawn from the shared stream. Left on the
            # stream it would reintroduce the order dependence one layer below
            # the weights themselves.
            local_rng = np.random.default_rng(
                stable_seed(self._seed, src_name, target, prior_w, new_w)
                if self._content_init
                else self._rng.integers(0, 2**32))
            src_winners_cpu = np.asarray(
                to_cpu(src.winners) if hasattr(src.winners, 'get') else src.winners
            )
            for idx, win in enumerate(new_indices):
                alloc = int(splits_per_new[idx][from_index]) if idx < len(splits_per_new) else 0
                if alloc <= 0 or src.w == 0:
                    continue
                sample_size = min(alloc, len(src.winners))
                if sample_size <= 0:
                    continue
                chosen = local_rng.choice(src_winners_cpu, size=sample_size, replace=False)
                col_idx = self._expansion_col(int(win), prior_w)
                if 0 <= col_idx < phys_cols:
                    conn.weights[chosen, col_idx] = 1.0
                    # `chosen` are EXISTING rows and `_expansion_col` can reuse
                    # an already-materialised column, so this write can land
                    # inside a region the degree counter has already tallied.
                    self.mark_column_dirty(conn, col_idx)

    # -- stim->area vector growth -------------------------------------------

    def _stim_conns_for(self, target: str):
        """Cached ``[(stim_name, connectome)]`` for *target*, in registration order.

        Same sequence the legacy path gets from iterating ``_stim_conns.items()``
        and dropping targets with no connectome, so the order in which
        stimuli are offered to the growth loop -- and therefore the order in
        which they draw from ``self._rng`` -- is unchanged.
        """
        cached = self._stim_target_cache.get(target)
        if cached is not None and cached[0] == self._stim_conn_version:
            return cached[1], cached[2]
        entries = []
        for stim_name, tgt_map in self._stim_conns.items():
            conn = tgt_map.get(target)
            if conn is not None:
                entries.append((stim_name, conn))
        by_name = {name: conn for name, conn in entries}
        self._stim_target_cache[target] = (self._stim_conn_version, entries, by_name)
        return entries, by_name

    def _stim_capacity(self, needed: int, n: int, cur_cap: int) -> int:
        """Physical capacity for a stim vector that must hold *needed* slots.

        Once the area is materialized past ``_dense_stim_threshold`` it is
        clearly not sparse, so allocate straight to ``n`` and never reallocate
        again. Areas that stay below the threshold keep the doubling growth.
        Capacity never affects sampled values -- only how much room exists.
        """
        thr = self._dense_stim_threshold
        if thr is not None and n > 0 and needed >= thr * n:
            return n
        cap = max(needed, cur_cap * 2, 16)
        return min(cap, n) if n > 0 else cap

    def _grow_stim_vector(self, conn, n: int, old: int, new_len: int, fill) -> None:
        """Extend one stim->area vector to *new_len*, writing *fill* in the tail.

        ``conn.weights`` is kept as a view of exactly ``new_len`` elements over
        an over-allocated buffer, so every reader still sees a vector whose
        length is the area's ever-fired count -- identical to the ``concatenate``
        the legacy path did, minus the O(w) copy on every step.
        """
        xp = self._xp
        cur = conn.weights
        buf = getattr(conn, "_cap_buf", None)
        # getattr(cur, "base", None) is not buf catches a weights array that was
        # replaced wholesale (normalize_weights, unpickling, clone) and so is no
        # longer backed by our buffer.
        if buf is None or getattr(cur, "base", None) is not buf or new_len > buf.shape[0]:
            cap = self._stim_capacity(
                new_len, n, 0 if buf is None else int(buf.shape[0]),
            )
            buf = xp.zeros(cap, dtype=xp.float32)
            if old > 0:
                buf[:old] = cur[:old]
            conn._cap_buf = buf
        if fill is None:
            buf[old:new_len] = 0.0
        else:
            buf[old:new_len] = self._to_xp(fill)
        conn.weights = buf[:new_len]

    def _expand_stim_vectors_legacy(self, target, firing_stimuli, new_w) -> None:
        """Original per-step ``concatenate`` growth. Kept as the A/B reference.

        ``firing_stimuli`` is the stimuli FIRING on this step -- NOT the ones
        connected to the area. Membership means "leave the new tail at ZERO,
        the caller will write it from each new winner's own afferent split";
        every other stimulus takes the background ``binomial(size, p)`` draw.
        Passing the connected set instead leaves every fiber silent, and a
        silent stimulus fiber has exactly the right shape -- see
        `materialize_area`, which had that bug.
        """
        xp = self._xp
        # dict.fromkeys, not set(): the loop below consumes ``self._rng`` once
        # per stimulus, so ITERATION ORDER DECIDES WHICH SLICE OF THE SEEDED
        # STREAM EACH STIMULUS GETS. These are str keys, and set-of-str order
        # varies with PYTHONHASHSEED, so the same seed produced different
        # stimulus weights in every process (same names, same shapes, different
        # values). Insertion order here is deterministic.
        stim_to_extend = dict.fromkeys(firing_stimuli)
        for stim_name, tgt_map in self._stim_conns.items():
            conn = tgt_map.get(target)
            if conn is not None and conn.sparse and len(conn.weights) < new_w:
                stim_to_extend[stim_name] = None
        for stim_name in stim_to_extend:
            conn = self._stim_conns[stim_name][target]
            if conn.sparse:
                old = len(conn.weights)
                if new_w > old:
                    add_len = new_w - old
                    if stim_name not in firing_stimuli:
                        stim_size = self._stimuli[stim_name].size
                        add = self._to_xp(self._rng.binomial(
                            stim_size, self._p_for(stim_name, target),
                            size=add_len).astype(np.float32))
                    else:
                        add = xp.zeros(add_len, dtype=xp.float32)
                    conn.weights = xp.concatenate([conn.weights, add])

    def _expand_stim_vectors_fast(self, target, tgt, firing_stimuli, new_w) -> None:
        """Amortized-capacity growth with batched background sampling.

        ``firing_stimuli`` means what it does in `_expand_stim_vectors_legacy`:
        stimuli FIRING now, whose tail is left at zero for the caller to fill.
        Not the connected set.

        Bit-identical to ``_expand_stim_vectors_legacy``:

        * the same ordered ``dict`` is built from the same insertion sequence,
          so it is iterated in the same order. This was a ``set`` and the claim
          was only true within one process: str hashing is randomized per
          process, so the RNG slice each stimulus received changed from run to
          run. Both paths must keep using ``dict.fromkeys`` or they diverge
          from each other as well as from themselves;
        * ``self._rng`` is consumed by the same stimuli in the same order;
        * consecutive draws are merged into one call only when the scalar
          ``(stim_size, add_len)`` match, and ``Generator.binomial`` with
          scalar parameters fills element-by-element, so
          ``binomial(s, p, a) ++ binomial(s, p, a) == binomial(s, p, 2a)``;
        * non-firing stimuli that draw nothing never split a run because they
          consume no randomness.

        Only the allocation changes: capacity is over-allocated (doubling, or
        straight to ``n`` past ``_dense_stim_threshold``) instead of a fresh
        ``concatenate`` per stimulus per step.
        """
        entries, by_name = self._stim_conns_for(target)
        stim_to_extend = dict.fromkeys(firing_stimuli)
        for stim_name, conn in entries:
            if conn.sparse and len(conn.weights) < new_w:
                stim_to_extend[stim_name] = None

        # Pass 1 -- resolve, in insertion order, what each stimulus needs.
        plan = []  # (conn, old, add_len, stim_size or None when no rng draw)
        for stim_name in stim_to_extend:
            conn = by_name.get(stim_name)
            if conn is None:
                conn = self._stim_conns[stim_name][target]
            if not conn.sparse:
                continue
            old = len(conn.weights)
            if new_w <= old:
                continue
            size = (None if stim_name in firing_stimuli
                    else self._stimuli[stim_name].size)
            plan.append((conn, old, new_w - old, size))
        if not plan:
            return

        # Pass 2 -- batch maximal runs of identical (stim_size, add_len) draws.
        draws = {}
        rng_idx = [i for i, item in enumerate(plan) if item[3] is not None]
        i = 0
        while i < len(rng_idx):
            size, add_len = plan[rng_idx[i]][3], plan[rng_idx[i]][2]
            j = i + 1
            while (j < len(rng_idx)
                   and plan[rng_idx[j]][3] == size
                   and plan[rng_idx[j]][2] == add_len):
                j += 1
            count = j - i
            if count == 1:
                draws[rng_idx[i]] = self._rng.binomial(
                    size, self.p, size=add_len).astype(np.float32)
            else:
                block = self._rng.binomial(
                    size, self.p, size=add_len * count).astype(np.float32)
                for t in range(count):
                    draws[rng_idx[i + t]] = block[t * add_len:(t + 1) * add_len]
            i = j

        # Pass 3 -- write.
        n = tgt.n
        for idx, (conn, old, _add_len, size) in enumerate(plan):
            self._grow_stim_vector(
                conn, n, old, new_w, draws[idx] if size is not None else None,
            )

    def _write_area_expansion_edges(
        self,
        src_name: str,
        target: str,
        conn,
        inputs_names: list,
        new_indices: list,
        splits_per_new: list,
        prior_w: int,
        phys_cols: int,
    ) -> None:
        """Write Hebbian edge allocations without matrix reallocation."""
        src = self._areas[src_name]
        from_index = inputs_names.index(src_name)
        local_rng = np.random.default_rng(self._rng.integers(0, 2**32))
        src_winners_cpu = np.asarray(
            to_cpu(src.winners) if hasattr(src.winners, 'get') else src.winners
        )
        for idx, win in enumerate(new_indices):
            alloc = int(splits_per_new[idx][from_index]) if idx < len(splits_per_new) else 0
            if alloc <= 0 or src.w == 0:
                continue
            sample_size = min(alloc, len(src.winners))
            if sample_size <= 0:
                continue
            chosen = local_rng.choice(src_winners_cpu, size=sample_size, replace=False)
            col_idx = self._expansion_col(int(win), prior_w)
            if 0 <= col_idx < phys_cols:
                conn.weights[chosen, col_idx] = 1.0
                # See the note at the other sampling site: this can write into
                # an already-counted (row, column) region.
                self.mark_column_dirty(conn, col_idx)

    def _expansion_col(self, win: int, prior_w: int) -> int:
        """Column to write a first-time winner's sampled afferents into.

        This SHOULD simply be ``win``: ``win`` is the new neuron's compact
        index, and the allocation drawn from ``compute_input_splits`` describes
        that neuron's own incoming synapses.  The legacy expression
        ``win - prior_w`` instead writes them into columns ``0, 1, 2, ...`` --
        the neurons materialized in the very first projection.  It is only
        correct on the first projection into an area, where ``prior_w == 0``.

        MEASURED CONSEQUENCE.  From the second projection onward every recruit
        dumps its afferents onto the oldest columns, which are exactly the
        neurons most likely to be in the surviving assembly.  Assembly-internal
        connection density inflates to 0.189 against p=0.05 (the dense/explicit
        engine and the reference implementation both give ~0.09), and the
        stored assembly ends up receiving 3.5x the recurrent drive of the rest
        of the area from an UNRELATED assembly (dense engine and reference:
        ~1.0x).  That turns every stored assembly into a hair-trigger attractor
        which captures any independent stimulus, and it is the dominant reason
        self-recurrence collapsed in this engine.  Corrected, independent
        stimuli hold chance overlap at 15 rounds (see Brain.project_rounds).

        WHY IT IS GATED ON ``norm_init`` RATHER THAN JUST FIXED.  The write
        affects EVERY sparse projection, not only recurrent ones, so
        correcting it unconditionally would shift every existing result in the
        repository.  It is therefore scoped to the opt-in norm_init path.  This
        is a defect worth fixing globally on its own, with its own regression
        sweep -- it is not specific to norm_init.
        """
        return win if self.norm_init else win - prior_w

    # -- State accessors ----------------------------------------------------
