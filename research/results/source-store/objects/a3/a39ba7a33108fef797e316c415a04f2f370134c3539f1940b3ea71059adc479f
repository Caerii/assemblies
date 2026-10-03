"""Does word-learning CAPACITY obey the anchor law?  PREREG_word_capacity.md.

The learner is `unaligned_scenes.Aligner` exactly as it passed U1-U3; only
n, k, the phon stimulus size and the vocabulary size V change between cells.
V* is where per-type alignment (against the whole inventory, chance 1/V)
crosses 0.90, read by the shared `ceiling_from_curve` standard -- a curve,
not a grid point, and CENSORED when it never crosses.

    python -m research.experiments.word_capacity_run --tag UNIQUE --smoke \
        --seeds 42 1 2

The shared runner requires a unique tag, records the complete protocol and
alignment semantics, and refuses overwrites.  `--smoke` checks the API on a
tiny grid; its numbers are VOID.  See PREREG_word_capacity.md for the registered
twenty-seed invocation.
"""
from __future__ import annotations

import os
import math
import random
import sys
import time
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)

from neural_assemblies.diagnostics import ensemble_from_values           # noqa: E402
from _substrate import ceiling_from_curve                               # noqa: E402
import unaligned_scenes as U                                            # noqa: E402
from research.experiments.word_capacity_protocol import REGISTERED_PROTOCOL  # noqa: E402


# ---------------------------------------------------------------------------
# synthetic grounded experience
# ---------------------------------------------------------------------------

def corpus(V, seed, *, protocol=REGISTERED_PROTOCOL):
    """Build the synthetic experience from the explicit protocol value.

    V referents (IDENT_i, CAT_{i mod category_count}); scenes contain
    referents_per_scene referents; the sentence names them. Returns
    (experience, targets, words, features)
    in the learner's own shapes."""
    rng = random.Random(seed + protocol.corpus_seed_offset)
    bundles = [tuple(sorted((f"IDENT_{i}", f"CAT_{i % protocol.category_count}")))
               for i in range(V)]
    words = [f"w{i}" for i in range(V)]
    n_scenes = max(
        V * protocol.exposures_per_referent // protocol.referents_per_scene,
        protocol.referents_per_scene,
    )
    exp = []
    for _ in range(n_scenes):
        idx = rng.sample(range(V), protocol.referents_per_scene)
        exp.append(([words[i] for i in idx], [bundles[i] for i in idx]))
    targets = {words[i]: bundles[i] for i in range(V)}
    features = sorted({f for b in bundles for f in b})
    return exp, targets, words, features


# ---------------------------------------------------------------------------
# one cell
# ---------------------------------------------------------------------------

def type_accuracy(seed, V, n, k, stim_size, *, protocol=REGISTERED_PROTOCOL):
    exp, targets, words, features = corpus(V, seed, protocol=protocol)
    exposures = Counter(w for ws, _b in exp for w in ws)
    # FEAT is FIXED across cells (registration: n=1000, k=50); only LEX and
    # the phon anchor vary. Letting FEAT follow n made the n=4000 cell read
    # WORSE than n=1000 at every exposure level -- more FEAT columns
    # competing at readout -- which is a readout floor, not capacity.
    al = U.Aligner(seed, words, features, scaling=True,
                   n=n, k=k, stim_size=stim_size,
                   feat_n=protocol.feature_area[0], feat_k=protocol.feature_area[1],
                   p=protocol.connection_probability, beta=protocol.plasticity)
    al.train(exp, random.Random(seed + protocol.training_seed_offset))
    inventory = sorted({b for _w, bs in exp for b in bs})
    inv_asm = {b: al.bundle_assembly(b) for b in inventory}
    hits = 0
    scored = 0
    for w in words:
        if exposures[w] < protocol.minimum_exposures:
            continue
        rec = al.reconstruct(w)
        best = max(inventory, key=lambda b: rec.overlap(inv_asm[b]))
        hits += int(best == targets[w])
        scored += 1
    return hits / max(scored, 1), scored


def type_accuracy_hashed(
    seeds, V, n, k, stim_size, track_pinned=False, *,
    protocol=REGISTERED_PROTOCOL, aligner_semantics=None,
):
    """All seeds as ONE batch of brains on the hashed substrate.

    The corpus is shared across the brains (seeded by the first seed): the
    brains differ by connectome, which is what a seed varies in every other
    hashed study. The numpy path draws a corpus per seed; that is the one
    protocol difference between the engines here, and it is a nuisance
    variable, not a treatment. Returns per-brain accuracies, and the
    pinned-winner trace when asked (DESIGN_hashed_aligner.md).
    """
    if protocol.corpus_seed_scope != "shared-batch":
        raise ValueError("hashed alignment requires a shared-batch corpus protocol")
    import torch
    from neural_assemblies.core.torch_engine._hashed_aligner import HashedAligner
    exp, targets, words, features = corpus(V, seeds[0], protocol=protocol)
    exposures = Counter(w for ws, _b in exp for w in ws)
    # Unclipped, max-relative pricing, anchors at gain 1/p -- the exact
    # regime the parity gate verified (DESIGN_hashed_aligner.md).
    al = HashedAligner(seeds, words, features, n=n, k=k,
                       feat_n=protocol.feature_area[0],
                       feat_k=protocol.feature_area[1], stim_size=stim_size,
                       p=protocol.connection_probability, beta=protocol.plasticity,
                       rounds_word=protocol.rounds_per_pair,
                       track_pinned=track_pinned,
                       aligner_semantics=aligner_semantics)
    al.train(exp, random.Random(seeds[0] + protocol.training_seed_offset))
    inventory = sorted({b for _w, bs in exp for b in bs})
    scored = [w for w in words if exposures[w] >= protocol.minimum_exposures]
    table = al.overlap_table(scored, inventory)            # [V, I, B]
    best = table.argmax(dim=1)                             # [V, B]
    tgt = torch.tensor([inventory.index(targets[w]) for w in scored],
                       device=best.device).view(-1, 1)
    acc = (best == tgt).float().mean(dim=0).cpu().numpy()  # per brain
    return acc, len(scored), al.pinned


def _bytes_per_brain(V, n, feat_n, *, protocol=REGISTERED_PROTOCOL):
    """The aligner's per-brain tensors at vocabulary V: per-bundle drive and
    two jitters [I, feat_n], the prepare-time constants [F + 1, feat_n] and
    the per-word / per-feature stimulus bases, all float32."""
    I, F = V, V + protocol.category_count
    return 4 * (3 * I * feat_n + (F + 1) * feat_n + V * n + F * feat_n)


def run_cell_scheduled(
    name, seeds, vs, feat=None, *, protocol=REGISTERED_PROTOCOL,
    aligner_semantics=None,
):
    """Every (V, seed) task of a cell in as few launches as the memory budget
    allows (layer 1). Each brain has its own corpus (seeded by its seed),
    vocabulary, bundle inventory and schedule; only the area shape is
    shared. Returns the same curve record as `run_cell` so `judge` cannot
    tell the difference."""
    if protocol.corpus_seed_scope != "per-brain":
        raise ValueError("scheduled alignment requires a per-brain corpus protocol")
    cell = protocol.cell(name)
    n = cell.n
    if feat is not None and tuple(feat) != protocol.feature_area:
        raise ValueError("feature area must come from the selected protocol")
    feat = protocol.feature_area
    feat_n, feat_k = feat
    tasks = [(V, seed) for V in vs for seed in seeds]
    per_task = {
        task: _bytes_per_brain(task[0], n, feat_n, protocol=protocol)
        for task in tasks
    }
    curve = {V: [] for V in vs}
    for chunk, _bytes in _chunks(tasks, per_task, vs, protocol):
        for V, acc in _run_chunk(name, chunk, feat, protocol, aligner_semantics):
            curve[V].append(acc)
    _print_curve(curve)
    return curve


def _chunks(tasks, per_task, vs, protocol):
    """The launches of one cell, in order, with their estimated bytes. A
    chunk holds ONE vocabulary size: every brain's bundle tensors are padded
    to the chunk's largest V, so mixing sizes pays the largest for all."""
    out = []
    for V in vs:
        chunk, used = [], 0
        for t in (t for t in tasks if t[0] == V):
            if chunk and used + per_task[t] > protocol.launch_budget_bytes:
                out.append((chunk, used))
                chunk, used = [], 0
            chunk.append(t)
            used += per_task[t]
        if chunk:
            out.append((chunk, used))
    return out


def _print_curve(curve, label=""):
    for V in curve:
        print(f"      {label}V={V:4d}: type-acc {' '.join(f'{a:.3f}' for a in curve[V])}"
              f"  (chance {1 / V:.3f})", flush=True)


#: worker threads (one CUDA stream each)
CONCURRENT_WORKERS = 8
#: a chunk's MEASURED device peak over `_bytes_per_brain`'s estimate, with a
#: margin: 0.76 to 1.00 at V = 256 and 1024 on cells A and C since `prepare`
#: builds its anchors one at a time (1.58 to 2.23 before)
PEAK_FACTOR = 1.15
#: device memory left free beside the chunks in flight
RESERVE_BYTES = 1 << 30


def run_cells_concurrent(jobs, *, workers=CONCURRENT_WORKERS, budget_bytes=None):
    """Every chunk of several scheduled cells, in flight together.

    `jobs` maps a key to (name, seeds, protocol, aligner_semantics); returns
    {key: curve}, each curve exactly what `run_cell_scheduled` returns.

    THE TRAINING KERNEL RUNS ONE WARP PER BRAIN, so a 20-brain launch holds
    20 warps of the ~3,000 the card can keep resident, and its time is set by
    its step count alone (DESIGN_memory_throughput.md). Chunks of different
    cells, rates and vocabulary sizes are independent, so they run on
    separate CUDA streams (one per worker thread), as many at a time as the
    byte budget allows; the longest chunks start first. Each chunk is built
    and trained exactly as in the serial loop -- same brains, same padding,
    same schedule -- so every accuracy is the serial one.

    THE BUDGET IS THE CARD'S REAL FREE MEMORY. Admitting chunks against an
    estimate that was half their real peak overcommitted the card, and on
    Windows the driver then backs CUDA allocations with system memory instead
    of failing: every chunk ran ~60x slower. A chunk is admitted at
    PEAK_FACTOR times its estimate against the memory free at the start less
    RESERVE_BYTES, and the allocator is capped there, so an overcommit raises
    instead of paging."""
    import threading
    from concurrent.futures import ThreadPoolExecutor
    import torch
    from neural_assemblies.core.torch_engine._scheduled_aligner import ScheduledAligner

    plan = []                                   # (key, index, chunk, bytes, name, ...)
    for key, (name, seeds, protocol, semantics) in jobs.items():
        if protocol.corpus_seed_scope != "per-brain":
            raise ValueError("scheduled alignment requires a per-brain corpus protocol")
        n, vs = protocol.cell(name).n, protocol.vocabulary_sizes
        tasks = [(V, seed) for V in vs for seed in seeds]
        per_task = {t: _bytes_per_brain(t[0], n, protocol.feature_area[0], protocol=protocol)
                    for t in tasks}
        for i, (chunk, used) in enumerate(_chunks(tasks, per_task, vs, protocol)):
            plan.append((key, i, chunk, int(PEAK_FACTOR * used), name, protocol, semantics))
    # longest first: the largest vocabulary is the longest schedule
    plan.sort(key=lambda c: -max(V for V, _seed in c[2]))
    gate = threading.Condition()
    in_flight = [0]
    local = threading.local()
    results = {}

    def run(item):
        key, i, chunk, need, name, protocol, semantics = item
        with gate:
            gate.wait_for(lambda: in_flight[0] + need <= budget_bytes or in_flight[0] == 0)
            in_flight[0] += need
        try:
            stream = getattr(local, "stream", None)
            if stream is None:
                stream = local.stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                results[(key, i)] = _run_chunk(name, chunk, protocol.feature_area,
                                               protocol, semantics, release_cache=False)
            stream.synchronize()
        finally:
            with gate:
                in_flight[0] -= need
                gate.notify_all()

    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    budget = free - RESERVE_BYTES
    if budget_bytes is not None:
        budget = min(budget, budget_bytes)
    if budget <= 0:
        raise RuntimeError("no device memory free for the lexicon chunks")
    budget_bytes = budget
    torch.cuda.set_per_process_memory_fraction(
        min(1.0, (torch.cuda.memory_reserved() + budget + RESERVE_BYTES // 2) / total))
    previous = ScheduledAligner.release_cache
    ScheduledAligner.release_cache = False
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for future in [pool.submit(run, item) for item in plan]:
                future.result()
    finally:
        ScheduledAligner.release_cache = previous
        torch.cuda.set_per_process_memory_fraction(1.0)
    torch.cuda.empty_cache()
    curves = {}
    for key, (name, seeds, protocol, _semantics) in jobs.items():
        curve = {V: [] for V in protocol.vocabulary_sizes}
        i = 0
        while (key, i) in results:
            for V, acc in results[(key, i)]:
                curve[V].append(acc)
            i += 1
        curves[key] = curve
    return curves


_TASKS: dict = {}


def _task(V, seed, protocol):
    """One brain's corpus, vocabulary, inventory and schedule. They depend on
    (V, seed) and the protocol's corpus fields only -- not on the cell, the
    connection probability or the plasticity -- so a study builds each once,
    not once per rate and cell. Read-only once built."""
    key = (V, seed, protocol.corpus_seed_offset, protocol.category_count,
           protocol.exposures_per_referent, protocol.referents_per_scene,
           protocol.training_seed_offset, protocol.corpus_seed_scope)
    hit = _TASKS.get(key)
    if hit is not None:
        return hit
    from neural_assemblies.core.torch_engine._scheduled_aligner import schedule_of
    exp, targets, words, features = corpus(V, seed, protocol=protocol)
    exposures = Counter(w for ws, _b in exp for w in ws)
    inventory = sorted({b for _w, bs in exp for b in bs})
    wi = {w: i for i, w in enumerate(words)}
    bi = {b: j for j, b in enumerate(inventory)}
    order = list(range(len(exp)))
    random.Random(seed + protocol.training_seed_offset).shuffle(order)
    task = dict(V=V, seed=seed, words=words, features=features, inventory=inventory,
                targets=targets, exposures=exposures, sched=schedule_of(exp, wi, bi, order))
    if len(_TASKS) > 4096:
        _TASKS.clear()
    _TASKS[key] = task
    return task


def _run_chunk(name, tasks, feat, protocol, aligner_semantics=None, *, release_cache=True):
    import torch
    from neural_assemblies.core.torch_engine._scheduled_aligner import (
        ScheduledAligner, pad_schedules, schedule_of)
    cell = protocol.cell(name)
    n, k, stim = cell.n, cell.k, cell.stimulus_size
    feat_n, feat_k = feat
    per = [_task(V, seed, protocol) for V, seed in tasks]
    Vmax = max(len(t["words"]) for t in per)
    Fmax = max(len(t["features"]) for t in per)
    Imax = max(len(t["inventory"]) for t in per)
    Fper = max(len(b) for t in per for b in t["inventory"])
    B = len(per)
    feats = torch.full((B, Imax, Fper), -1, dtype=torch.int64)
    tgt = torch.full((B, Vmax), -1, dtype=torch.int64)
    expo = torch.zeros(B, Vmax, dtype=torch.int64)
    nb = torch.zeros(B, dtype=torch.int64)
    for b, t in enumerate(per):
        fi = {f: i for i, f in enumerate(t["features"])}
        bi = {bb: j for j, bb in enumerate(t["inventory"])}
        for j, bb in enumerate(t["inventory"]):
            for sl, f in enumerate(bb):
                feats[b, j, sl] = fi[f]
        for i, w in enumerate(t["words"]):
            # a referent that never entered a scene has no bundle; its word
            # is below MIN_EXPOSURES and unscored (target -1)
            tgt[b, i] = bi.get(t["targets"][w], -1)
            expo[b, i] = t["exposures"][w]
        nb[b] = len(t["inventory"])
    W, Bd = pad_schedules([t["sched"] for t in per])
    t0 = time.perf_counter()
    # word/feature INDEX i means brain b's own word i: every brain seeds its
    # phon fibers by (seed_b, "phon_i"), so brains share nothing but shape
    al = ScheduledAligner([
        t["seed"] * protocol.connectome_seed_stride + t["V"] for t in per
    ], n=n, k=k,
                          feat_n=feat_n, feat_k=feat_k, n_words=Vmax,
                          n_features=Fmax, stim_size=stim,
                          p=protocol.connection_probability,
                          beta=protocol.plasticity,
                          rounds_word=protocol.rounds_per_pair,
                          aligner_semantics=aligner_semantics)
    al.prepare(feats)
    al.train(W, Bd, device_loop=True)          # layer 3: one launch per chunk
    acc, scored = al.type_accuracy(tgt, nb, expo, protocol.minimum_exposures)
    acc = acc.cpu().numpy()
    print(f"    {name} n={n} k={k} s={stim} FEAT {feat_n}x{feat_k}: {B} brains "
          f"(V x seed) in one launch, {W.shape[1]} steps  "
          f"[{time.perf_counter() - t0:.0f}s]", flush=True)
    del al
    if release_cache:
        torch.cuda.empty_cache()
    return [(t["V"], float(acc[b])) for b, t in enumerate(per)]


def run_cell(name, seeds, vs, engine="numpy", track_pinned=False,
             feat=None, *, protocol=REGISTERED_PROTOCOL, aligner_semantics=None):
    if engine not in {"numpy", "hashed", "scheduled"}:
        raise ValueError(f"unknown word-capacity engine: {engine}")
    expected_scope = "shared-batch" if engine == "hashed" else "per-brain"
    if protocol.corpus_seed_scope != expected_scope:
        raise ValueError(
            f"{engine} requires a {expected_scope} corpus protocol",
        )
    if tuple(vs) != protocol.vocabulary_sizes:
        raise ValueError("vocabulary sizes must come from the selected protocol")
    if feat is not None and tuple(feat) != protocol.feature_area:
        raise ValueError("feature area must come from the selected protocol")
    feat = protocol.feature_area
    if engine == "scheduled":
        return run_cell_scheduled(
            name, seeds, vs, feat=feat, protocol=protocol,
            aligner_semantics=aligner_semantics,
        )
    cell = protocol.cell(name)
    n, k, s = cell.n, cell.k, cell.stimulus_size
    curve = {}                                      # V -> [acc per seed]
    pinned = {}
    for V in vs:
        accs = []
        if engine == "hashed":
            t0 = time.perf_counter()
            acc, scored, pin = type_accuracy_hashed(seeds, V, n, k, s,
                                                     track_pinned,
                                                     protocol=protocol,
                                                     aligner_semantics=aligner_semantics)
            accs = [float(a) for a in acc]
            if pin:
                pinned[V] = pin
            print(f"    {name} n={n} k={k} s={s} V={V:4d} hashed B={len(seeds)}: "
                  f"type-acc {' '.join(f'{a:.3f}' for a in accs)} "
                  f"(chance {1 / V:.3f}, n={scored})"
                  + (f"  pinned min {min(pin):.3f} mean "
                     f"{sum(pin) / len(pin):.3f}" if pin else "")
                  + f"  [{time.perf_counter() - t0:.0f}s]", flush=True)
        else:
            for seed in seeds:
                t0 = time.perf_counter()
                acc, scored = type_accuracy(
                    seed, V, n, k, s, protocol=protocol,
                )
                accs.append(acc)
                print(f"    {name} n={n} k={k} s={s} V={V:4d} seed {seed:2d}: "
                      f"type-acc {acc:.3f} (chance {1 / V:.3f}, n={scored})  "
                      f"[{time.perf_counter() - t0:.0f}s]", flush=True)
        curve[V] = accs
        # stop early once the curve is clearly below threshold on every seed
        if max(accs) < protocol.threshold - protocol.early_stop_margin:
            break
    return curve


def ceilings(curve, seeds, threshold=None, *, protocol=REGISTERED_PROTOCOL):
    """Per-seed V* by the shared standard; ensemble across seeds."""
    threshold = protocol.threshold if threshold is None else threshold
    sizes = tuple(curve)
    if not sizes or sizes != protocol.vocabulary_sizes[:len(sizes)]:
        raise ValueError("capacity curve must follow a nonempty protocol-grid prefix")
    if len(seeds) != len(set(seeds)):
        raise ValueError("capacity seeds must be unique")
    if any(len(values) != len(seeds) for values in curve.values()):
        raise ValueError("every capacity point must contain one value per seed")
    if any(
        type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1
        for values in curve.values() for value in values
    ):
        raise ValueError("capacity accuracies must be finite probabilities")
    stars, censored = [], 0
    for i, _seed in enumerate(seeds):
        pts = [(V, accs[i]) for V, accs in curve.items()]
        c = ceiling_from_curve(
            pts, threshold=threshold,
            interior_band=protocol.ceiling_interior_band,
        )
        # Censored in EITHER direction: never crossed (high) or never above
        # the threshold at all (low -- the standard returns the smallest V
        # uncensored there, which would read as a value).
        if c.censored or max(a for _v, a in pts) <= threshold:
            censored += 1
        stars.append(float(c.m_star))
    return stars, censored


def capacity_report(results, seeds, threshold=None, *, protocol=REGISTERED_PROTOCOL):
    """Return the registered curve summaries and bars as strict JSON values."""
    cells = {}
    ensembles = {}
    censored = {}
    for name, curve in results.items():
        stars, count = ceilings(curve, seeds, threshold, protocol=protocol)
        cell = protocol.cell(name)
        n, k, stim = cell.n, cell.k, cell.stimulus_size
        ensemble = ensemble_from_values(
            stars, label=f"{name} n={n} k={k} s={stim} V*", keys=seeds,
        )
        ensembles[name], censored[name] = ensemble, count
        cells[name] = {
            "n": n, "k": k, "stimulus_size": stim, "n_over_k": n / k,
            "ceiling": {
                "values": list(ensemble.values), "seed_ids": list(ensemble.keys),
                "mean": ensemble.mean, "ci95_half_width": ensemble.ci,
                "ci95_lo": ensemble.low, "ci95_hi": ensemble.high,
            },
            "censored_seeds": count,
        }

    def available(*names):
        return all(name in ensembles and censored[name] == 0 for name in names)

    bars = {}
    if available("A", "B", "C"):
        a, b, c = (ensembles[name] for name in ("A", "B", "C"))
        step1 = (b.mean - a.mean) > (a.high - a.mean) + (b.mean - b.low)
        step2 = (c.mean - b.mean) > (b.high - b.mean) + (c.mean - c.low)
        bars["W1"] = {"status": "PASS" if step1 and step2 else "FAIL",
                      "a_lt_b_beyond_pooled_ci": step1,
                      "b_lt_c_beyond_pooled_ci": step2}
    else:
        bars["W1"] = {"status": "VOID", "reason": "missing or censored cell"}
    if available("B", "D"):
        ratio = ensembles["D"].mean / ensembles["B"].mean
        status = "PASS" if abs(ratio - 1) <= protocol.w2_pass_tolerance else (
            "FAIL" if abs(ratio - 1) > protocol.w2_fail_tolerance
            else "INCONCLUSIVE"
        )
        bars["W2"] = {"status": status, "d_over_b": ratio}
    else:
        bars["W2"] = {"status": "VOID", "reason": "missing or censored cell"}
    if available("B", "E"):
        ratio = ensembles["E"].mean / ensembles["B"].mean
        status = "PASS" if ratio >= protocol.w3_pass_ratio else (
            "FAIL" if ratio <= protocol.w3_fail_ratio else "INCONCLUSIVE"
        )
        bars["W3"] = {"status": status, "e_over_b": ratio}
    else:
        bars["W3"] = {"status": "VOID", "reason": "missing or censored cell"}
    return {"cells": cells, "bars": bars}


# ---------------------------------------------------------------------------

def ladder(*_args, **_kwargs):
    """Reject the former unrecorded FEAT-ladder execution path."""
    raise RuntimeError(
        "use `python -m research.experiments.word_capacity_ladder_run --tag UNIQUE`",
    )


def judge(results, seeds, *, protocol=REGISTERED_PROTOCOL):
    print("\n=== BARS (PREREG_word_capacity.md) ===")
    report = capacity_report(results, seeds, protocol=protocol)
    for name, cell in report["cells"].items():
        ceiling = cell["ceiling"]
        print(f"  {name}: {ceiling['mean']:.4f} +/- {ceiling['ci95_half_width']:.4f} "
              f"(n={len(seeds)}, {min(ceiling['values']):.4f}.."
              f"{max(ceiling['values']):.4f})   n/k {cell['n_over_k']:.0f}   "
              f"censored seeds {cell['censored_seeds']}/{len(seeds)}")
    print(f"  {report['bars']['W1']}")
    print(f"  {report['bars']['W2']}")
    print(f"  {report['bars']['W3']}")
    return report


def main(argv=None):
    """Route every new invocation through immutable schema-8 evidence."""
    from research.experiments.word_capacity_run import main as run
    return run(argv)


if __name__ == "__main__":
    main()
