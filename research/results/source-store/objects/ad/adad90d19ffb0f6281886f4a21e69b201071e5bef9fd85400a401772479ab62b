#!/usr/bin/env python3
"""Benchmark EmergentParser training throughput with streaming progress.

Usage::

    python research/experiments/benchmark_emergent_training.py
    python research/experiments/benchmark_emergent_training.py --quick
    TRAIN_PROGRESS=0 python research/experiments/benchmark_emergent_training.py
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List

from neural_assemblies.assembly_calculus.emergent.train_progress import (
    finish_progress,
    start_progress,
)
from neural_assemblies.core.engine import list_engines


@dataclass
class BenchResult:
    name: str
    seconds: float
    count: int = 1
    meta: Dict = field(default_factory=dict)

    @property
    def per_op_ms(self) -> float:
        return (self.seconds / max(self.count, 1)) * 1000


def _timeit(fn, *, repeats: int = 1, warmup: int = 0) -> BenchResult:
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    elapsed = time.perf_counter() - t0
    return BenchResult(name=fn.__name__, seconds=elapsed, count=repeats)


def bench_projection_micro(prog, *, engine: str = "auto") -> List[BenchResult]:
    from neural_assemblies.core.brain import Brain
    from neural_assemblies.assembly_calculus.ops import project
    from neural_assemblies.assembly_calculus.emergent.training.perf import resolve_engine

    resolved = resolve_engine(engine)
    results: List[BenchResult] = []
    configs = [
        ("small", 3000, 30, 10),
        ("default", 10000, 100, 10),
        ("large_n", 50000, 100, 10),
    ]
    for label, n, k, rounds in configs:
        with prog.phase(f"setup_{label}", f"n={n} k={k} engine={resolved}"):
            brain = Brain(p=0.05, seed=42, engine=resolved, n_hint=n)
            brain.add_stimulus("s", k)
            brain.add_area("A", n, k, 0.1)
            brain.add_area("B", n, k, 0.1)
            project(brain, "s", "A", rounds=rounds)

        def one_project():
            brain.project({}, {"A": ["B"]})

        with prog.phase(f"bench_{label}", "50 projections"):
            r = _timeit(one_project, repeats=50, warmup=5)
        r.name = f"project_A_to_B_{label}"
        r.meta = {"n": n, "k": k, "rounds_setup": rounds, "engine": resolved}
        results.append(r)
        prog.info(f"{r.name}: {r.per_op_ms:.2f} ms/proj")
    return results


def bench_engines(prog) -> List[BenchResult]:
    from neural_assemblies.core.brain import Brain
    from neural_assemblies.assembly_calculus.ops import project

    results: List[BenchResult] = []
    n, k, rounds = 10000, 100, 8
    for engine in list_engines():
        with prog.phase(f"engine_{engine}"):
            try:
                brain = Brain(p=0.05, seed=42, engine=engine)
                brain.add_stimulus("s", k)
                brain.add_area("A", n, k, 0.1)
                brain.add_area("B", n, k, 0.1)
                project(brain, "s", "A", rounds=rounds)
            except Exception as exc:
                results.append(BenchResult(
                    name=f"engine_{engine}",
                    seconds=0.0,
                    meta={"error": str(exc)},
                ))
                prog.info(f"engine_{engine}: SKIP ({exc})")
                continue

            def one_project():
                brain.project({}, {"A": ["B"]})

            r = _timeit(one_project, repeats=30, warmup=3)
            r.name = f"engine_{engine}"
            r.meta = {"n": n, "k": k}
            results.append(r)
            prog.info(f"{r.name}: {r.per_op_ms:.2f} ms/proj")
    return results


def bench_fast_training(prog, *, n: int, k: int, rounds: int) -> List[BenchResult]:
    """Compare train_for_agent with default vs fast_training budgets."""
    from neural_assemblies.assembly_calculus.emergent import (
        EmergentParser,
        build_vocabulary_preset,
    )

    vocab = build_vocabulary_preset("core")
    results: List[BenchResult] = []

    for label, fast in (("default", False), ("fast", True)):
        with prog.phase(f"train_for_agent_{label}"):
            p = EmergentParser(
                n=n, k=k, seed=42, rounds=rounds,
                vocabulary=vocab, fast_training=fast,
            )
            start_progress(f"train_for_agent_{label}")
            t0 = time.perf_counter()
            p.train_for_agent(include_blocks=False)
            finish_progress()
            elapsed = time.perf_counter() - t0
            r = BenchResult(
                f"train_for_agent_{label}", elapsed,
                meta={
                    "fast_training": fast,
                    "train_rounds": p.rounds,
                    "infer_rounds": p.inference_rounds,
                    "bridge_rounds": p.bridge_rounds,
                    "engine": getattr(p, "engine_name", "numpy_sparse"),
                },
            )
            results.append(r)
            prog.info(
                f"{label}: {elapsed:.1f}s "
                f"(train={p.rounds} infer={p.inference_rounds})",
            )
    return results


def bench_parser_phases(prog, *, n: int, k: int, rounds: int) -> List[BenchResult]:
    from neural_assemblies.assembly_calculus.emergent import (
        EmergentParser,
        build_vocabulary_preset,
    )
    from neural_assemblies.assembly_calculus.emergent.curriculum import CurriculumTrainer

    vocab = build_vocabulary_preset("core")
    results: List[BenchResult] = []

    with prog.phase("init_parser", f"n={n} k={k} vocab={len(vocab)}"):
        t0 = time.perf_counter()
        p = EmergentParser(n=n, k=k, seed=42, rounds=rounds, vocabulary=vocab)
        elapsed = time.perf_counter() - t0
        results.append(BenchResult(
            "init_parser_44_areas", elapsed,
            meta={
                "areas": len(p.brain.areas),
                "vocab": len(vocab),
                "engine": getattr(p, "engine_name", "numpy_sparse"),
            },
        ))
        prog.info(f"engine={getattr(p, 'engine_name', 'numpy_sparse')}")

    with prog.section("train_for_agent"):
        p = EmergentParser(n=n, k=k, seed=42, rounds=rounds, vocabulary=vocab)
        start_progress("train_for_agent")
        t0 = time.perf_counter()
        p.train_for_agent(include_blocks=True)
        finish_progress()
        elapsed = time.perf_counter() - t0
        results.append(BenchResult("train_for_agent_core", elapsed))
        prog.info(f"train_for_agent_core: {elapsed:.1f}s")

    vocab_med = build_vocabulary_preset("medium")
    with prog.phase("curriculum_FIRST_WORDS", f"vocab={len(vocab_med)}"):
        p = EmergentParser(n=n, k=k, seed=42, rounds=rounds, vocabulary=vocab_med)
        t0 = time.perf_counter()
        CurriculumTrainer(p).train_stage("FIRST_WORDS")
        elapsed = time.perf_counter() - t0
        results.append(BenchResult(
            "curriculum_FIRST_WORDS_medium", elapsed,
            meta={"vocab": len(vocab_med)},
        ))
        prog.info(f"curriculum_FIRST_WORDS: {elapsed:.1f}s")

    with prog.phase("parse_incremental_x20"):
        p = EmergentParser(n=n, k=k, seed=42, rounds=rounds, vocabulary=vocab)
        words = ["the", "dog", "chases", "the", "cat"]
        t0 = time.perf_counter()
        for _ in range(20):
            p.parse_incremental(words)
        elapsed = time.perf_counter() - t0
        results.append(BenchResult("parse_incremental_5w_x20", elapsed, count=20))
        prog.info(f"parse_incremental: {elapsed / 20 * 1000:.1f} ms/parse")

    return results


def estimate_scaling(prog) -> Dict:
    from neural_assemblies.assembly_calculus.emergent import EmergentParser
    from neural_assemblies.assembly_calculus.emergent.curriculum.data import (
        create_instruction_sentences,
        create_training_sentences,
    )
    from neural_assemblies.assembly_calculus.emergent.curriculum.dialogue import (
        get_dialogue_curriculum,
    )

    n_sents = (
        len(create_training_sentences())
        + len(create_instruction_sentences())
        + len(get_dialogue_curriculum())
    )
    configs = [(3000, 30, 4), (5000, 50, 6), (10000, 100, 10)]
    rows = []
    for n, k, rounds in configs:
        with prog.phase(f"scale_n{n}_k{k}"):
            start_progress(f"train_for_agent n={n}")
            t0 = time.perf_counter()
            p = EmergentParser(n=n, k=k, seed=42, rounds=rounds)
            p.train_for_agent()
            finish_progress()
            elapsed = time.perf_counter() - t0
        rows.append({
            "n": n, "k": k, "rounds": rounds,
            "train_for_agent_s": round(elapsed, 2),
            "sentences": n_sents,
            "sents_per_s": round(n_sents / elapsed, 2),
        })
        prog.info(
            f"n={n} k={k}: {elapsed:.1f}s ({n_sents / elapsed:.2f} sents/s)",
        )
    return {"sentence_count": n_sents, "configs": rows}


def profile_train_for_agent(prog) -> str:
    from neural_assemblies.assembly_calculus.emergent import EmergentParser

    def run():
        p = EmergentParser(n=10000, k=100, seed=42, rounds=8)
        p.train_for_agent()

    with prog.phase("cprofile_train_for_agent"):
        pr = cProfile.Profile()
        pr.enable()
        run()
        pr.disable()
    buf = io.StringIO()
    ps = pstats.Stats(pr, stream=buf).sort_stats("cumulative")
    ps.print_stats(25)
    return buf.getvalue()


def print_summary(results: List[BenchResult], stream=sys.stdout) -> None:
    print("\n=== SUMMARY ===", file=stream)
    for r in results:
        if r.meta.get("error"):
            print(f"  {r.name}: ERROR {r.meta['error']}", file=stream)
        elif r.count > 1:
            print(
                f"  {r.name}: {r.seconds:.2f}s "
                f"({r.per_op_ms:.2f} ms/op) meta={r.meta}",
                file=stream,
            )
        else:
            print(f"  {r.name}: {r.seconds:.2f}s meta={r.meta}", file=stream)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--quick", action="store_true",
        help="Skip scaling sweep and cProfile (faster)",
    )
    ap.add_argument("-n", type=int, default=10000)
    ap.add_argument("-k", type=int, default=100)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--engine", default="auto", help="Compute engine (auto, numpy_sparse, torch_sparse)")
    args = ap.parse_args()

    prog = start_progress("benchmark_emergent")
    all_results: List[BenchResult] = []

    prog.info(f"engines: {list_engines()}")

    with prog.section("micro_projections"):
        all_results.extend(bench_projection_micro(prog, engine=args.engine))

    with prog.section("engine_comparison"):
        all_results.extend(bench_engines(prog))

    with prog.section("parser_phases"):
        all_results.extend(bench_parser_phases(
            prog, n=args.n, k=args.k, rounds=args.rounds,
        ))

    with prog.section("fast_vs_default"):
        all_results.extend(bench_fast_training(
            prog, n=args.n, k=args.k, rounds=args.rounds,
        ))

    if not args.quick:
        with prog.section("scaling_sweep"):
            est = estimate_scaling(prog)
            prog.info(f"corpus sentences: {est['sentence_count']}")

        with prog.section("cprofile"):
            profile_out = profile_train_for_agent(prog)
            print("\n=== cProfile top 25 ===", file=sys.stderr)
            print(profile_out, file=sys.stderr)

    finish_progress()
    print_summary(all_results)


if __name__ == "__main__":
    main()
