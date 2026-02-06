#!/usr/bin/env python
"""
NEMO Performance Benchmark
==========================

Version: 1.0.0
Date: 2025-11-30

Detailed performance profiling of the NEMO emergent language system.
Identifies bottlenecks at the kernel level.

Run:
    cd src
    uv run python -m nemo.language.emergent.tests.benchmark
"""

import time
import cupy as cp
import torch
from typing import Dict, List
from collections import defaultdict
from contextlib import contextmanager

# Ensure CUDA is synced for accurate timing
def sync():
    cp.cuda.Stream.null.synchronize()


@contextmanager
def timer(name: str, results: Dict[str, List[float]]):
    """Context manager for timing operations."""
    sync()
    start = time.perf_counter()
    yield
    sync()
    elapsed = (time.perf_counter() - start) * 1000  # ms
    results[name].append(elapsed)


def print_timing_results(results: Dict[str, List[float]], title: str):
    """Print formatted timing results."""
    print(f"\n{'='*70}")
    print(title)
    print('='*70)
    print(f"{'Operation':<40} {'Total (ms)':>10} {'Calls':>8} {'Avg (ms)':>10}")
    print('-'*70)
    
    total_time = 0
    sorted_results = sorted(results.items(), key=lambda x: sum(x[1]), reverse=True)
    
    for name, times in sorted_results:
        total = sum(times)
        count = len(times)
        avg = total / count if count > 0 else 0
        total_time += total
        print(f"{name:<40} {total:>10.2f} {count:>8} {avg:>10.4f}")
    
    print('-'*70)
    print(f"{'TOTAL':<40} {total_time:>10.2f}")
    print('='*70)


def benchmark_kernel_operations():
    """Benchmark raw kernel operations."""
    print("\n" + "="*70)
    print("KERNEL OPERATION BENCHMARKS")
    print("="*70)
    
    from src.nemo.core.kernel import projection_fp16_kernel, hebbian_kernel
    
    n, k = 10000, 100
    iterations = 500
    
    results = defaultdict(list)
    
    # Setup
    active = cp.random.randint(0, n, k, dtype=cp.uint32).reshape(1, -1)
    result = cp.zeros((1, n), dtype=cp.float16)
    seeds = cp.array([12345], dtype=cp.uint32)
    p = 0.1
    bs = 512
    gx = (n + bs - 1) // bs
    
    # Benchmark projection kernel
    print(f"\nProjection kernel: n={n}, k={k}, {iterations} iterations")
    for i in range(iterations):
        with timer("projection_kernel", results):
            projection_fp16_kernel(
                (gx, 1), (bs,),
                (active, result, cp.uint32(k), cp.uint32(n), cp.uint32(1), seeds, cp.float32(p)),
                shared_mem=k * 4
            )
    
    # Benchmark topk selection
    result_torch = torch.as_tensor(result[0], device='cuda')
    for i in range(iterations):
        with timer("torch_topk", results):
            _, winners_idx = torch.topk(result_torch, k, sorted=False)
    
    # CuPy alternative
    result_cp = result[0].astype(cp.float32)
    for i in range(iterations):
        with timer("cupy_argsort", results):
            winners = cp.argsort(result_cp)[-k:]
    
    # Benchmark Hebbian kernel
    prev_active = cp.random.randint(0, n, k, dtype=cp.uint32)
    new_active = cp.random.randint(0, n, k, dtype=cp.uint32)
    max_learned = 1000000
    l_src = cp.zeros(max_learned, dtype=cp.uint32)
    l_dst = cp.zeros(max_learned, dtype=cp.uint32)
    l_delta = cp.zeros(max_learned, dtype=cp.float32)
    l_num = cp.zeros(1, dtype=cp.uint32)
    
    grid = (k * k + bs - 1) // bs
    
    print(f"\nHebbian kernel: k={k}, {iterations} iterations")
    for i in range(iterations):
        with timer("hebbian_kernel", results):
            hebbian_kernel(
                (grid,), (bs,),
                (l_src, l_dst, l_delta, l_num, prev_active, new_active,
                 cp.uint32(k), cp.float32(0.1), cp.float32(2.0),
                 cp.uint32(max_learned), cp.uint32(12345), cp.float32(p))
            )
    
    print_timing_results(results, "KERNEL BENCHMARKS")
    return results


def benchmark_brain_operations():
    """Benchmark brain-level operations."""
    print("\n" + "="*70)
    print("BRAIN OPERATION BENCHMARKS")
    print("="*70)
    
    from src.nemo.language.emergent.brain import EmergentNemoBrain
    from src.nemo.language.emergent.params import EmergentParams
    from src.nemo.language.emergent.areas import Area
    
    params = EmergentParams()
    brain = EmergentNemoBrain(params, verbose=False)
    
    iterations = 200
    results = defaultdict(list)
    
    # Create test inputs
    test_input = cp.random.randint(0, params.n, params.k, dtype=cp.uint32)
    
    print(f"\nBrain operations: n={params.n}, k={params.k}, {iterations} iterations")
    
    # Benchmark _get_or_create
    for i in range(iterations):
        with timer("get_or_create", results):
            brain._get_or_create(Area.PHON, f"word_{i}")
    
    # Benchmark _project without learning
    for i in range(iterations):
        with timer("project_no_learn", results):
            brain._project(Area.NOUN_CORE, test_input, learn=False)
        brain._clear_area(Area.NOUN_CORE)
    
    # Benchmark _project with learning
    for i in range(iterations):
        with timer("project_with_learn", results):
            brain._project(Area.NOUN_CORE, test_input, learn=True)
        # Don't clear to allow Hebbian learning
    
    # Benchmark store_learned_assembly
    assembly = cp.random.randint(0, params.n, params.k, dtype=cp.uint32)
    for i in range(iterations):
        with timer("store_assembly", results):
            brain.store_learned_assembly(Area.NOUN_CORE, f"word_{i}", assembly)
    
    # Benchmark get_learned_assembly
    for i in range(iterations):
        with timer("get_assembly", results):
            brain.get_learned_assembly(Area.NOUN_CORE, f"word_{i % 100}")
    
    # Benchmark get_assembly_overlap
    a1 = cp.random.randint(0, params.n, params.k, dtype=cp.uint32)
    a2 = cp.random.randint(0, params.n, params.k, dtype=cp.uint32)
    for i in range(iterations):
        with timer("assembly_overlap", results):
            brain.get_assembly_overlap(a1, a2)
    
    print_timing_results(results, "BRAIN OPERATION BENCHMARKS")
    return results


def benchmark_learner_operations():
    """Benchmark learner-level operations."""
    print("\n" + "="*70)
    print("LEARNER OPERATION BENCHMARKS")
    print("="*70)
    
    from src.nemo.language.emergent.learner import EmergentLanguageLearner
    from src.nemo.language.emergent.params import EmergentParams, GroundingContext
    
    params = EmergentParams()
    learner = EmergentLanguageLearner(params, verbose=False)
    
    iterations = 100
    results = defaultdict(list)
    
    # Create test contexts
    visual_ctx = GroundingContext(visual=['object'])
    motor_ctx = GroundingContext(motor=['action'])
    empty_ctx = GroundingContext()
    
    print(f"\nLearner operations: {iterations} iterations")
    
    # Benchmark present_word_with_grounding (visual)
    for i in range(iterations):
        with timer("present_word_visual", results):
            learner.present_word_with_grounding(f"noun_{i}", visual_ctx, position=0, learn=True)
    
    # Benchmark present_word_with_grounding (motor)
    for i in range(iterations):
        with timer("present_word_motor", results):
            learner.present_word_with_grounding(f"verb_{i}", motor_ctx, position=1, learn=True)
    
    # Benchmark present_word_with_grounding (function)
    for i in range(iterations):
        with timer("present_word_function", results):
            learner.present_word_with_grounding(f"det_{i}", empty_ctx, position=0, learn=True)
    
    # Benchmark get_emergent_category
    for i in range(iterations):
        with timer("get_category", results):
            learner.get_emergent_category(f"noun_{i % 50}")
    
    print_timing_results(results, "LEARNER OPERATION BENCHMARKS")
    return results


def benchmark_sentence_processing():
    """Benchmark full sentence processing."""
    print("\n" + "="*70)
    print("SENTENCE PROCESSING BENCHMARKS")
    print("="*70)
    
    from src.nemo.language.emergent.learner import EmergentLanguageLearner
    from src.nemo.language.emergent.params import EmergentParams
    from src.nemo.language.emergent.training_data import create_training_data
    
    params = EmergentParams()
    learner = EmergentLanguageLearner(params, verbose=False)
    training_data = create_training_data()[:50]
    
    results = defaultdict(list)
    
    print(f"\nSentence processing: {len(training_data)} sentences")
    
    # Warm up
    for sentence in training_data[:5]:
        learner.present_grounded_sentence(
            sentence.words, sentence.contexts, sentence.roles, sentence.mood, learn=True
        )
    
    # Benchmark
    for sentence in training_data:
        with timer("present_sentence", results):
            learner.present_grounded_sentence(
                sentence.words, sentence.contexts, sentence.roles, sentence.mood, learn=True
            )
        
        # Break down by sentence length
        length_key = f"sentence_len_{len(sentence.words)}"
        results[length_key].append(results["present_sentence"][-1])
    
    print_timing_results(results, "SENTENCE PROCESSING BENCHMARKS")
    
    # Analyze by sentence length
    print("\nBy Sentence Length:")
    for key in sorted([k for k in results.keys() if k.startswith("sentence_len_")]):
        times = results[key]
        print(f"  {key}: avg={sum(times)/len(times):.2f}ms, count={len(times)}")
    
    return results


def benchmark_parsing():
    """Benchmark parsing operations."""
    print("\n" + "="*70)
    print("PARSING BENCHMARKS")
    print("="*70)
    
    from src.nemo.language.emergent.learner import EmergentLanguageLearner
    from src.nemo.language.emergent.params import EmergentParams
    from src.nemo.language.emergent.training_data import create_training_data
    from src.nemo.language.emergent.parser import SentenceParser
    
    # First train a model
    params = EmergentParams()
    learner = EmergentLanguageLearner(params, verbose=False)
    training_data = create_training_data()
    
    print("Training model...")
    for sentence in training_data:
        learner.present_grounded_sentence(
            sentence.words, sentence.contexts, sentence.roles, sentence.mood, learn=True
        )
    
    # Create parser
    parser = SentenceParser(learner)
    
    # Test sentences
    test_sentences = [
        ["the", "dog", "runs"],
        ["a", "cat", "chases", "the", "bird"],
        ["she", "finds", "the", "food"],
        ["the", "big", "dog", "eats", "food"],
    ]
    
    results = defaultdict(list)
    iterations = 50
    
    print(f"\nParsing: {iterations} iterations per sentence")
    
    for sentence in test_sentences:
        for i in range(iterations):
            with timer(f"parse_{len(sentence)}_words", results):
                parser.parse(sentence)
    
    print_timing_results(results, "PARSING BENCHMARKS")
    return results


def benchmark_memory_usage():
    """Benchmark memory usage."""
    print("\n" + "="*70)
    print("MEMORY USAGE ANALYSIS")
    print("="*70)
    
    import gc
    
    from src.nemo.language.emergent.learner import EmergentLanguageLearner
    from src.nemo.language.emergent.params import EmergentParams
    from src.nemo.language.emergent.training_data import create_training_data
    
    # Clear memory
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    
    mempool = cp.get_default_memory_pool()
    
    def get_memory_mb():
        return mempool.used_bytes() / 1024 / 1024
    
    print(f"Initial GPU memory: {get_memory_mb():.2f} MB")
    
    # Create learner
    params = EmergentParams()
    learner = EmergentLanguageLearner(params, verbose=False)
    print(f"After learner init: {get_memory_mb():.2f} MB")
    
    # Train with increasing data
    training_data = create_training_data()
    
    checkpoints = [10, 50, 100, 200, 349]
    for checkpoint in checkpoints:
        for i, sentence in enumerate(training_data[:checkpoint]):
            if i >= (checkpoints[checkpoints.index(checkpoint) - 1] if checkpoints.index(checkpoint) > 0 else 0):
                learner.present_grounded_sentence(
                    sentence.words, sentence.contexts, sentence.roles, sentence.mood, learn=True
                )
        print(f"After {checkpoint} sentences: {get_memory_mb():.2f} MB")
    
    # Check learned weights
    total_learned = 0
    for area_idx in range(len(learner.brain.l_num)):
        num = int(learner.brain.l_num[area_idx].get()[0])
        total_learned += num
    print(f"\nTotal learned connections: {total_learned:,}")
    
    # Check assemblies
    total_assemblies = 0
    for area in learner.brain.learned_assemblies:
        total_assemblies += len(learner.brain.learned_assemblies[area])
    print(f"Total stored assemblies: {total_assemblies:,}")


def run_all_benchmarks():
    """Run all benchmarks."""
    print("="*70)
    print("NEMO EMERGENT LANGUAGE SYSTEM - FULL BENCHMARK SUITE")
    print("="*70)
    print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    all_results = {}
    
    # Run benchmarks
    all_results['kernels'] = benchmark_kernel_operations()
    all_results['brain'] = benchmark_brain_operations()
    all_results['learner'] = benchmark_learner_operations()
    all_results['sentences'] = benchmark_sentence_processing()
    all_results['parsing'] = benchmark_parsing()
    benchmark_memory_usage()
    
    # Summary
    print("\n" + "="*70)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("="*70)
    
    # Analyze results
    kernel_results = all_results['kernels']
    brain_results = all_results['brain']
    
    proj_time = sum(kernel_results.get('projection_kernel', [0]))
    topk_time = sum(kernel_results.get('torch_topk', [0]))
    hebb_time = sum(kernel_results.get('hebbian_kernel', [0]))
    
    print("\n1. KERNEL ANALYSIS:")
    print(f"   - Projection: {proj_time/500:.3f} ms/call")
    print(f"   - TopK: {topk_time/500:.3f} ms/call")
    print(f"   - Hebbian: {hebb_time/500:.3f} ms/call")
    
    proj_brain = sum(brain_results.get('project_with_learn', [0]))
    print(f"\n2. BRAIN PROJECTION (with learning): {proj_brain/200:.3f} ms/call")
    
    overlap_time = sum(brain_results.get('assembly_overlap', [0]))
    print(f"\n3. ASSEMBLY OVERLAP: {overlap_time/200:.4f} ms/call")
    
    sentence_time = sum(all_results['sentences'].get('present_sentence', [0]))
    n_sentences = len(all_results['sentences'].get('present_sentence', []))
    if n_sentences > 0:
        print(f"\n4. SENTENCE PROCESSING: {sentence_time/n_sentences:.2f} ms/sentence")
    
    print("\n" + "-"*70)
    print("KEY BOTTLENECKS:")
    print("-"*70)
    
    if topk_time > proj_time * 0.5:
        print("  ⚠ TopK selection is significant - consider cupy.argsort")
    
    if hebb_time > proj_time:
        print("  ⚠ Hebbian learning is slow - consider sparse updates")
    
    store_time = sum(brain_results.get('store_assembly', [0]))
    if store_time > 10:
        print("  ⚠ Assembly storage overhead - consider lazy storage")
    
    print("\n" + "="*70)
    print("BENCHMARK COMPLETE")
    print("="*70)
    
    return all_results


if __name__ == "__main__":
    run_all_benchmarks()

