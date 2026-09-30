"""One-off smoke script — run: uv run python neural_assemblies/tests/_smoke_wobbly_e2e.py"""
import os

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ["TRAIN_PROGRESS"] = "0"
os.environ["EMERGENT_DEV_CURRICULUM"] = "1"

from neural_assemblies.assembly_calculus.emergent import EmergentParser, build_vocabulary_preset
from neural_assemblies.assembly_calculus.emergent.acquisition import (
    bootstrap_from_wobbly_memory,
    classify_word_bootstrapped,
    decompose_holdout_classification,
    format_wobbly_report,
    mine_wobbly_episodes,
    parse_with_wobbly_probes,
    run_developmental_acquisition,
)

N, K = 3000, 30

print("=== Smoke 1: live probes on partially exposed parser ===")
p = EmergentParser(
    n=N, k=K, seed=7, fast_training=True,
    vocabulary=build_vocabulary_preset("medium"),
)
for w in ("the", "small", "bird", "runs", "dog"):
    if w not in p.stim_map:
        p.register_word(w)
sents = [
    ["the", "small", "bird"],
    ["the", "bird", "runs"],
    ["the", "small", "dog"],
]
for s in sents:
    p.ingest_raw_sentence(s)

result, probes = parse_with_wobbly_probes(p, ["the", "small", "bird", "runs"])
wobbly = [pr for pr in probes if pr.wobbly]
print("sentence: the small bird runs")
print("categories:", result["categories"])
print("has roles:", bool(result.get("roles")))
print("has phrases:", bool(result.get("phrases")))
print(f"probes: {len(probes)} words, {len(wobbly)} wobbly")
for pr in probes:
    print(
        f"  {pr.word!r}: cat={pr.category} n400={pr.n400:.2f} "
        f"p600={pr.p600:.2f} stab={pr.phrase_stability:.2f} "
        f"err={pr.error_active} wobbly={pr.wobbly}",
    )

mem = mine_wobbly_episodes(p, sents)
print()
print(format_wobbly_report(mem)[:900])

report = bootstrap_from_wobbly_memory(p, mem)
print()
print("=== Smoke 2: bootstrap report ===")
print(report)
cat, scores = classify_word_bootstrapped(p, "small")
print(f"small after bootstrap: {cat} (conf={scores.get('_confidence', 0):.3f})")

print()
print("=== Smoke 3: developmental acquisition to SENTENCES ===")
p2 = EmergentParser(
    n=N, k=K, seed=42, fast_training=True,
    vocabulary=build_vocabulary_preset("core"),
)
holdout = {"bird", "finds", "small"}
acq = run_developmental_acquisition(
    p2,
    max_stage="SENTENCES",
    holdout_words=holdout,
    wobbly_bootstrap=True,
    seed=42,
)
wobbly_stages = [r for r in acq.reflections if r.wobbly_bootstrap]
print("stages run:", " -> ".join(acq.stages_run))
print(f"wobbly bootstrap passes: {len(wobbly_stages)} / {len(acq.reflections)}")
for r in wobbly_stages[:4]:
    wb = r.wobbly_bootstrap or {}
    print(
        f"  {r.stage}: episodes={wb.get('episodes', 0)} "
        f"assigned={wb.get('assigned', {})}",
    )

decomp = decompose_holdout_classification(
    p2, {"bird": "NOUN", "finds": "VERB", "small": "ADJ"},
)
print(f"holdout bootstrapped accuracy: {decomp['accuracy_bootstrapped']:.0%}")
for word, expected in [("bird", "NOUN"), ("finds", "VERB"), ("small", "ADJ")]:
    cat, _ = classify_word_bootstrapped(p2, word)
    print(f"  {word}: expected={expected} got={cat}")
