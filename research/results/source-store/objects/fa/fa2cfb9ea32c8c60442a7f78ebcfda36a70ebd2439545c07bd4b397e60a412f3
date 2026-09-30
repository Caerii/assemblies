"""Post-hoc drive probe (labeled as such -- no registered bar): WHY does
the split+scaling configuration answer PL for everything on Brown?

Hypothesis stated before reading: per-column homeostatic normalization
equalizes both value areas' column sums, so the SG area's mass is split
over ~10x more forms than the PL area's; an SG word's trained drive into
SG lands near (or below) the norm-initialized BACKGROUND drive into PL,
and cross-area MI comparison (the weak primitive) inverts.
Prediction: for SG words, drive_PL / drive_SG >= ~1; for PL words,
drive_PL >> drive_SG (both effects push PL).
"""
import os
import statistics as st
import sys

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from childes_graduation import build_number_slice, load_corpus  # noqa: E402
import childes_graduation as cg  # noqa: E402

utts, _s, _f = load_corpus()
sentences, labels, exposures, ambiguous = build_number_slice(utts)
r = None  # build one cell inline, mirroring run_cell(seed=42, A)
import random

import numpy as np

from neural_assemblies.assembly_calculus.emergent import EmergentParser
from neural_assemblies.assembly_calculus.emergent.core.areas import (
    NUMBER, feature_value_area,
)
from neural_assemblies.assembly_calculus.emergent.core.grounding import (
    GroundingContext,
)

random.seed(42)
np.random.seed(42)
p = EmergentParser(n=3000, k=30, seed=42, fast_training=True)
for w in labels:
    if w not in p.stim_map:
        p.register_word(w)
    p.word_grounding[w] = GroundingContext(visual=[w])
scaled = frozenset(feature_value_area(NUMBER, l) for l in ("SG", "PL"))
eng = p.brain._engine
eng.synaptic_scaling = scaled
p.brain._synaptic_scaling = scaled
eng.synaptic_scaling_deferred = True
p.morph_flush_every = 40
p.train_number(sentences, labels=labels)

sg_ratio, pl_ratio = [], []
rows = []
for w, lab in sorted(labels.items(), key=lambda kv: -exposures[kv[0]]):
    _got, diag = p.recall_number(w)
    s = diag["scores"]
    if not s or s.get("SG", 0) <= 0:
        continue
    ratio = s["PL"] / s["SG"]
    (pl_ratio if lab == "PL" else sg_ratio).append(ratio)
    if len(rows) < 12:
        rows.append((w, lab, exposures[w], round(s["SG"], 1),
                     round(s["PL"], 1)))

print("form / label / exposure / drive_SG / drive_PL")
for r_ in rows:
    print("  ", r_)
print(f"SG words: n={len(sg_ratio)} median PL/SG drive ratio "
      f"{st.median(sg_ratio):.3f} (>{1.0} means inversion)")
print(f"PL words: n={len(pl_ratio)} median PL/SG drive ratio "
      f"{st.median(pl_ratio):.3f}")
frac_inverted = sum(x >= 1 for x in sg_ratio) / len(sg_ratio)
print(f"fraction of SG words whose PL drive wins: {frac_inverted:.3f}")
