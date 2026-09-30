"""#149 adoption gate for split_feature_areas=True as DEFAULT, n=10 paired.

REGISTERED BAR (before data): flip the default iff, at the default corpus
(uniform-50, n=3000, k=30, default "mi" readout), the PAIRED per-seed
delta (split - shared) over seeds 42-51 satisfies:
  G1  balanced tense delta: mean > -0.05 AND paired CI does not exclude 0
      from below at mean <= -0.05 (i.e., no measured harm);
  G2  number SG under split: mean >= 0.8 (the slow test's pinned class);
  G3  number PL under split: mean >= shared's mean - 0.05 (PL is at chance
      on shared -- split must not be WORSE than chance-level).
If G1 fails, the default stays False and the adoption is regime-scoped to
>=200-frame budgets (where E15/E19b measured split's win).
Context: seed-42 alone read PAST -0.26 (the pinned floor failure), but
seeds 43/44 read -0.26/+0.17 -- noise-dominated, hence this ensemble.
"""
import os
import statistics as st

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")

from neural_assemblies.assembly_calculus.emergent import EmergentParser
from neural_assemblies.assembly_calculus.emergent.curriculum import (
    CurriculumTrainer,
)
from neural_assemblies.assembly_calculus.emergent.vocabulary_builder import (
    build_vocabulary_preset,
)
from neural_assemblies.assembly_calculus.emergent.evaluation.morph_features \
    import attested_morph_sets
from neural_assemblies.lexicon.data import NOUNS

SEEDS = list(range(42, 52))


def cell(seed, split):
    p = EmergentParser(n=3000, k=30, seed=seed,
                       vocabulary=build_vocabulary_preset("core"),
                       fast_training=True, split_feature_areas=split)
    t = CurriculumTrainer(p)
    for stage in ("FIRST_WORDS", "VOCABULARY_SPURT", "TWO_WORD", "SENTENCES"):
        t.train_stage(stage)
    sets_ = attested_morph_sets(p)

    def acc(items, label, recall):
        return sum(recall(w)[0] == label for w in items) / max(1, len(items))

    past = acc(sets_["PAST"], "PAST", p.recall_tense)
    pres = acc(sets_["PRESENT"], "PRESENT", p.recall_tense)
    sg_items = [e["lemma"] for e in NOUNS
                if e.get("forms", {}).get("plural")
                and e["lemma"] in p.stim_map][:15]
    pl_items = [e["forms"]["plural"] for e in NOUNS
                if e.get("forms", {}).get("plural")
                and e["forms"]["plural"] in p.stim_map][:15]
    sg = acc(sg_items, "SG", p.recall_number)
    pl = acc(pl_items, "PL", p.recall_number)
    return {"past": past, "pres": pres, "bal": (past + pres) / 2,
            "sg": sg, "pl": pl, "n_pl_items": len(pl_items)}


rows = {}
for seed in SEEDS:
    rows[seed] = {s: cell(seed, s) for s in (True, False)}
    r = rows[seed]
    print(f"seed={seed} split bal={r[True]['bal']:.3f} "
          f"(P{r[True]['past']:.2f}/p{r[True]['pres']:.2f} "
          f"sg={r[True]['sg']:.2f} pl={r[True]['pl']:.2f})  "
          f"shared bal={r[False]['bal']:.3f} "
          f"(P{r[False]['past']:.2f}/p{r[False]['pres']:.2f} "
          f"sg={r[False]['sg']:.2f} pl={r[False]['pl']:.2f})", flush=True)

deltas = [rows[s][True]["bal"] - rows[s][False]["bal"] for s in SEEDS]
mean = st.mean(deltas)
ci = 2.262 * st.stdev(deltas) / len(deltas) ** 0.5
sg_mean = st.mean(rows[s][True]["sg"] for s in SEEDS)
pl_split = st.mean(rows[s][True]["pl"] for s in SEEDS)
pl_shared = st.mean(rows[s][False]["pl"] for s in SEEDS)
print(f"\nG1 paired balanced-tense delta = {mean:+.4f} +/- {ci:.4f}")
print(f"G2 split SG mean = {sg_mean:.4f}")
print(f"G3 split PL mean = {pl_split:.4f} vs shared PL mean = {pl_shared:.4f}")
