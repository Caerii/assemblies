"""#150 (task #30): the CHILDES graduation -- the five-level law on real
child-directed speech.

REGISTERED before any transcript has been read (the corpus statistics ARE
the data; these bars were committed while data/childes/ was empty).

WHY THIS EXPERIMENT. Every number-arc result stands on a synthetic corpus
whose statistics we chose. Real CDS is Zipfian, repetitive, and large --
exactly the regime whose failure modes the arc named (mass concentration,
teacher-label merging, flush schedule, repetition renormalization). The
graduation asks whether the LAWS transfer, not whether a headline number
does.

PHASE 0 -- corpus census (no substrate). From adult utterances
(curriculum/childes.py; %mor-aligned only for teacher signals):
  C1  Zipf shape: rank-frequency slope on the token spectrum in
      [-1.6, -0.6] (the classic law with tolerance; measured, not
      assumed -- if CDS here is NOT Zipfian, every "real corpus is
      Zipfian" argument in the notes must be revised).
  C2  Coverage: >= 300 aligned utterances whose nouns all carry %mor
      number and whose words can enter the parser's stimulus space.
      BELOW this the experiment STOPS and reports scoping, because a
      substrate run on a starved slice would measure the slice, not
      the corpus ([[the-calibration-frames-were-untrained]]).
  C3  Plural share among noun tokens reported (no bar -- it parameterizes
      the frequency-imbalance prediction below).

PHASE 1 -- the number exam on real CDS (only if C2 passes). Parser at the
production configuration (split default, K=40, deferred scaling on the
value areas; morph_readout per E15's crossing: report BOTH). Seeds 42-51,
paired across arms by seed.
  F1  EXPOSURE LAW TRANSFERS: Spearman(per-form CDS exposure,
      per-form recall correctness) > 0 with the n=10 CI excluding 0.
      This is the arc's central law (rho 0.50-0.87 on five synthetic
      corpora) meeting its first natural distribution.
  F2  SCHEDULE RIGHT WALL TRANSFERS: paired delta (K=40 - K=0) on
      balanced number is >= 0; claim the wall only if the CI excludes 0.
      (E18/E19b measured the wall at >= 200 episodes; if the usable CDS
      slice trains fewer, F2 reports underpowered rather than failed.)
  F3  ARCHITECTURE TRANSFERS: paired (split - shared) PL delta > 0.
      The #149 gate measured +0.33 on the synthetic default corpus; a
      sign flip here would say the split's PL rescue is an artifact of
      the generator, which would be a MAJOR finding against adoption.
  Guards: SG stays >= 0.8 in every arm; lexicon health (no area
      collapse per diagnostics.area_health) on 3 probe words per arm.

DATA. data/childes/<CorpusName>/*.cha (CHAT format, TalkBank). The
census records corpus name, file count, and utterance counts so the
result names its sample. NO transcript content is committed to the repo
(TalkBank terms: cite, do not redistribute) -- only derived statistics.

Run: python childes_graduation.py [census|full]
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")

import numpy as np

from neural_assemblies.assembly_calculus.emergent.curriculum.childes import (
    frequency_spectrum,
    mor_number_teacher,
    read_cha,
    read_childesdb_jsonl,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..",
                        "data", "childes")
OUT_PATH = os.path.join(os.path.dirname(__file__),
                        "childes_graduation_results.json")
SEEDS = list(range(42, 52))


def load_corpus():
    """All .cha and childes-db .jsonl files under data/childes/,
    recursively (the jsonl route is the registered auth-wall deviation
    -- see fetch_childes_brown.py)."""
    utts, stats_list, files = [], [], []
    for root, _dirs, names in os.walk(DATA_DIR):
        for fn in sorted(names):
            if fn.endswith(".cha"):
                reader = read_cha
            elif fn.endswith(".jsonl"):
                reader = read_childesdb_jsonl
            else:
                continue
            path = os.path.join(root, fn)
            text = open(path, encoding="utf-8", errors="replace").read()
            u, s = reader(text)
            utts.extend(u)
            stats_list.append(s)
            files.append(os.path.relpath(path, DATA_DIR))
    return utts, stats_list, files


def zipf_slope(freq) -> float:
    """OLS slope of log(freq) on log(rank), ranks 1..min(500, V)."""
    counts = sorted(freq.values(), reverse=True)[:500]
    if len(counts) < 50:
        return float("nan")
    x = np.log(np.arange(1, len(counts) + 1))
    y = np.log(np.asarray(counts, float))
    slope = float(np.polyfit(x, y, 1)[0])
    return slope


def census() -> dict:
    utts, stats_list, files = load_corpus()
    if not files:
        return {"error": "no .cha files under data/childes/ -- "
                         "acquire a TalkBank corpus first"}
    freq = frequency_spectrum(utts)
    aligned = [u for u in utts if u.mor is not None]
    noun_tokens = pl_tokens = 0
    teachable = 0
    for u in aligned:
        t = mor_number_teacher(u)
        if t:
            teachable += 1
        noun_tokens += len(t)
        pl_tokens += sum(v == "PL" for v in t.values())
    out = {
        "files": files,
        "utterances_kept": len(utts),
        "aligned": len(aligned),
        "misaligned": sum(s.mor_misaligned for s in stats_list),
        "types": len(freq),
        "tokens": sum(freq.values()),
        "C1_zipf_slope": zipf_slope(freq),
        "C2_teachable_utterances": teachable,
        "C3_plural_share": (pl_tokens / noun_tokens) if noun_tokens else None,
    }
    out["C1_pass"] = (-1.6 <= out["C1_zipf_slope"] <= -0.6)
    out["C2_pass"] = teachable >= 300
    return out


# ---------------------------------------------------------------------------
# PHASE 1 -- registered protocol details (fixed before the run; the census
# had already passed C1 -1.096 / C2 40859 when these were written, but no
# substrate cell had been run).
#
#   N_TRAIN = 1000 teachable utterances in corpus order. Chosen to MATCH
#   the terminal cell's PL mass: at C3 = 0.102 plural share this slice
#   carries ~130 PL noun episodes, the zipf-200 regime where every
#   adopted mechanism was measured. Total noun episodes reported for
#   F2's power clause (>= 200 = powered).
#   TEACHER: corpus annotation via train_number(labels=...) -- one
#   GLOBAL form->label map, majority over the slice; forms attested
#   with BOTH labels are trained on majority but EXCLUDED from the exam
#   and counted (a homograph is not a fair exam item for either label).
#   EXAM: every unambiguous trained form, recall_number; correctness
#   against the corpus label; BOTH readouts recorded (mi = default,
#   overlap = production-at-scale), bars evaluated on mi.
#   ARMS (paired by seed, seeds 42-51):
#     A split+K40 (the adopted defaults; deferred scaling on value areas)
#     B split+K0  (per-phase flush)      -> F2 = A - B balanced, paired
#     C shared+K40 (legacy architecture) -> F3 = A - C on PL acc, paired
#   F1 on arm A: diagnostics.spearman(exposure, mi_correct) per seed.
#   DECISION RULES: F1 mean > 0 with n=10 CI excluding 0; F2 claim the
#   wall only if CI excludes 0 (else direction-only); F3 pass iff CI
#   excludes 0. GUARD: SG >= 0.8 per arm mean -- a failure is reported
#   as a finding, never adjusted away.
#   Parser: n=3000, k=30, fast_training -- the slow-test scale. Corpus
#   nouns registered with visual grounding (the papers' grounding
#   assumption for objects); np/random reseeded per cell (the global
#   RNG leak lesson).
# ---------------------------------------------------------------------------

N_TRAIN = 1000

#: CG_SMOKE=1: API-breakage check ONLY (1 seed, 50 utterances, n=600) --
#: direction is never read from a smoke run, per the standing rule.
SMOKE = os.environ.get("CG_SMOKE") == "1"
if SMOKE:
    SEEDS = [999]
    N_TRAIN = 50


def build_number_slice(utts, n_train=None):
    if n_train is None:
        n_train = N_TRAIN
    from collections import Counter
    sl, label_counts = [], {}
    for u in utts:
        t = mor_number_teacher(u)
        if not t:
            continue
        sl.append(u)
        for w, lab in t.items():
            label_counts.setdefault(w, Counter())[lab] += 1
        if len(sl) >= n_train:
            break
    labels, ambiguous = {}, []
    exposures = {}
    for w, c in label_counts.items():
        if len(c) > 1:
            ambiguous.append(w)
        labels[w] = c.most_common(1)[0][0]
        exposures[w] = sum(c.values())
    sentences = [list(u.words) for u in sl]
    return sentences, labels, exposures, sorted(ambiguous)


def run_cell(seed, split, flush_k, sentences, labels, exposures, ambiguous):
    import random

    import numpy as np_

    from neural_assemblies.assembly_calculus.emergent import EmergentParser
    from neural_assemblies.assembly_calculus.emergent.core.areas import (
        NUMBER, feature_value_area,
    )
    from neural_assemblies.assembly_calculus.emergent.core.grounding import (
        GroundingContext,
    )

    random.seed(seed)
    np_.random.seed(seed)
    n, k = (600, 20) if SMOKE else (3000, 30)
    p = EmergentParser(n=n, k=k, seed=seed, fast_training=True,
                       split_feature_areas=split)
    for w in labels:
        if w not in p.stim_map:
            p.register_word(w)
        p.word_grounding[w] = GroundingContext(visual=[w])
    scaled = (frozenset(feature_value_area(NUMBER, l) for l in ("SG", "PL"))
              if split else frozenset((NUMBER,)))
    eng = p.brain._engine
    eng.synaptic_scaling = scaled
    p.brain._synaptic_scaling = scaled
    eng.synaptic_scaling_deferred = True
    p.morph_flush_every = flush_k
    p.train_number(sentences, labels=labels)

    rows = []
    for w, lab in labels.items():
        if w in ambiguous:
            continue
        got, diag = p.recall_number(w)
        mi = diag.get("mi_answer", got)
        ov = diag.get("overlap_answer")
        ov = ov if ov is not None else mi
        rows.append({"form": w, "label": lab, "exposure": exposures[w],
                     "mi": mi == lab, "ov": ov == lab})

    def acc(lab, key):
        xs = [r[key] for r in rows if r["label"] == lab]
        return (sum(xs) / len(xs)) if xs else None

    return {
        "rows": rows,
        "sg_mi": acc("SG", "mi"), "pl_mi": acc("PL", "mi"),
        "sg_ov": acc("SG", "ov"), "pl_ov": acc("PL", "ov"),
    }


def phase1():
    import statistics as st

    from neural_assemblies.diagnostics import spearman

    utts, _stats, files = load_corpus()
    sentences, labels, exposures, ambiguous = build_number_slice(utts)
    noun_episodes = sum(exposures.values())
    n_pl_exam = sum(1 for w, l in labels.items()
                    if l == "PL" and w not in ambiguous)
    if n_pl_exam == 0:
        raise SystemExit("no unambiguous PL exam forms in the slice -- "
                         "the registered stop, widen N_TRAIN")
    arms = {"A": (True, 40), "B": (True, 0), "C": (False, 40)}
    seeds_out = {}
    for seed in SEEDS:
        seeds_out[seed] = {}
        for arm, (split, k) in arms.items():
            r = run_cell(seed, split, k, sentences, labels, exposures,
                         ambiguous)
            seeds_out[seed][arm] = r
            print(f"seed={seed} arm={arm} sg_mi={r['sg_mi']:.3f} "
                  f"pl_mi={r['pl_mi']:.3f} sg_ov={r['sg_ov']:.3f} "
                  f"pl_ov={r['pl_ov']:.3f}", flush=True)

    def bal(r):
        return (r["sg_mi"] + r["pl_mi"]) / 2

    f1 = [spearman([row["exposure"] for row in seeds_out[s]["A"]["rows"]],
                   [row["mi"] for row in seeds_out[s]["A"]["rows"]])
          for s in SEEDS]
    f2 = [bal(seeds_out[s]["A"]) - bal(seeds_out[s]["B"]) for s in SEEDS]
    f3 = [seeds_out[s]["A"]["pl_mi"] - seeds_out[s]["C"]["pl_mi"]
          for s in SEEDS]

    def mci(xs):
        xs = [x for x in xs if x == x]  # NaN seeds reported, not averaged
        if len(xs) < 2:
            return {"mean": xs[0] if xs else None, "ci": None, "n": len(xs)}
        return {"mean": st.mean(xs),
                "ci": 2.262 * st.stdev(xs) / len(xs) ** 0.5, "n": len(xs)}

    out = {
        "files": files, "n_train": len(sentences),
        "noun_episodes": noun_episodes, "exam_forms": len(labels),
        "ambiguous_excluded": len(ambiguous),
        "F1_spearman": mci(f1), "F1_per_seed": f1,
        "F2_paired_A_minus_B": mci(f2),
        "F3_paired_PL_A_minus_C": mci(f3),
        "guard_sg": {arm: st.mean(seeds_out[s][arm]["sg_mi"]
                                  for s in SEEDS) for arm in arms},
        "per_seed": {s: {a: {k: v for k, v in seeds_out[s][a].items()
                             if k != "rows"} for a in arms} for s in SEEDS},
    }
    return out


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "census"
    results = {"census": census()}
    print(json.dumps(results["census"], indent=2))
    if mode == "full":
        if not results["census"].get("C2_pass"):
            print("C2 FAILED -- Phase 1 does not run (the registered "
                  "stop, not an error).")
        else:
            results["phase1"] = phase1()
            print(json.dumps({k: v for k, v in results["phase1"].items()
                              if k != "per_seed"}, indent=2))
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"-> {OUT_PATH}")


if __name__ == "__main__":
    main()
