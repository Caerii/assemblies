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


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "census"
    results = {"census": census()}
    print(json.dumps(results["census"], indent=2))
    if mode == "full":
        if not results["census"].get("C2_pass"):
            print("C2 FAILED -- Phase 1 does not run (the registered "
                  "stop, not an error).")
        else:
            raise SystemExit(
                "Phase 1 harness lands in the next unit -- census first, "
                "per the registration.")
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"-> {OUT_PATH}")


if __name__ == "__main__":
    main()
