"""Is the P600 reading PATHWAY LEARNING, or just which core area the word is in?

THE QUESTION THIS SETTLES. `anchored_p600_live` documents its mechanism as "a
category violation routes a WRONGLY-TYPED CORE through an UNTRAINED PATHWAY".
Those are TWO claims wearing one sentence:

    (a) UNTRAINED PATHWAY -- the word's assembly has weak synapses into the
        expected role area, because it was never bound there;
    (b) WRONGLY-TYPED CORE -- the drive is read out of a different brain area
        (VERB_CORE rather than NOUN_CORE), which differs in size, training
        history and degree distribution.

Every ERP contrast shipped so far varies (a) and (b) TOGETHER, because the only
way it makes a violation is to put a verb in a noun slot. So "trained pathway
vs untrained pathway" has never been separated from "two different areas with
different statistics", and (b) alone would produce a separation with no
learning in it at all.

THE DESIGN. Hold the SOURCE AREA FIXED and vary ONLY the pathway. The parser
supplies the split for free:

    trained nouns                      71
      in role_lexicons[ROLE_PATIENT]   36   <- patient pathway IS trained
      in [ROLE_AGENT] but NOT patient  29   <- role-trained, patient pathway is NOT

Both groups are NOUNs, so both project into NOUN_CORE and both are read as
NOUN_CORE -> ROLE_PATIENT. Both have been bound into SOME role area, so
"has any role training" is held constant too. The only thing that differs is
whether THIS word's assembly ever strengthened synapses into ROLE_PATIENT.

THREE ARMS, one sentence skeleton, one probe position:

    patient_trained   the dog want <noun bound into ROLE_PATIENT>
    agent_only        the dog want <noun bound into ROLE_AGENT only>
    verb_object       the dog want <trained VERB>          <- the standard violation

which DECOMPOSES the shipped effect:

  * `verb_object` vs `patient_trained` is the shipped contrast, and varies
    (a) and (b) together;
  * `agent_only` vs `patient_trained` isolates (a) -- pathway learning alone,
    same source area;
  * the gap between those two isolates (b) -- area identity alone.

FREQUENCY IS MATCHED BY CONSTRUCTION. Every item is chosen with ZERO
occurrences in the corpus that trained this depth, so the arms cannot differ by
how often the word was seen -- only by where it was bound. Words are also
required to classify NOUN (or VERB) on the parser being measured, because
trained and correctly-classified are independent properties and `small` is the
standing proof.

PRE-REGISTERED PREDICTIONS, stated before running:

  * If the metric reads PATHWAY LEARNING: AUC(agent_only above patient_trained)
    is clearly above 0.5 -- a noun never bound into ROLE_PATIENT delivers less
    drive there and so reads as a bigger deficit.
  * If it reads AREA IDENTITY: that AUC sits at ~0.5 while
    AUC(verb_object above patient_trained) stays high. The shipped effect would
    then contain no word-level learning at all, and "untrained pathway" would
    be the wrong description of it.

Either result is publishable and neither is the one to hope for.

GRANULARITY. 10 items per arm gives 100 pairs, so the AUC moves in steps of
0.01 rather than the shipped frame sets' 1/9 -- which is itself part of the
point, since a coarse statistic is how a ceiling gets mistaken for an effect.

WARM PARSERS ARE ACCEPTABLE HERE, and that is a claim about this design rather
than a shortcut. Every arm is measured on the SAME parser within a seed, so the
contrast is within-substrate; the cache threatens A/Bs that compare code
versions across arms, which this is not. Seeds are still swept because a single
parser is a realization, not an ensemble.
"""
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")
os.environ.setdefault("EMERGENT_ERP_FAST", "1")

from neural_assemblies.assembly_calculus.emergent.core.areas import (   # noqa: E402
    ROLE_AGENT, ROLE_PATIENT,
)
from neural_assemblies.assembly_calculus.emergent.evaluation.erp.frames import (  # noqa: E402
    collect_frame_samples,
)
from neural_assemblies.assembly_calculus.emergent.evaluation.sweep import (  # noqa: E402
    get_parser_cache,
)
from neural_assemblies.diagnostics import separation                    # noqa: E402

SEEDS = [11, 12, 42, 7, 19]
#: Sweepable so the AREA contrast can be re-read at a depth with far less role
#: training. `VOCABULARY_SPURT` would be the ideal zero-role control -- its
#: phases are ["lexicon", "distributional"], no roles -- but
#: `assess_erp_readiness` SHUTS the p600 gate when no role binding is stored,
#: so every probe would return the early 0.0 and the contrast would be void.
#: TWO_WORD is the shallowest depth that keeps the gate open.
DEPTH = os.environ.get("PATHWAY_DEPTH", "SENTENCES")
N_PER_ARM = 10
SUBJECT, VERB, DET = "dog", "want", "the"


def _corpus_counts():
    from neural_assemblies.lexicon.curriculum.stage4_sentences import STAGE4_CORPUS
    from neural_assemblies.lexicon.curriculum.stage3_two_word import STAGE3_CORPUS
    return Counter(
        w for line in list(STAGE4_CORPUS) + list(STAGE3_CORPUS)
        for w in line.split()
    )


def _pick(parser, counts):
    """Three matched item pools, or None if the parser cannot supply them."""
    nouns = set(parser.core_lexicons.get("NOUN_CORE", {}))
    verbs = set(parser.core_lexicons.get("VERB_CORE", {}))
    patient = set(parser.role_lexicons.get(ROLE_PATIENT, {}))
    agent = set(parser.role_lexicons.get(ROLE_AGENT, {}))

    def zero_freq(words):
        return sorted(w for w in words if counts.get(w, 0) == 0)

    def classified(words, want):
        out = []
        for w in words:
            cat, _ = parser.classify_word(w)
            if cat == want:
                out.append(w)
            if len(out) >= N_PER_ARM:
                break
        return out

    return {
        "patient_trained": classified(zero_freq(nouns & patient), "NOUN"),
        "agent_only": classified(zero_freq((nouns & agent) - patient), "NOUN"),
        "verb_object": classified(zero_freq(verbs), "VERB"),
    }


def main():
    counts = _corpus_counts()
    per_seed = {}

    for seed in SEEDS:
        parser = get_parser_cache().fork(DEPTH, seed=seed)
        pools = _pick(parser, counts)
        short = {k: len(v) for k, v in pools.items() if len(v) < N_PER_ARM}
        if short:
            print(f"seed {seed}: POOLS TOO SMALL {short} -- arms would differ "
                  f"in size as well as in kind; skipping this seed rather than "
                  f"comparing unequal arms")
            continue

        frames = [
            (arm, f"{arm}:{w}", [DET, SUBJECT, VERB, w])
            for arm, words in pools.items() for w in words
        ]
        samples = collect_frame_samples(parser, frames)
        by_arm = {}
        for s in samples:
            by_arm.setdefault(s.label, []).append(s.p600)
        per_seed[seed] = (pools, by_arm)

        print(f"=== depth={DEPTH} seed={seed} ===")
        for arm in ("patient_trained", "agent_only", "verb_object"):
            vals = by_arm.get(arm, [])
            mean = sum(vals) / len(vals) if vals else float("nan")
            print(f"  {arm:<16} n={len(vals):<3} mean p600={mean:.4f}  "
                  f"items={pools[arm]}")
        pt = by_arm.get("patient_trained", [])
        ao = by_arm.get("agent_only", [])
        vo = by_arm.get("verb_object", [])
        if pt and ao:
            s1 = separation(ao, pt, label="agent_only_vs_patient")
            print(f"  (a) PATHWAY only  AUC={s1.auc:.4f}  span={s1.span:.4f}")
        if pt and vo:
            s2 = separation(vo, pt, label="verb_vs_patient")
            print(f"  (a+b) SHIPPED     AUC={s2.auc:.4f}  span={s2.span:.4f}")
        if ao and vo:
            s3 = separation(vo, ao, label="verb_vs_agent_only")
            print(f"  (b) AREA only     AUC={s3.auc:.4f}  span={s3.span:.4f}")
        print()

    if not per_seed:
        print("no seed produced comparable pools -- nothing measured")
        return

    print("=" * 66)
    print(f"ACROSS SEEDS at depth={DEPTH} (mean +/- half-range)")
    for name, hi, lo in (
        ("(a) PATHWAY only : agent_only vs patient_trained", "agent_only", "patient_trained"),
        ("(a+b) SHIPPED    : verb_object vs patient_trained", "verb_object", "patient_trained"),
        ("(b) AREA only    : verb_object vs agent_only", "verb_object", "agent_only"),
    ):
        aucs = []
        for _seed, (_pools, by_arm) in sorted(per_seed.items()):
            if by_arm.get(hi) and by_arm.get(lo):
                aucs.append(separation(by_arm[hi], by_arm[lo], label=name).auc)
        if aucs:
            m = sum(aucs) / len(aucs)
            half = (max(aucs) - min(aucs)) / 2
            print(f"  {name:<52} AUC={m:.4f} +/-{half:.4f}  "
                  f"per-seed={[round(a, 3) for a in aucs]}")
    print()
    print("READING IT. (a) near 0.5 with (a+b) high means the shipped effect")
    print("contains NO word-level pathway learning -- it is area identity, and")
    print("`untrained pathway` is the wrong description of it. (a) clearly")
    print("above 0.5 means pathway learning is real and separable.")


if __name__ == "__main__":
    main()
