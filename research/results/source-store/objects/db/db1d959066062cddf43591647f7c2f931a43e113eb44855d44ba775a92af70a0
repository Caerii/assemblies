"""Computed next-token baselines and oracle ceilings for the A3 corpora, through the shared runner.

No substrate runs here. Every predictor is a count estimated on the seed's
own TRAIN corpus (or the generator's exact conditional), scored on the
seed's TEST corpus by mean reciprocal rank of the true next word with a
seeded random tie-break -- the transducer study's own units, so the
transducer's MRR can be placed between a bigram and an oracle state.

Corpus families (`--corpus`):

  study4-template   the nine-phase template of study 4 (`study4/ntp.py`):
                    unigram, bigram, class-bigram, phase (the oracle state),
                    phase-exact (the generator's own class distribution given
                    the phase, uniform within a class: the true ceiling).
                    A perfect state adds ~0.02 MRR over a bigram here, which
                    is why the agreement corpus exists.
  agreement-chain   the chain corpus of PREREG_agreement_corpus.md at a
                    distractor gap (`--gap`): unigram, bigram, phase-only
                    oracle, and the phase+number oracle. The registered
                    acceptance criterion (oracle minus bigram, paired lower
                    bound >= GAP_BAR) is evaluated and reported as the verdict.

Run:  python -m research.runner a3-oracle-ceiling --corpus agreement-chain --gap 2 --tag UNIQUE
      python -m research.runner a3-oracle-ceiling --corpus study4-template --tag UNIQUE
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
import random

from neural_assemblies import describe_computed_baseline
from neural_assemblies.diagnostics import ensemble_from_values, paired_delta
from research.runner import (
    experiment_parser, run_experiment, validate_registered_seeds,
    validate_seed_identities,
)

PROTOCOL = "sequence.a3-oracle-ceiling"
VERSION = "2"          # version 1 printed means only; seeds 42..61 on study4
REGISTRATIONS = {
    "study4-template": "research/notes/sequence/PREREG_seq_a3_transducer.md",
    "agreement-chain": "research/notes/sequence/PREREG_agreement_corpus.md",
}
REGISTERED_SEEDS = tuple(range(42, 62))
VOCAB, N_TRAIN, N_TEST, TIE_SEED = 50, 200, 25, 0

STUDY4_PREDICTORS = ("unigram", "bigram", "class-bigram", "phase", "phase-exact")
CHAIN_PREDICTORS = ("unigram", "bigram", "phase-oracle", "phase-number-oracle")


# --- study-4 template ------------------------------------------------------

def _study4_phases(sentence, class_of):
    """The template phase AFTER each word: the state the machine is in when
    predicting the next word. Phases: 0 start; 1 after DET1; 2 after ADJ1;
    3 after NOUN1; 4 after VERB; 5 after PREP; 6 after DET(after PREP);
    7 after DET(after VERB); 8 after ADJ2; 9 after NOUN2."""
    out, ph = [], 0
    for w in sentence:
        c = class_of[w]
        nxt = {(0, 'DET'): 1, (1, 'ADJ'): 2, (1, 'NOUN'): 3, (2, 'NOUN'): 3,
               (3, 'VERB'): 4, (4, 'PREP'): 5, (4, 'DET'): 7, (5, 'DET'): 6,
               (6, 'NOUN'): 9, (7, 'ADJ'): 8, (7, 'NOUN'): 9, (8, 'NOUN'): 9}[(ph, c)]
        out.append(nxt)
        ph = nxt
    return out


def _exact_next_class(ph):
    return {1: {'ADJ': .5, 'NOUN': .5}, 2: {'NOUN': 1}, 3: {'VERB': 1},
            4: {'PREP': .5, 'DET': .5}, 5: {'DET': 1}, 6: {'NOUN': 1},
            7: {'ADJ': .5, 'NOUN': .5}, 8: {'NOUN': 1}}[ph]


def _mrr(test, words, scorer, rng):
    rr, n = 0.0, 0
    for s in test:
        for i, (a, truth) in enumerate(zip(s, s[1:])):
            keyed = sorted(((-scorer(a, i, s, w), rng.random(), w) for w in words))
            rr += 1.0 / ([w for _, _, w in keyed].index(truth) + 1)
            n += 1
    return rr / n


def study4_predictors(seed, *, vocab, n_train, n_test, tie_seed):
    from research.experiments.study4 import ntp
    class_of = {w: c for c, ws in ntp.WORD_CLASSES.items() for w in ws}
    words = ntp.vocabulary(vocab)
    keep = set(words)
    tr = [[w for w in s if w in keep] for s in ntp.generate(n_train, seed)]
    te = [[w for w in s if w in keep] for s in ntp.generate(n_test, seed + 500)]
    uni, big, cbig, pha = Counter(), defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    for s in tr:
        ph = _study4_phases(s, class_of)
        for i, (a, nxt) in enumerate(zip(s, s[1:])):
            uni[nxt] += 1
            big[a][nxt] += 1
            cbig[class_of[a]][nxt] += 1
            pha[ph[i]][nxt] += 1
    phases_of = {id(s): _study4_phases(s, class_of) for s in te}
    rng = random.Random(tie_seed)
    return {
        "unigram": _mrr(te, words, lambda a, i, s, w: uni[w], rng),
        "bigram": _mrr(te, words, lambda a, i, s, w: big[a][w], rng),
        "class-bigram": _mrr(te, words, lambda a, i, s, w: cbig[class_of[a]][w], rng),
        "phase": _mrr(te, words, lambda a, i, s, w: pha[phases_of[id(s)][i]][w], rng),
        "phase-exact": _mrr(
            te, words,
            lambda a, i, s, w: (_exact_next_class(phases_of[id(s)][i]).get(class_of[w], 0.0)
                                / len(ntp.WORD_CLASSES[class_of[w]])),
            rng),
    }


# --- agreement chain -------------------------------------------------------

def chain_predictors(seed, *, gap, vocab, n_train, n_test, tie_seed):
    from research.experiments.study4 import ntp_agree
    ntp_agree.use_chain(True, gap=gap)
    uni, big, phase, full = ntp_agree.oracle_gap(
        seed, vocab_size=vocab, n_train=n_train, n_test=n_test, tie_seed=tie_seed)
    return {"unigram": uni, "bigram": big, "phase-oracle": phase,
            "phase-number-oracle": full}


# --- runner ----------------------------------------------------------------

def _ensemble(e):
    return {**asdict(e), "low": e.low, "high": e.high}


def experiment(record):
    if record.get("mode", "study") == "study":
        validate_seed_identities(record["seeds"], REGISTERED_SEEDS)
    seeds = record["seeds"]
    p = record["parameters"]
    corpus = p["corpus"]
    sizes = dict(vocab=p["vocab"], n_train=p["n_train"], n_test=p["n_test"],
                 tie_seed=p["tie_seed"])
    per_seed = {}
    for seed in seeds:
        if corpus == "study4-template":
            per_seed[seed] = study4_predictors(seed, **sizes)
        elif corpus == "agreement-chain":
            per_seed[seed] = chain_predictors(seed, gap=p["gap"], **sizes)
        else:
            raise ValueError(f"unknown corpus family {corpus!r}")
    names = tuple(p["predictors"])
    if any(set(per_seed[s]) != set(names) for s in seeds):
        raise RuntimeError("predictor columns differ from the recorded protocol")
    ensembles = {name: ensemble_from_values([per_seed[s][name] for s in seeds], name, keys=list(seeds))
                 for name in names}
    for name in names:
        print(f"  {ensembles[name]}", flush=True)
    oracle = "phase" if corpus == "study4-template" else "phase-number-oracle"
    gap_delta = paired_delta(ensembles[oracle], ensembles["bigram"], label=f"{oracle} - bigram")
    print(f"  {gap_delta}", flush=True)
    out = {
        "predictors": {name: _ensemble(ensembles[name]) for name in names},
        "paired": {f"{oracle}_minus_bigram": _ensemble(gap_delta)},
        "rows": [{"seed": s, **per_seed[s]} for s in seeds],
    }
    if record["mode"] == "smoke":
        out["verdict"] = "VOID"
    elif corpus == "agreement-chain":
        from research.experiments.study4 import ntp_agree
        bar = ntp_agree.GAP_BAR
        out["bars"] = {f"corpus accepted: oracle - bigram lower bound >= {bar}": bool(gap_delta.low >= bar)}
        out["verdict"] = "PASS" if gap_delta.low >= bar else "FAIL"
    else:
        # The template's ceiling is reported, not judged: no bar was registered for it.
        out["verdict"] = "UNJUDGED"
    out["scope"] = "computed baselines and oracle ceilings; no substrate ran"
    return out


def main(argv=None):
    parser = experiment_parser(
        "Computed next-token baselines and oracle ceilings for the A3 corpora",
        engines=("computed_baseline",), default_seeds=REGISTERED_SEEDS,
    )
    parser.add_argument("--corpus", choices=tuple(REGISTRATIONS), default="agreement-chain")
    parser.add_argument("--gap", type=int, default=2,
                        help="distractor gap of the agreement chain (ignored for the template)")
    args = parser.parse_args(argv)
    validate_registered_seeds(parser, args, REGISTERED_SEEDS)
    if args.gap < 1:
        parser.error("--gap must be positive")
    template = args.corpus == "study4-template"
    parameters = {"corpus": args.corpus, "gap": None if template else args.gap,
                  "vocab": VOCAB, "n_train": N_TRAIN, "n_test": N_TEST, "tie_seed": TIE_SEED,
                  "predictors": list(STUDY4_PREDICTORS if template else CHAIN_PREDICTORS)}
    path = run_experiment(
        script=Path(__file__), protocol=PROTOCOL, protocol_version=VERSION,
        registration=REGISTRATIONS[args.corpus], engine=args.engine,
        seeds=args.seeds, tag=args.tag, smoke=args.smoke, parameters=parameters,
        baseline_semantics=describe_computed_baseline(corpus=args.corpus, tie_seed=TIE_SEED),
        measure=experiment,
    )
    print(path)


if __name__ == "__main__":
    main()
