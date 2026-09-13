"""Registered position-specific test of subject-number representation in temporal arcs.

Registration: research/notes/sequence/PREREG_temporal_positions.md
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from neural_assemblies.diagnostics import ensemble_from_values, paired_delta
from neural_assemblies import describe_hashed_transducer
from research.experiments.study4.ntp_agree import CHAIN_CLASSES, generate_chain
from research.experiments.temporal_observations import capture_chain_arcs
from research.runner import (
    ExperimentOutput, experiment_parser, run_experiment,
    validate_registered_seeds, validate_seed_identities,
)


ARMS = {
    "g0": {"predict_gain": 0.0, "state_blind": False},
    "g1": {"predict_gain": 1.0, "state_blind": False},
    "blind_g1": {"predict_gain": 1.0, "state_blind": True},
}
GAPS = (2, 3, 4, 5, 6)


def chain_positions(gap):
    """Processed positions of the chain AUX N^g VERB N^g PRON N^g (TAG unobserved).

    Agreeing tokens sit at 0, g+1 and 2g+2; every other position is a
    distractor noun. Offset j (1..g) is the j-th noun after an agreeing
    token, the axis the decay law is fitted on.
    """
    if type(gap) is not int or gap not in GAPS:
        raise ValueError(f"gap must be one of {GAPS}")
    agreeing = [0, gap + 1, 2 * gap + 2]
    processed = 3 * gap + 3
    offsets = {str(j): [a + j for a in agreeing] for j in range(1, gap + 1)}
    distractors = sorted(p for p in range(processed) if p not in agreeing)
    return {"processed_positions": processed, "agreeing_positions": agreeing,
            "distractor_positions": distractors, "offset_positions": offsets,
            "first_positions": offsets["1"],
            "second_positions": offsets.get("2", [])}


def registered(gap):
    """The registered protocol at one gap (PREREG_temporal_positions.md, Amendment 1)."""
    return {
        "gap": gap, "n": 10_000, "n_arc": 10_000, "k": 200,
        "p": 0.05, "organ_p": 0.20, "beta": 0.10, "w_max": 20.0,
        "refracted_strength": 0.10, "ground_rounds": 5, "train_rounds": 3,
        "train_sentences": 200, "test_sentences": 25, "max_potentiations": 64,
        "batch_size": 5, "state_mode": "copy", "norm_init": True,
        "arms": ARMS, **chain_positions(gap),
        "bars": {"amplification": 0.05, "state_dependence": 0.05,
                 "plain_carry": 0.02, "blind_upper": 0.02,
                 "carry_starts": 0.05, "persists": 0.02},
    }


def smoke(gap):
    return {**registered(gap), "n": 128, "n_arc": 128, "k": 8, "p": 0.2,
            "organ_p": 0.3, "ground_rounds": 2, "train_rounds": 2,
            "train_sentences": 8, "test_sentences": 12,
            "max_potentiations": 256, "batch_size": 3}


REGISTERED = registered(2)
SMOKE = smoke(2)


def _ensemble(value, label, seeds):
    result = ensemble_from_values(value, label=label, keys=seeds)
    return {**asdict(result), "low": result.low, "high": result.high}


def _paired(left, right, label, seeds):
    a = ensemble_from_values(left, label=f"{label}:left", keys=seeds)
    b = ensemble_from_values(right, label=f"{label}:right", keys=seeds)
    result = paired_delta(a, b, label=label)
    return {**asdict(result), "low": result.low, "high": result.high}


def _position_values(reports, processed):
    values = []
    for report in reports:
        rows = report["analysis"]["positions"]
        by_position = {row["position"]: row for row in rows}
        if set(by_position) != set(range(processed)):
            raise ValueError(
                f"arm report does not contain the complete gap position curve ({processed} positions)")
        values.append({position: by_position[position]["contrast"] for position in by_position})
    return values


def score_arms(arms, seeds, parameters):
    """Summarize brain-level estimands; sentence pairs never become replicates."""
    if set(arms) != set(ARMS) or any([row["seed"] for row in arms[name]] != seeds for name in ARMS):
        raise ValueError("all arms must preserve the same registered seed identities and order")
    gap = parameters["gap"]
    expected = chain_positions(gap)
    if any(parameters[key] != expected[key] for key in expected):
        raise ValueError(f"position estimands differ from the registered gap-{gap} protocol")
    distractors, offsets = expected["distractor_positions"], expected["offset_positions"]
    processed = expected["processed_positions"]
    summaries, raw = {}, {}
    for arm, reports in arms.items():
        curves = _position_values(reports, processed)
        metrics = {
            "D": [sum(curve[p] for p in distractors) / len(distractors) for curve in curves],
        }
        for j, positions in offsets.items():
            metrics[f"D{j}"] = [sum(curve[p] for p in positions) / len(positions)
                                for curve in curves]
        raw[arm] = metrics
        summaries[arm] = {name: _ensemble(values, f"{arm}:{name}", seeds)
                          for name, values in metrics.items()}
        summaries[arm]["positions"] = {
            str(position): _ensemble([curve[position] for curve in curves],
                                     f"{arm}:position-{position}", seeds)
            for position in range(processed)}
        last = f"D{gap}"
        summaries[arm]["D1_minus_D2"] = _paired(
            metrics["D1"], metrics["D2"], f"{arm}:D1-D2", seeds)
        summaries[arm]["D1_minus_Dlast"] = _paired(
            metrics["D1"], metrics[last], f"{arm}:D1-{last}", seeds)
        # the decay axis: brain-level means per offset, reported for the fit
        summaries[arm]["decay_curve"] = {
            j: summaries[arm][f"D{j}"]["mean"] for j in offsets}
    comparisons = {
        "g1_minus_g0": _paired(raw["g1"]["D"], raw["g0"]["D"], "g1-g0:D", seeds),
        "g1_minus_blind": _paired(raw["g1"]["D"], raw["blind_g1"]["D"],
                                   "g1-blind:D", seeds),
    }
    bars = parameters["bars"]
    last = f"D{gap}"
    checks = {
        "TP-1 predicted-win amplification": comparisons["g1_minus_g0"]["low"] > bars["amplification"],
        "TP-2 state dependence": comparisons["g1_minus_blind"]["low"] > bars["state_dependence"],
        "TP-3 plain conjunction carry": summaries["g0"]["D"]["low"] > bars["plain_carry"],
        "TP-4 state-blind negative": summaries["blind_g1"]["D"]["high"] < bars["blind_upper"],
    }
    if gap > 2 or "carry_starts" in bars:
        # Amendment 1 (the decay law): the carry starts at the first distractor,
        # decays across the gap, and is still present at the last distractor
        # before the agreeing token. Blind and plain-conjunction gates per gap.
        checks.update({
            "DL-1 carry starts at offset 1 (g=1)": summaries["g1"]["D1"]["low"] > bars["carry_starts"],
            "DL-2 carry decays across the gap (g=1)": summaries["g1"]["D1_minus_Dlast"]["low"] > 0.0,
            f"DL-3 carry persists at offset {gap} (g=1)": summaries["g1"][last]["low"] > bars["persists"],
            "DL-4 state-blind negative": summaries["blind_g1"]["D"]["high"] < bars["blind_upper"],
            "DL-5 plain conjunction absent at offset 2 (g=0)": summaries["g0"]["D2"]["high"] < bars["blind_upper"],
        })
    return {"summaries": summaries, "comparisons": comparisons, "checks": checks,
            "instrument_valid": checks["TP-4 state-blind negative"]}


def _training_schedule(corpora, word_index):
    from neural_assemblies.core._torch_ops import torch_ops

    rows = []
    for sentences in corpora:
        words, targets, starts = [], [], []
        for sentence in sentences:
            for position, (word, target) in enumerate(zip(sentence, sentence[1:])):
                words.append(word_index[word]); targets.append(word_index[target])
                starts.append(position == 0)
        rows.append((words, targets, starts))
    length = max(len(row[0]) for row in rows)
    W = torch_ops.full((len(rows), length), -1, dtype=torch_ops.int64)
    T = torch_ops.full((len(rows), length), -1, dtype=torch_ops.int64)
    St = torch_ops.zeros(len(rows), length, dtype=torch_ops.bool)
    for brain, (words, targets, starts) in enumerate(rows):
        W[brain, :len(words)] = torch_ops.tensor(words)
        T[brain, :len(targets)] = torch_ops.tensor(targets)
        St[brain, :len(starts)] = torch_ops.tensor(starts)
    return W, T, St


def run_arm(seeds, parameters, arm_name, organ_semantics):
    from neural_assemblies.core._torch_ops import torch_ops
    from neural_assemblies.core.torch_engine._hashed_transducer import HashedTransducer

    arm = parameters["arms"][arm_name]
    words = [word for values in CHAIN_CLASSES.values() for word in values]
    if len(words) != 50 or len(set(words)) != 50:
        raise RuntimeError("registered chain vocabulary is no longer exactly 50 unique words")
    reports = []
    for offset in range(0, len(seeds), parameters["batch_size"]):
        group = seeds[offset:offset + parameters["batch_size"]]
        train = [generate_chain(parameters["train_sentences"], seed,
                                gap=parameters["gap"]) for seed in group]
        test = [generate_chain(parameters["test_sentences"], seed + 500,
                               gap=parameters["gap"]) for seed in group]
        transducer = HashedTransducer(
            group, words, n=parameters["n"], n_arc=parameters["n_arc"],
            n_state=parameters["n_arc"], k=parameters["k"], p=parameters["p"],
            beta=parameters["beta"], organ_p=parameters["organ_p"],
            w_max=parameters["w_max"], norm_init=parameters["norm_init"],
            max_potentiations=parameters["max_potentiations"],
            refracted_strength=parameters["refracted_strength"],
            state_mode=parameters["state_mode"],
            predict_gain=arm["predict_gain"], device="cuda",
            organ_semantics=organ_semantics)
        transducer.ground(rounds=parameters["ground_rounds"])
        W, T, St = _training_schedule(train, transducer.word_index)
        transducer.train_schedules(W, T, St, rounds=parameters["train_rounds"])
        captured = capture_chain_arcs(
            transducer, test, gap=parameters["gap"], rounds=parameters["train_rounds"],
            state_blind=arm["state_blind"])
        for corpus, report in zip(test, captured):
            report["test_corpus"] = corpus
            report["arm"] = arm_name
            reports.append(report)
        del transducer, W, T, St
        torch_ops.cuda.empty_cache()
    return reports


def _validate_parameters(parameters, mode):
    gap = parameters.get("gap")
    if gap not in GAPS:
        raise ValueError(f"gap must be one of {GAPS}")
    expected = smoke(gap) if mode == "smoke" else registered(gap)
    if parameters != expected:
        raise ValueError("resolved parameters differ from the fixed temporal-position protocol")


def experiment(record):
    parameters, seeds = record["parameters"], record["seeds"]
    _validate_parameters(parameters, record["mode"])
    if record["mode"] == "study":
        validate_seed_identities(seeds, list(range(82, 102)))
    arms = {name: [] for name in ARMS}
    # Keep each seed chunk adjacent across arms so paired brains share execution conditions.
    size = parameters["batch_size"]
    for offset in range(0, len(seeds), size):
        group = seeds[offset:offset + size]
        for name in ARMS:
            arms[name].extend(run_arm(
                group, parameters, name,
                record["execution_semantics"]["profiles"][name],
            ))
    scored = score_arms(arms, seeds, parameters)
    passed = scored["instrument_valid"] and all(scored["checks"].values())
    corpora = {}
    for index, seed in enumerate(seeds):
        corpus = arms["g0"][index]["test_corpus"]
        if any(arms[name][index]["test_corpus"] != corpus for name in ARMS):
            raise RuntimeError("paired arms did not evaluate the same test corpus")
        corpora[str(seed)] = corpus
    raw_arms = {name: [{key: value for key, value in report.items()
                        if key != "test_corpus"} for report in reports]
                for name, reports in arms.items()}
    observations = {
        **scored, "raw_observations": {
            "attachment": "raw-frames.json.gz", "format": "temporal-position-frames-v1",
            "brain_count": len(seeds), "arm_count": len(ARMS),
            "frame_count": sum(len(report["frames"]) for reports in arms.values()
                               for report in reports),
        },
        "verdict": "VOID" if record["mode"] == "smoke" else ("PASS" if passed else "FAIL"),
        "scope": "position-specific subject-number representation in the fixed gap-2 chain",
    }
    return ExperimentOutput(observations, {
        "raw-frames.json.gz": {"format": "temporal-position-frames-v1",
                               "corpora": corpora, "arms": raw_arms}})


def main(argv=None):
    parser = experiment_parser(__doc__ or "Temporal position study",
                               engines=("hashed_transducer",),
                               default_seeds=tuple(range(82, 102)))
    parser.add_argument("--gap", type=int, default=2, choices=GAPS,
                        help="distractor nouns between agreeing tokens (Amendment 1: 3..6)")
    args = parser.parse_args(argv)
    validate_registered_seeds(parser, args, tuple(range(82, 102)))
    parameters = smoke(args.gap) if args.smoke else registered(args.gap)
    path = run_experiment(
        script=Path(__file__), protocol="sequence.temporal-positions", protocol_version="2",
        registration="research/notes/sequence/PREREG_temporal_positions.md",
        engine=args.engine, seeds=args.seeds, tag=args.tag, smoke=args.smoke,
        minimum_study_seeds=20, parameters=parameters,
        organ_semantics={
            name: describe_hashed_transducer(
                w_max=parameters["w_max"], norm_init=parameters["norm_init"],
                refracted_strength=parameters["refracted_strength"],
                state_mode=parameters["state_mode"],
                predict_gain=arm["predict_gain"],
            )
            for name, arm in ARMS.items()
        },
        measure=experiment)
    print(path)


if __name__ == "__main__":
    main()
