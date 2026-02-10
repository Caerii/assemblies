"""
Parser Scaling Laws Experiment (Tier 1)

How do assembly calculus parser capabilities scale with network parameters
(n, k, vocabulary size, sentence complexity)?

Hypotheses:
H1: Classification accuracy increases monotonically with n, plateauing ~n=20000.
H2: Assembly width k=sqrt(n) is optimal -- deviations reduce accuracy.
H3: Role-binding fidelity degrades gracefully with vocabulary size.
H4: The parser handles sentence lengths up to ~15 words before role confusion.

Protocol:
1. n-scaling: Fix k=100, vary n. Measure classification accuracy.
2. k-scaling: Fix n=10000, vary k. Measure classification accuracy.
3. Vocabulary scaling: Fix n=10000, k=100, vary vocab size.
4. Length scaling: Fix n=10000, k=100, vary sentence length.

References:
- Papadimitriou et al., PNAS 117(25):14464-14472, 2020
- Mitropolsky & Papadimitriou, "Simulated Language Acquisition", 2025
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from research.experiments.base import ExperimentBase, ExperimentResult, summarize, ttest_vs_null
from src.assembly_calculus.emergent import EmergentParser
from src.assembly_calculus.emergent.grounding import GroundingContext
from src.assembly_calculus.emergent.training_data import GroundedSentence


@dataclass
class ScalingConfig:
    """Configuration for parser scaling experiments."""
    p: float = 0.05
    beta: float = 0.1
    rounds: int = 10
    n_seeds: int = 5
    n_values: tuple = (2000, 5000, 10000, 20000)
    k_values: tuple = (25, 50, 100, 200)
    vocab_sizes: tuple = (10, 20, 50, 100)
    sentence_lengths: tuple = (3, 5, 7, 10, 13)


# -- Word pools for vocabulary construction ------------------------------------

_NOUNS = [
    ("dog", ["DOG", "ANIMAL"]), ("cat", ["CAT", "ANIMAL"]),
    ("bird", ["BIRD", "ANIMAL"]), ("boy", ["BOY", "PERSON"]),
    ("girl", ["GIRL", "PERSON"]), ("ball", ["BALL", "OBJECT"]),
    ("book", ["BOOK", "OBJECT"]), ("food", ["FOOD", "OBJECT"]),
    ("table", ["TABLE", "FURNITURE"]), ("car", ["CAR", "OBJECT"]),
    ("fish", ["FISH", "ANIMAL"]), ("tree", ["TREE", "PLANT"]),
    ("house", ["HOUSE", "BUILDING"]), ("horse", ["HORSE", "ANIMAL"]),
    ("apple", ["APPLE", "FOOD"]), ("chair", ["CHAIR", "FURNITURE"]),
    ("water", ["WATER", "SUBSTANCE"]), ("bread", ["BREAD", "FOOD"]),
    ("cup", ["CUP", "OBJECT"]), ("door", ["DOOR", "OBJECT"]),
    ("flower", ["FLOWER", "PLANT"]), ("ship", ["SHIP", "VEHICLE"]),
    ("star", ["STAR", "OBJECT"]), ("river", ["RIVER", "NATURE"]),
    ("mouse", ["MOUSE", "ANIMAL"]), ("cake", ["CAKE", "FOOD"]),
    ("stone", ["STONE", "OBJECT"]), ("cloud", ["CLOUD", "NATURE"]),
    ("box", ["BOX", "OBJECT"]), ("baby", ["BABY", "PERSON"]),
]
_VERBS = [
    ("runs", ["RUNNING", "MOTION"]), ("sees", ["SEEING", "PERCEPTION"]),
    ("eats", ["EATING", "CONSUMPTION"]), ("chases", ["CHASING", "PURSUIT"]),
    ("plays", ["PLAYING", "ACTION"]), ("sleeps", ["SLEEPING", "REST"]),
    ("reads", ["READING", "COGNITION"]), ("finds", ["FINDING", "PERCEPTION"]),
    ("likes", ["LIKING", "EMOTION"]), ("makes", ["MAKING", "CREATION"]),
    ("hits", ["HITTING", "ACTION"]), ("drops", ["DROPPING", "MOTION"]),
    ("holds", ["HOLDING", "POSSESSION"]), ("moves", ["MOVING", "MOTION"]),
    ("opens", ["OPENING", "ACTION"]), ("pulls", ["PULLING", "MOTION"]),
    ("pushes", ["PUSHING", "MOTION"]), ("takes", ["TAKING", "POSSESSION"]),
    ("throws", ["THROWING", "MOTION"]), ("watches", ["WATCHING", "PERCEPTION"]),
]
_ADJS = [
    ("big", ["SIZE", "BIG"]), ("small", ["SIZE", "SMALL"]),
    ("red", ["COLOR", "RED"]), ("fast", ["SPEED", "FAST"]),
    ("happy", ["EMOTION", "HAPPY"]), ("old", ["AGE", "OLD"]),
    ("new", ["AGE", "NEW"]), ("soft", ["TEXTURE", "SOFT"]),
    ("tall", ["SIZE", "TALL"]), ("dark", ["COLOR", "DARK"]),
]
_DETS = ["the", "a"]

_MOD_TO_CAT = {
    "visual": "NOUN", "motor": "VERB", "properties": "ADJ",
    "spatial": "PREP", "social": "PRON", "temporal": "ADV", "none": "DET",
}


def _build_vocab(n_nouns: int, n_verbs: int, n_adj: int = 3,
                 n_det: int = 2) -> Dict[str, GroundingContext]:
    """Build a vocabulary of the given size from word pools."""
    v: Dict[str, GroundingContext] = {}
    for w, f in _NOUNS[:n_nouns]:
        v[w] = GroundingContext(visual=f)
    for w, f in _VERBS[:n_verbs]:
        v[w] = GroundingContext(motor=f)
    for w, f in _ADJS[:n_adj]:
        v[w] = GroundingContext(properties=f)
    for w in _DETS[:n_det]:
        v[w] = GroundingContext()
    return v


def _vocab_labels(vocab: Dict[str, GroundingContext]) -> Dict[str, str]:
    """Map each word to its expected POS category label."""
    return {w: _MOD_TO_CAT.get(c.dominant_modality, "DET")
            for w, c in vocab.items()}


def _gen_svo(vocab: Dict[str, GroundingContext], rng: np.random.Generator,
             n: int = 30) -> List[GroundedSentence]:
    """Generate simple SVO training sentences (intransitive, transitive, adj)."""
    nouns = [w for w, c in vocab.items() if c.dominant_modality == "visual"]
    verbs = [w for w, c in vocab.items() if c.dominant_modality == "motor"]
    adjs = [w for w, c in vocab.items() if c.dominant_modality == "properties"]
    dets = [w for w, c in vocab.items() if c.dominant_modality == "none"] or ["the"]
    if not nouns or not verbs:
        return []
    pick = lambda lst: lst[rng.integers(len(lst))]
    sents: List[GroundedSentence] = []
    for i in range(n):
        pat = i % 3
        if pat == 0:
            ws = [pick(dets), pick(nouns), pick(verbs)]
            rs = [None, "agent", "action"]
        elif pat == 1:
            n1, n2 = pick(nouns), pick(nouns)
            for _ in range(5):
                if n2 != n1 or len(nouns) == 1:
                    break
                n2 = pick(nouns)
            ws = [pick(dets), n1, pick(verbs), pick(dets), n2]
            rs = [None, "agent", "action", None, "patient"]
        else:
            if adjs:
                ws = [pick(dets), pick(adjs), pick(nouns), pick(verbs)]
                rs = [None, None, "agent", "action"]
            else:
                ws = [pick(dets), pick(nouns), pick(verbs)]
                rs = [None, "agent", "action"]
        ctx = [vocab.get(w, GroundingContext()) for w in ws]
        sents.append(GroundedSentence(words=ws, contexts=ctx, roles=rs))
    return sents


def _gen_length(vocab: Dict[str, GroundingContext], rng: np.random.Generator,
                length: int, n: int = 8
                ) -> Tuple[List[GroundedSentence], List[Dict[str, str]]]:
    """Generate sentences of a specific target word length."""
    nouns = [w for w, c in vocab.items() if c.dominant_modality == "visual"]
    verbs = [w for w, c in vocab.items() if c.dominant_modality == "motor"]
    adjs = [w for w, c in vocab.items() if c.dominant_modality == "properties"]
    dets = [w for w, c in vocab.items() if c.dominant_modality == "none"] or ["the"]
    if not nouns or not verbs:
        return [], []
    pick = lambda lst: lst[rng.integers(len(lst))]
    sents, role_maps = [], []
    for _ in range(n):
        subj, obj = pick(nouns), pick(nouns)
        for _ in range(5):
            if obj != subj or len(nouns) == 1:
                break
            obj = pick(nouns)
        v = pick(verbs)
        ws = [pick(dets), subj, v, pick(dets), obj]
        rs: List[Optional[str]] = [None, "agent", "action", None, "patient"]
        rmap = {subj: "AGENT", v: "ACTION", obj: "PATIENT"}
        while len(ws) < length:
            rem = length - len(ws)
            if rem >= 2 and adjs:
                ws.extend([pick(dets), pick(adjs)]); rs.extend([None, None])
            if length - len(ws) >= 1:
                ws.append(pick(nouns)); rs.append(None)
            else:
                break
        ws, rs = ws[:length], rs[:length]
        ctx = [vocab.get(w, GroundingContext()) for w in ws]
        sents.append(GroundedSentence(words=ws, contexts=ctx, roles=rs))
        role_maps.append(rmap)
    return sents, role_maps


# -- Trial runner --------------------------------------------------------------


def _run_trial(n: int, k: int, p: float, beta: float, rounds: int,
               vocab: Dict[str, GroundingContext],
               train_sents: List[GroundedSentence],
               labels: Dict[str, str], seed: int) -> Dict[str, float]:
    """Train a parser and measure classification + role accuracy for one seed."""
    parser = EmergentParser(n=n, k=k, p=p, beta=beta, seed=seed,
                            rounds=rounds, vocabulary=vocab)
    parser.train(sentences=train_sents)
    # Classification accuracy
    correct = total = 0
    for word, expected in labels.items():
        grounding = parser.word_grounding.get(word)
        predicted, _ = parser.classify_word(word, grounding=grounding)
        correct += int(predicted == expected)
        total += 1
    cls_acc = correct / max(total, 1)
    # Role accuracy on first few training sentences
    rc = rt = 0
    for sent in train_sents[:5]:
        result = parser.parse(sent.words)
        roles = result.get("roles", {})
        for w, er in zip(sent.words, sent.roles):
            if er is None:
                continue
            rc += int(roles.get(w) == er.upper())
            rt += 1
    return {"classification_accuracy": cls_acc,
            "role_accuracy": rc / max(rt, 1)}


# -- Main experiment -----------------------------------------------------------


class ParserScalingLawsExperiment(ExperimentBase):
    """Test how parser capabilities scale with n, k, vocab size, sentence length."""

    def __init__(self, results_dir: Path = None, seed: int = 42,
                 verbose: bool = True):
        super().__init__(
            name="parser_scaling_laws", seed=seed,
            results_dir=(results_dir or
                         Path(__file__).parent.parent.parent / "results" / "applications"),
            verbose=verbose,
        )

    def run(self, config: Optional[ScalingConfig] = None, **kwargs) -> ExperimentResult:
        """Run all four scaling sweeps and return structured results."""
        if config is None:
            config = ScalingConfig()
        self._start_timer()
        seeds = [self.seed + i for i in range(config.n_seeds)]

        self.log("=" * 60)
        self.log("Parser Scaling Laws Experiment")
        self.log(f"  p={config.p}, beta={config.beta}, rounds={config.rounds}")
        self.log(f"  n_seeds={config.n_seeds}")
        self.log("=" * 60)

        metrics: Dict[str, Any] = {}
        base_vocab = _build_vocab(n_nouns=8, n_verbs=6, n_adj=3)
        base_labels = _vocab_labels(base_vocab)
        n_categories = len(set(base_labels.values()))
        chance = 1.0 / max(n_categories, 1)

        # ---- Sweep 1: n-scaling (area size) ----
        self.log("\n--- Sweep 1: n-scaling ---")
        n_results = []
        for n_val in config.n_values:
            accs, roles = [], []
            for s in seeds:
                t = _run_trial(n_val, 100, config.p, config.beta, config.rounds,
                               base_vocab, _gen_svo(base_vocab, np.random.default_rng(s)),
                               base_labels, s)
                accs.append(t["classification_accuracy"])
                roles.append(t["role_accuracy"])
            row = {"n": n_val, "classification": summarize(accs),
                   "role_accuracy": summarize(roles),
                   "test_vs_chance": ttest_vs_null(accs, chance)}
            n_results.append(row)
            self.log(f"  n={n_val:5d}: acc={row['classification']['mean']:.3f}"
                     f"+/-{row['classification']['sem']:.3f}  "
                     f"role={row['role_accuracy']['mean']:.3f}")
        metrics["n_scaling"] = n_results

        n_means = [r["classification"]["mean"] for r in n_results]
        h1 = all(n_means[i] <= n_means[i+1] + 0.01 for i in range(len(n_means)-1))
        metrics["h1_monotonic_increase"] = h1
        self.log(f"  H1 (monotonic): {h1}")

        # ---- Sweep 2: k-scaling (assembly width) ----
        self.log("\n--- Sweep 2: k-scaling ---")
        k_results = []
        for k_val in config.k_values:
            accs, roles = [], []
            for s in seeds:
                t = _run_trial(10000, k_val, config.p, config.beta, config.rounds,
                               base_vocab, _gen_svo(base_vocab, np.random.default_rng(s)),
                               base_labels, s)
                accs.append(t["classification_accuracy"])
                roles.append(t["role_accuracy"])
            tag = " <-- sqrt(n)" if k_val == int(np.sqrt(10000)) else ""
            row = {"k": k_val, "sqrt_n": int(np.sqrt(10000)),
                   "classification": summarize(accs),
                   "role_accuracy": summarize(roles),
                   "test_vs_chance": ttest_vs_null(accs, chance)}
            k_results.append(row)
            self.log(f"  k={k_val:3d}: acc={row['classification']['mean']:.3f}"
                     f"+/-{row['classification']['sem']:.3f}{tag}")
        metrics["k_scaling"] = k_results

        k_map = {r["k"]: r["classification"]["mean"] for r in k_results}
        best_k = max(k_map, key=k_map.get)
        metrics["h2_sqrt_n_optimal"] = (best_k == int(np.sqrt(10000)))
        metrics["h2_best_k"] = best_k
        self.log(f"  H2 (sqrt(n) optimal): {metrics['h2_sqrt_n_optimal']}  "
                 f"(best k={best_k})")

        # ---- Sweep 3: Vocabulary scaling ----
        self.log("\n--- Sweep 3: Vocabulary scaling ---")
        v_results = []
        for vsize in config.vocab_sizes:
            nn = max(3, int(vsize * 0.40))
            nv = max(2, int(vsize * 0.30))
            na = max(1, int(vsize * 0.15))
            nd = max(1, vsize - nn - nv - na)
            sv = _build_vocab(nn, nv, na, nd)
            sl = _vocab_labels(sv)
            accs, roles = [], []
            for s in seeds:
                t = _run_trial(10000, 100, config.p, config.beta, config.rounds,
                               sv, _gen_svo(sv, np.random.default_rng(s),
                                            n=max(30, len(sv)*2)),
                               sl, s)
                accs.append(t["classification_accuracy"])
                roles.append(t["role_accuracy"])
            nc = len(set(sl.values()))
            row = {"target_size": vsize, "actual_size": len(sv),
                   "classification": summarize(accs),
                   "role_accuracy": summarize(roles),
                   "test_vs_chance": ttest_vs_null(accs, 1.0/max(nc, 1))}
            v_results.append(row)
            self.log(f"  vocab={len(sv):3d}: acc={row['classification']['mean']:.3f}"
                     f"+/-{row['classification']['sem']:.3f}  "
                     f"role={row['role_accuracy']['mean']:.3f}")
        metrics["vocab_scaling"] = v_results

        v_accs = [r["classification"]["mean"] for r in v_results]
        if len(v_accs) >= 2:
            max_drop = max(v_accs[i] - v_accs[i+1] for i in range(len(v_accs)-1))
        else:
            max_drop = 0.0
        metrics["h3_graceful_degradation"] = max_drop < 0.30
        metrics["h3_max_step_drop"] = max_drop
        self.log(f"  H3 (graceful): {metrics['h3_graceful_degradation']}  "
                 f"(max drop={max_drop:.3f})")

        # ---- Sweep 4: Sentence length scaling ----
        self.log("\n--- Sweep 4: Length scaling ---")
        l_results = []
        lv = _build_vocab(n_nouns=10, n_verbs=8, n_adj=5)
        ll = _vocab_labels(lv)
        for slen in config.sentence_lengths:
            parse_vals, role_vals = [], []
            for s in seeds:
                parser = EmergentParser(n=10000, k=100, p=config.p, beta=config.beta,
                                        seed=s, rounds=config.rounds, vocabulary=lv)
                parser.train(sentences=_gen_svo(lv, np.random.default_rng(s), n=40))
                tsents, tmaps = _gen_length(lv, np.random.default_rng(s+1000), slen)
                sc = rc = rt = 0
                for sent, rmap in zip(tsents, tmaps):
                    res = parser.parse(sent.words)
                    cats = res.get("categories", {})
                    ct = sum(1 for w in sent.words if w in ll)
                    cc = sum(1 for w in sent.words if cats.get(w) == ll.get(w))
                    if ct > 0 and cc / ct >= 0.7:
                        sc += 1
                    for w, er in rmap.items():
                        rc += int(res.get("roles", {}).get(w) == er)
                        rt += 1
                parse_vals.append(sc / max(len(tsents), 1))
                role_vals.append(rc / max(rt, 1))
            row = {"length": slen,
                   "parse_correctness": summarize(parse_vals),
                   "role_accuracy": summarize(role_vals)}
            l_results.append(row)
            self.log(f"  len={slen:2d}: parse={row['parse_correctness']['mean']:.3f}"
                     f"+/-{row['parse_correctness']['sem']:.3f}  "
                     f"role={row['role_accuracy']['mean']:.3f}")
        metrics["length_scaling"] = l_results

        h4_max = None
        for r in l_results:
            if r["parse_correctness"]["mean"] >= 0.5:
                h4_max = r["length"]
        metrics["h4_max_parseable_length"] = h4_max
        self.log(f"  H4 (max length >=50%): {h4_max}")

        # ---- Summary ----
        duration = self._stop_timer()
        self.log(f"\n{'='*60}")
        self.log("PARSER SCALING LAWS SUMMARY")
        self.log(f"  H1 (n monotonic):         {metrics['h1_monotonic_increase']}")
        self.log(f"  H2 (k=sqrt(n) optimal):   {metrics['h2_sqrt_n_optimal']} "
                 f"(best k={metrics['h2_best_k']})")
        self.log(f"  H3 (graceful degradation): {metrics['h3_graceful_degradation']} "
                 f"(max drop={metrics['h3_max_step_drop']:.3f})")
        self.log(f"  H4 (max parseable length): {metrics['h4_max_parseable_length']}")
        self.log(f"  Duration: {duration:.1f}s")
        self.log("=" * 60)

        return ExperimentResult(
            experiment_name=self.name,
            parameters={"p": config.p, "beta": config.beta,
                        "rounds": config.rounds, "n_seeds": config.n_seeds,
                        "n_values": list(config.n_values),
                        "k_values": list(config.k_values),
                        "vocab_sizes": list(config.vocab_sizes),
                        "sentence_lengths": list(config.sentence_lengths),
                        "seed": self.seed},
            metrics=metrics, raw_data={}, duration_seconds=duration,
        )


def main():
    ap = argparse.ArgumentParser(description="Parser Scaling Laws (Tier 1)")
    ap.add_argument("--quick", action="store_true",
                    help="Quick run: fewer seeds, smaller sweeps")
    args = ap.parse_args()

    if args.quick:
        cfg = ScalingConfig(n_seeds=3,
                            n_values=(2000, 5000, 10000),
                            k_values=(50, 100, 200),
                            vocab_sizes=(10, 20, 50),
                            sentence_lengths=(3, 5, 7, 10))
    else:
        cfg = ScalingConfig()

    exp = ParserScalingLawsExperiment(verbose=True)
    result = exp.run(config=cfg)
    exp.save_result(result, "_quick" if args.quick else "")
    print(f"\nTotal time: {result.duration_seconds:.1f}s")


if __name__ == "__main__":
    main()
