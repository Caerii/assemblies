"""R1d: one-shot held-out role/content binding with native assembly calculus.

A held-out combination receives exactly ONE binding operation after its
lexical assembly is established. Readout uses the library's read_binding,
never an independently written inference schedule.

Not evidence of systematic generalization until native runs, baseline
comparison, and controls pass. Run from repo root:
  uv run python -m research.multimodal.role_recombination --output r1d.json
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class Protocol:
    n: int = 600
    k: int = 24
    p: float = 0.1
    beta: float = 0.1
    lex_rounds: int = 8
    tail_rounds: int = 1
    seeds: tuple[int, ...] = (11, 17, 23)
    contents: int = 8
    roles: tuple[str, ...] = ("SUBJ", "OBJ")

    def validate(self):
        if not (1 < self.k < self.n and 0 < self.p <= 1 and 0 < self.beta < 1):
            raise ValueError("invalid neural geometry")
        if self.lex_rounds < 1 or self.tail_rounds < 0 or self.contents < 4:
            raise ValueError("invalid learning budget")
        if self.contents % 2:
            raise ValueError("contents must be even for balanced held-out split")
        if len(self.roles) != 2 or len(set(self.roles)) != 2:
            raise ValueError("exactly two distinct roles required")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("unique seeds required")
        return self


def split(contents: int, roles: tuple[str, str]):
    """Balanced checkerboard: every content AND every role occurs in training."""
    train = [(i, role) for i in range(contents) for j, role in enumerate(roles)
             if (i + j) % 2 == 0]
    heldout = [(i, role) for i in range(contents) for j, role in enumerate(roles)
               if (i + j) % 2 == 1]
    assert not set(train) & set(heldout)
    assert len(train) == len(heldout) == contents
    return train, heldout


def _native():
    from neural_assemblies.core.brain import Brain
    from neural_assemblies.assembly_calculus import project, bind, read_binding
    return Brain, project, bind, read_binding


def _ids(assembly, k: int):
    x = np.sort(np.asarray(assembly.neuron_ids, dtype=np.uint32))
    if len(x) != k or len(np.unique(x)) != k:
        raise AssertionError("invalid native stable neuron IDs")
    return x


def score(probe, references, target, k, n):
    """Rank within one role; never compare winners across different areas."""
    x = _ids(probe, k)
    rankings = sorted(((len(np.intersect1d(x, _ids(ref, k), assume_unique=True)),
                        key) for key, ref in references.items()),
                      key=lambda pair: (-pair[0], pair[1]))
    target_hits = len(np.intersect1d(x, _ids(references[target], k),
                                    assume_unique=True))
    competitor = max((hits for hits, key in rankings if key != target), default=-1)
    margin = target_hits - competitor
    return {
        # A tie is NOT successful discriminative retrieval, even if an
        # arbitrary deterministic order happens to favor the target.
        "top1": int(rankings[0][1] == target and margin > 0),
        "tie_broken_top1": int(rankings[0][1] == target),
        "top1_margin": int(margin),
        "overlap": float(target_hits / k),
        "chance_overlap": float(k / n),
        "rankings": rankings[:3],
    }


def distinct_fraction(assemblies, k):
    codes = [tuple(map(int, _ids(a, k))) for a in assemblies]
    return float(len(set(codes)) / len(codes)) if codes else 0.0


def trial(seed: int, cfg: Protocol):
    Brain, project, bind, read_binding = _native()
    train, heldout = split(cfg.contents, cfg.roles)
    brain = Brain(engine="numpy_exact", seed=seed, p=cfg.p, norm_init=False)
    brain.add_area("LEX", n=cfg.n, k=cfg.k, beta=cfg.beta)
    for role in cfg.roles:
        brain.add_area(role, n=cfg.n, k=cfg.k, beta=cfg.beta)
    for i in range(cfg.contents):
        brain.add_stimulus(f"word_{i}", cfg.k)

    # Establish all lexical assemblies BEFORE binding. Their stable neuron
    # IDs are replayed directly, preventing lexical drift from contaminating
    # the role-binding experiment.
    lexical = {}
    for i in range(cfg.contents):
        brain.clear_activity(["LEX"])
        lexical[i] = project(brain, f"word_{i}", "LEX",
                             rounds=cfg.lex_rounds, recurrent=True)

    lexical_distinctness = distinct_fraction(list(lexical.values()), cfg.k)
    references = {role: {} for role in cfg.roles}
    for i, role in train:
        brain.clear_activity([role])
        references[role][i] = bind(
            brain, "LEX", role, source_assembly=lexical[i],
            tail_rounds=cfg.tail_rounds)

    # Read is conducted after training, not interleaved with writes.
    # An unbound pair can still retrieve something, but must NOT be
    # credited as success simply for producing k winners.
    def read(i, role):
        with brain.frozen():
            return read_binding(brain, "LEX", role,
                                source_assembly=lexical[i],
                                tail_rounds=cfg.tail_rounds)

    pretest = []
    for i, role in heldout:
        got = read(i, role)
        candidates = references[role]
        nearest = max((len(np.intersect1d(_ids(got, cfg.k),
                                          _ids(ref, cfg.k))), j)
                      for j, ref in candidates.items())
        pretest.append({"content": i, "role": role, "max_old_overlap": nearest[0] / cfg.k})

    initial_role_distinctness = {
        role: distinct_fraction(list(references[role].values()), cfg.k)
        for role in cfg.roles
    }
    # Freeze the decoder's candidate universe BEFORE adding held-out pairs.
    # Otherwise a lower post-learning top-1 score can be caused simply by
    # adding distractors, not by forgetting a learned association.
    original_candidates = {role: dict(references[role]) for role in cfg.roles}
    old_before = {(i, role): score(read(i, role), original_candidates[role], i, cfg.k, cfg.n)
                  for i, role in train}

    # One exposure per held-out combination, with no extra sentence training.
    for i, role in heldout:
        brain.clear_activity([role])
        references[role][i] = bind(
            brain, "LEX", role, source_assembly=lexical[i],
            tail_rounds=cfg.tail_rounds)

    final_role_distinctness = {
        role: distinct_fraction(list(references[role].values()), cfg.k)
        for role in cfg.roles
    }
    # One read per item, two decoders: fixed candidates isolate retrieval
    # drift; expanded candidates measure actual post-learning discrimination.
    old_probes = {(i, role): read(i, role) for i, role in train}
    old_after_fixed = {(i, role): score(old_probes[i, role],
                                      original_candidates[role], i, cfg.k, cfg.n)
                       for i, role in train}
    old_after_expanded = {(i, role): score(old_probes[i, role],
                                         references[role], i, cfg.k, cfg.n)
                          for i, role in train}
    novel = {(i, role): score(read(i, role), references[role], i, cfg.k, cfg.n)
             for i, role in heldout}

    def mean(rows, field):
        return float(np.mean([v[field] for v in rows.values()]))

    # The exact-key baseline stores training pairs and, after the single
    # exposure, the held-out pairs. It is a deliberately strong memory control;
    # not a parametric compositional-generalization baseline.
    return {
        "seed": seed, "train": train, "heldout": heldout,
        "lexical_distinctness": lexical_distinctness,
        "initial_role_distinctness": initial_role_distinctness,
        "final_role_distinctness": final_role_distinctness,
        "lexicon_digest": hashlib.sha256(b"".join(
            _ids(lexical[i], cfg.k).tobytes()
            for i in range(cfg.contents))).hexdigest(),
        "pretest": pretest,
        "familiar_before_top1": mean(old_before, "top1"),
        "familiar_after_top1": mean(old_after_fixed, "top1"),
        "familiar_after_expanded_top1": mean(old_after_expanded, "top1"),
        "heldout_after_one_binding_top1": mean(novel, "top1"),
        "heldout_after_one_binding_overlap": mean(novel, "overlap"),
        "old_retention_delta": mean(old_after_fixed, "top1") - mean(old_before, "top1"),
        "candidate_expansion_penalty": mean(old_after_expanded, "top1") - mean(old_after_fixed, "top1"),
        "exact_key_baseline_after_one_exposure_top1": 1.0,
        "per_item": {
            f"{role}/{i}": {"phase": "heldout" if (i, role) in heldout else "train",
                            "before": old_before.get((i, role)),
                            "after": (novel if (i, role) in heldout else old_after_fixed)[(i, role)],
                            "after_expanded": old_after_expanded.get((i, role))}
            for i in range(cfg.contents) for role in cfg.roles
        },
    }


def run(cfg: Protocol):
    cfg.validate()
    runs = [trial(seed, cfg) for seed in cfg.seeds]
    return {
        "status": "native_role_binding_one_shot_not_zero_shot",
        "config": asdict(cfg), "runs": runs,
        "summary": {
            key: float(np.mean([r[key] for r in runs]))
            for key in ("familiar_before_top1", "familiar_after_top1",
                        "heldout_after_one_binding_top1",
                        "heldout_after_one_binding_overlap", "old_retention_delta",
                        "familiar_after_expanded_top1", "candidate_expansion_penalty")
        },
        "limitations": [
            "Held-out role/content pairs receive one binding exposure; this is NOT zero-shot.",
            "Lexical assemblies are pre-established using native project; no unseen nouns.",
            "Reference snapshots are created during binding; self-retrieval may be easier than external semantic readout.",
            "Same-role top-1 identity does not demonstrate sentence-level compositional semantics.",
            "Distinctness metrics expose attractor collapse but do not prove meaningful semantic separation.",
            "Fixed-candidate retention isolates decoder expansion; expanded-candidate scores measure changed discriminability.",
            "Exact-key episodic memory is a strong positive control but lacks budget matching.",
            "Native engine execution and results are unverified until run in full checkout.",
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[11, 17, 23])
    ap.add_argument("--contents", type=int, default=8)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    output = json.dumps(run(Protocol(seeds=tuple(args.seeds),
                                     contents=args.contents)),
                        indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output)


if __name__ == "__main__":
    main()
