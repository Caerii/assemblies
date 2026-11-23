# R1d evaluation audit — decoder invariance and zero-shot controls

Date: 2026-11-23. Branch: `research/multimodal-continual-assemblies`.

## Two corrected scientific confounds

**Changing candidates is not forgetting.** Previously `familiar_before_top1`
ranked each stored binding against only the already-trained pairs, while
`familiar_after_top1` ranked it against all pairs, including newly acquired
distractors. This can create an apparent retention loss with unchanged
synapses or probe activity. The code now snapshots the original candidate
dictionary and evaluates:

- `familiar_before_top1`: original candidate set, before new bindings;
- `familiar_after_top1`: SAME original candidate set, after new bindings;
- `familiar_after_expanded_top1`: after new bindings, ALL candidates;
- `old_retention_delta`: same-decoder change, a defensible forgetting proxy;
- `candidate_expansion_penalty`: expanded minus fixed-decoder accuracy.

A separate unit test constructs an unchanged probe and target, adds a
distractor with a tied representation and lower ID, and confirms that the
decoder's rank changes despite no change in the probe. This is a diagnostic,
not a native-result claim.

**Ties are not evidence of discrimination.** The prior top-1 evaluator
awarded a correct prediction if deterministic tie-breaking happened to
pick the target. `score(...)["top1"]` now requires a *strictly positive*
margin above the best other candidate; a new `tie_broken_top1` retains
the old diagnostic behavior. Both versions remain visible to callers.

## R1e zero-shot calibration control

`zero_shot_baselines.py` constructs sparse lexical codes and two fixed,
invertible role permutations. Given any lexical code and role, the oracle
can encode then decode the representation by applying the inverse role
permutation. **It does not train on specific role/content combinations.**
The paired negative control is an exact-key dictionary containing only the
training pairs.

With defaults (n=1024, k=20, 16 contents, 2 roles, checkerboard split) and
seeds [0,1,17,42], a local independent copy produced:

| Control | Trained pairs | Held-out pairs |
|---|---:|---:|
| Fixed role-factorization oracle | 100% | 100% |
| Exact-key pair memorizer | 100% | 0% |

This is expected by construction, and establishes the *task distinction*:
memorization of trained pairs does not solve unseen role/content combinations.
It **does not** show that neural assemblies learned role factorization.
The oracle receives hand-specified bijective transforms, not learned
synapses or NEMO circuits.

## Local validation

A standalone local mirror of the pure structural calibration and a
decoder-artifact check passed **9 tests** under pytest. This was not a
full repository checkout. R1b/R1c/R1d native Brain tests remain unexecuted
in this environment; the Rust backend is likewise uncompiled.

```sh
uv run python -m pytest -q \
  research/multimodal/test_zero_shot_baselines.py \
  research/multimodal/test_role_recombination.py
uv run python -m research.multimodal.zero_shot_baselines
uv run python -m research.multimodal.role_recombination \
  --seeds 11 17 23 --contents 8 --output /tmp/r1d-native.json
```

## Next falsification gate

To claim actual zero-shot compositionality, train the native assembly model
on role structure and atomic lexical codes, WITHOUT training held-out
role/content pairs, and evaluate those pairs with the held-out target
constructed independently of the inference. Compare against role-only,
lexical-overlap, indexed retrieval, exact-key pair memory, and an explicit
factorized symbolic baseline. Measure strict top-1 and margin, collapsing
assemblies, and retained known-pair accuracy. A model that only reads
newly bound one-shot pairs has not demonstrated zero-shot transfer.
