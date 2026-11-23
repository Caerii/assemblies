# R1g native zero-shot lower-bound adapter

**Status:** committed; native execution pending.

`native_zero_shot.py` connects the R1f benchmark to the repository's
actual `Brain` API. It fully materializes lexical and role areas,
activates k-of-n lexical codes as stable neuron IDs, trains the role
projection using the native projection call, and evaluates inside nested
`read_only` and `frozen` scopes.

**Crucial distinction:** This adapter intentionally discards the teacher's
role-conditioned target during training. It is an **unsupervised projection
lower bound**, not a supervised Hebbian binding implementation and NOT
evidence that the network learns cyclic transformations.

The code must be executed in a full checkout to check that
`Brain.materialize_area`, `activate_assembly`, `Brain.project`, and
`Assembly.from_area` interoperate correctly with the chosen engine.
A mock-level regression test verifies scheduling and non-leakage but
cannot establish native numerical correctness.

Commands:

```sh
uv run python -m pytest -q research/multimodal/test_native_zero_shot.py
uv run python -m research.multimodal.native_zero_shot \
  --seed 42 --family cyclic_shift --n 512 --k 16 --contents 12
```

**Next real milestone:** design a native teacher-guided training schedule
that provides only observed (cue, target, role) pairs to `Brain` and
learns a reusable role operator. Do not expose the underlying role
permutation to the model. Before claiming zero-shot composition, prove
that the learned operator improves held-out exact recovery over the
unprivileged `ShiftRoleLearner`, the sparse Hebbian baseline, and
the unsupervised native lower bound. If the network cannot express a
shift from these examples, report a negative result and investigate
whether the representation or training rule lacks the requisite
inductive bias.
