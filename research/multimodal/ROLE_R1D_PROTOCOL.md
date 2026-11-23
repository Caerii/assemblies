# R1d preregistration — one-shot role/content recombination

**Status:** source + mock protocol tests committed; native numerical experiment has
not run in a full checkout. No results are claimed.
**Branch:** `research/multimodal-continual-assemblies`
**Entrypoint:** `python -m research.multimodal.role_recombination`

## Research question

Can a native assembly system preserve previously learned role bindings while
acquiring a held-out role/content pair in one local binding operation? We
test two different things, and MUST keep them separate:

1. Reuse of already learned lexical content within a new syntactic role
   (one-shot binding; allowed one new write)
2. *True systematic zero-shot transfer* (NO write of the held-out
   combination; a much harder, later test that this experiment does NOT claim)

## Split construction

With eight lexical items and two target role areas (SUBJ, OBJ), use
a balanced checkerboard: item `i` is trained for role `j` iff
`(i+j)%2 == 0`. Every item appears in one learned role; both role
areas are trained on four known items. The other eight specific pairs
are held out, then one-shot bound.

All lexical assemblies are learned first, independently of the roles, and
snapshotted as stable neuron IDs. The native library's `bind()` replays
those IDs for training, and `read_binding()` is the common inference
schedule. No resetting connections, no hand-rolled Python-side synapses.
This is important: those are established failure modes in the repository.

## Measurement steps, in order

1. Create one `Brain(engine="numpy_exact")` per seed with a LEX area and
   two role areas; register all lexical stimuli.
2. Learn each lexical assembly with a recurrent stimulus schedule.
3. Write the eight training role/content bindings using native `bind`.
4. Probe held-out pairs WITHOUT writing them; record their overlap against
   existing role bindings. Treat spontaneous output as negative control, not
   successful generalization.
5. Probe familiar bindings with native `read_binding` under `brain.frozen()`.
6. Bind each held-out pair with exactly one `bind` operation.
7. Probe all role/content pairs and measure familiar retention and novel
   top-1 recognition.

`read_binding()` itself uses `brain.read_only()` so measurement should
not recruit new synapses; the explicit outer `frozen()` additionally
disables plasticity. All references and predictions use native Assembly
snapshots (stable neuron IDs), not compact `Area.winners`.

## Registered metrics

- Familiar role binding top-1 BEFORE and AFTER acquiring held-out pairs.
- Held-out top-1 and normalized k-overlap after one binding exposure.
- Familiar retention delta.
- Fraction of distinct lexical winner sets, and per-role binding distinctness before and after the new writes. Identical attractors are failures, not successful compositions.
- Pre-exposure maximum overlap against existing role memories (unbound
  control).
- Exact-key episodic dictionary baseline: perfect for uniquely keyed
  pairs after the single write, but its storage budget is not matched.
- Paired seeds and all per-item results, including failures.

Tie-break by content index; a tie does not get charitable success.
The decoder ranks only inside a role area, NEVER by overlap across different
roles (different coordinate spaces). An overlap score of 1.0 for the
newly written snapshot is not enough; discriminability against competitors
is essential.

## Execution

```sh
uv run python -m pytest -q research/multimodal/test_role_recombination.py
uv run python -m research.multimodal.role_recombination \
    --seeds 11 17 23 --contents 8 --output /tmp/role-r1d.json
```

Record full repo commit, Python/NumPy versions, individual seeds, failures,
wall time, and output hash. Mock tests test *experimental scheduling*,
not correctness of native numerical dynamics.

## Interpretive constraints

- This is NOT unseen-noun generalization; every noun had a pretrained
  lexical assembly. Not grammatical acquisition or vision grounding.
- It does not test unseen sentence structure, causal relations, or recursive
  hierarchy.
- It provides no fair same-memory-budget comparison to indexed episodic
  memory or a symbolic role store.
- The role area can form an attractor that gives artificially high
  overlap; the pre-exposure unbound control and competing-memory rank matter.
- Reusing `bind` for learning and `read_binding` for measurement is
  intentional, but distinct semantics (asymmetric plasticity or borrowed
  clamps) must be audited on native execution.

**Next scientific gate:** after native execution, include a true zero-shot
role-transfer arm and held-out *sentences* with fixed content/role bindings,
plus independent semantic readout and retrieval cost/performance-matched
baselines. If this experiment only succeeds at re-reading what it wrote,
report that honestly and do not relabel it compositional generalization.
