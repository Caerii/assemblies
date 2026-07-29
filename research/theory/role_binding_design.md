# Role binding: what the paper does, what the demo does, and why one retrieves

**STATUS: substantially RETRACTED and rewritten.** The first version of this
document concluded that the demo parser's role binding has no addressable key,
that the difference from the synthetic task was **arity**, and that no amount of
parameter tuning could help. That conclusion was drawn from a measurement that
was comparing two different coordinate systems. With the metric fixed, the same
code in the same arms retrieves at **1.000**. Section 6 records exactly what
fell and what survived.

## 1. What the paper actually specifies

`neural_assemblies/language/parser.py` (`ParserBrain`) is a faithful
implementation of the TACL fiber-gated parser, and it is worth reading before
touching the emergent one. Its mechanism:

| element | how it works |
|---|---|
| `fiber_states[a][b]` | a set; **empty means disinhibited**. Non-empty = fiber closed. |
| `getProjectMap()` | builds the projection map from fiber and area states — projections happen **only along open fibers** |
| `applyFiberRule` | `INHIBIT` / `DISINHIBIT` per lexeme, so grammar decides which fibers open |
| LEX assemblies | **fixed, disjoint, index-addressed**: `range(word_index*k, (word_index+1)*k)` |
| `getWord(area, cue_area=...)` | reads the winners against those fixed blocks, or resolves through a **lexeme cache keyed by cue area** |
| `getActivatedFibers()` | the parse *is* the set of fibers that fired |

The load-bearing property: **a role area never has to store and discriminate
many word assemblies.** It holds one thing at a time. The binding lives in the
*gating pattern*, and the parse is recovered from which fibers were open — not
from assembly similarity inside a shared area.

## 2. The bug that produced four months of "role retrieval is at chance"

`assembly_calculus/parser.py` stores an assembly per word in a shared role area
and recovers it later by similarity. Measuring that requires intersecting the
live activity with each stored assembly — and the two live in **different index
spaces**:

```python
stored = {w: list(lex[w].winners) for w in words}          # NEURON IDs
live   = set(int(x) for x in brain.areas[role_area].winners)  # COMPACT indices
```

The sparse engine never materialises an area's full `n` neurons; a neuron gets a
compact slot only once it has won something. `Assembly.winners` holds stable
neuron IDs, `area.winners` holds compact positions, and `ops._snap` is the
one-way door between them. Intersecting the two compares unrelated coordinate
systems, so the overlap is whatever two arbitrary integer sets happen to share.

That is **exactly chance, invariant to every parameter** — which is precisely
what was observed and precisely what was over-interpreted. The A/B, on the
unchanged toy corpus with `BR_COMPACT_READOUT` as a negative control:

| arm | broken metric | matched index spaces |
|---|---|---|
| shipped (reset + recurrence) | 0.333 / 0.333 / 0.333 | **0.333 / 0.333 / 0.333** |
| no reset, feed-forward | 0.444 / 0.111 / 0.333 | **1.000 / 1.000 / 1.000** |
| no reset, feed-forward, dedup | 0.444 / 0.111 / 0.333 | **1.000 / 1.000 / 1.000** |

(AGENT / ACTION / PATIENT; chance 0.333.) The broken column reproduces the exact
figures previously reported, including the 0.444 that was read as a hint of a
weak effect. This failure mode is on record in this project's own notes as
having "silently voided the merge line" once already.

## 3. What the real defect is, and it is the shipped path

The **shipped** arm still sits at exactly chance with matched index spaces, so
one of the two original diagnoses survives intact:

1. `train_roles` calls `reset_area_connections(role_area)` after **every**
   word, zeroing the role area's connectome so k-WTA falls through to its index
   tie-break and returns the same lowest-index winners for every word. This
   reset is genuinely load-bearing in `train_lexicon`, where each word has its
   own grounding stimulus so the tie-break is never reached, and harmful in
   `train_roles`. **The fix must be per-call-site.**
2. `train_roles` also drives `{lex: [role], role: [role]}` — self-recurrence in
   a shared area, this project's documented collapse channel.

Remove both and retrieval is perfect. Note the shipped arm's signature: accuracy
exactly at chance *with a unit margin*, which is this project's recorded
"dead probe" pattern and should always prompt a look at the metric before the
mechanism.

## 4. Retrieval works, and it works well past the ladder's capacity bound

`bridge_capacity.py` runs the fixed metric on the 40-noun / 20-verb corpus, at
the gain derived from the phase map (`beta = g_c^(1/rounds) - 1 = 0.0718`) with
`kp` held at 10 and each `(word, role)` binding trained exactly once. At
n=1000, k=50 the ladder's capacity bound is `M_max ~ 1.15 n/k ~ 23`:

| area | M | α = M/M_max | diag | offdiag | retrieval | chance |
|---|---|---|---|---|---|---|
| AGENT | 16 | 0.70 | 1.000 | 0.079 | 1.000 | 0.062 |
| ACTION | 20 | 0.87 | 0.998 | 0.099 | 1.000 | 0.050 |
| PATIENT | 40 | **1.74** | 0.948 | 0.238 | **1.000** | 0.025 |

Two things to read off this. First, the **order parameters move exactly as the
map predicts**: as α rises past 1 the diagonal falls and the off-diagonal
climbs threefold — crowding is arriving on schedule. Second, **retrieval
accuracy does not break**, because the argmax still has margin (0.948 against
0.238). So α\* ≈ 1.15 is **not a constant of the substrate**. It was measured on
a depth-5 chain of shared areas, where errors compound across levels; role
binding here is depth 1, and a single association tolerates far more load.

The follow-up sweep varies load **alone** — n=1000, k=50, p=0.2, so k/n and kp
are both fixed and only M moves. This matters because the table above cannot
separate α from density: it varied k at fixed kp, so p = 10/k and k/n ran from
0.01 to 0.40, and at k/n = 0.4 an off-diagonal of 0.977 is forced by pigeonhole
rather than by load. The tell is that α = 1.74 appears twice with margins 0.694
and 0.464.

| M | α | distinct | diag | offdiag | margin | retrieval |
|---|---|---|---|---|---|---|
| 10 | 0.50 | 1.000 | 1.000 | 0.050 | 0.950 | 1.000 |
| 20 | 1.00 | 1.000 | 0.999 | 0.094 | 0.905 | 1.000 |
| 30 | 1.50 | 1.000 | 0.988 | 0.148 | 0.841 | 1.000 |
| 40 | 2.00 | 1.000 | 0.945 | 0.221 | 0.723 | 1.000 |

Retrieval is perfect throughout while the margin decays smoothly, so **α\* at
depth 1 is bounded below by 2.0 and is not yet located**. Extending the curve is
blocked on vocabulary rather than on parameters: past M=40 the sweep starts
adding verbs, which arrive through a *second* source fibre, and **load is
per-fibre, not per-area** — at M=60 the busiest fibre still carries 40 items.
The signature of that confound is a margin that *rises* (0.723 → 0.799 → 0.831),
which is not capacity recovering but the mean off-diagonal being diluted by
cross-fibre pairs whose weights are independent.

## 5. Two proper designs, and they are still not the same project

**(A) Gated — the paper's.** Role assignment is a *fiber* decision, and the
parse is read from the activated-fiber set. Already implemented in
`language/parser.py`. Measured separately as a **perfect positional template** —
1.000 on reversible sentences, 0.000 on irreversible — i.e. flawless structure
with *no lexical sensitivity whatsoever*.

**(B) Lexical — the demo's, and it does work.** A word's LEX assembly is an
addressable key: k of n neurons, near-disjoint across words, so a feed-forward
`LEX -> ROLE` fiber is an ordinary associative memory. This is what now measures
1.000.

These are complementary, which is the substance of task #33 — and the case is
stronger than before, because both routes are now known to work in their own
terms. `corpus.py` supplies the reversible/irreversible split needed to test the
composition directly.

## 6. What is retracted

**Retracted outright:**

- That the demo parser's role binding **has no addressable key**. It has one:
  the LEX assembly. Retrieval is 1.000.
- That the operative difference from the synthetic task is **arity** — two
  parents versus one. A single parent suffices; the second parent in the
  synthetic task was never what made it retrievable.
- That "no amount of β will give it one". The claim was untestable as posed,
  because the metric could not have responded to β no matter what happened in
  the brain. β *invariance* was the diagnostic tell and I read it as evidence
  for a structural claim instead of evidence against the measurement.
- The derivation that **no single β exists** at these operating points. Its
  arithmetic stands on its own terms and its conclusion — train each binding
  once, so `c_max = 1` — is what `bridge_capacity.py` does. But it was offered
  as the explanation of a failure that had a different cause, and the dedup arm
  retrieves at 1.000 *and so does the non-dedup arm*, so the corpus-frequency
  spread was not what was breaking anything here.

**Standing:**

- The `reset_area_connections` diagnosis, per call site: load-bearing for LEX,
  destructive for ROLE. The shipped arm is still at chance because of it.
- Self-recurrence in a shared area as the collapse channel.
- The phase map, now with a **confirmed quantitative signature** in the parser:
  diag and offdiag track α through the crowding onset (§4).
- That the lexicon is immune to gain for a mechanistic reason. Separately
  confirmed at 0.977 PHON-only reproduction, unchanged by role training.

## 7. An engine bug found on the way, and it is not confined to this experiment

The load sweep's first run reported a hard ceiling at exactly 40 distinct
assemblies for both M=50 and M=60, with `retrieval == distinct_frac` to three
decimals. That looked like capacity and was not. Shuffling the training order
left the collapsed set unchanged — always the 20 verbs, never a noun, with the
first collapse at training position 2 — which ruled out both accumulation and
order and pointed at the *source area*.

`LEX_VERB -> ROLE_AGENT` had `weights.shape == (0, 0)`. In the sparse engine an
area→area block is materialised lazily, and a source first used against an
already-grown target is marked for **deferred initialisation** so it can
contribute on the next round. That consumption site sits at the end of
`project_into` — *after* the "zero signal → preserve current assembly" early
return. So when the new source is the **only** source, it contributes nothing,
the total is zero, the function returns early, and the initialisation never
runs. Every round after is identical. Forever.

The symptom is not a zero assembly but a **stale** one, because that branch
preserves the target's current winners. Every verb "stored" the 40th noun's
assembly. Swapping the training order swapped which category died — 40 nouns
onto the last verb — so whichever source happens to project first is the only
one that ever works, and nothing warns.

Fixed by running the deferred init on both exits. After the fix, 60 of 60
assemblies are distinct in either training order and both fibres materialise.

**Scope.** The per-area sweep in §4 is unaffected and its numbers are unchanged
to four decimals, because each role area there receives from exactly one LEX
area (`bindings` sends nouns to AGENT/PATIENT and verbs to ACTION). The bug
needs *two* source areas into one target, introduced at different times — which
is the normal shape of helper-area architectures, so it is worth an audit rather
than a footnote.

## 8. Method notes

The 0.020 precondition reading that started this was *also* the
same index-space bug, in the gate I had written to catch exactly this class of
error. The gate fired correctly and my explanation of it was wrong twice over
before `lex_reproducibility.py` measured the three candidate causes separately
instead of arguing about them. Checking a precondition is not the same as
checking the precondition check.
