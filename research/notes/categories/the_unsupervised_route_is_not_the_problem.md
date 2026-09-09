# The curriculum's route is unsupervised — and it is not worse

## Step 1: which route runs, counted rather than inferred

Every writer of `role_lexicons` wrapped, counting entries written per depth:

| depth | route | entries written |
|---|---|---|
| `SENTENCES` | `train_unsupervised` (**unsupervised**) | 130 |
| `FULL_TRAIN` | `train_roles` (**supervised**) | 25 |

`train_roles` never fires on the curriculum path. Confirmed by counting, after
being wrong twice this session about what a code path did.

**So every number in this arc** — retrieval 0.33–0.42, re-binding redistributing,
the α ≈ 0.36 ceiling — **describes unsupervised role induction.** The careful
`bind()` protocol, the one three drifted copies were unified into, does not run
where the parser is actually trained.

## The correction that matters: `roles=[None]` is not a bug

I filed #116 partly on the claim that `roles=[None] * len(sent)` disables three
mechanisms from one line. That over-stated it.

`_generate_sentences` returns **token lists** — the CDS corpus is raw text with
no role annotations. There is nothing to put in that field. The unsupervised
route is not a fallback that got wired by accident; it is the only thing
available, and it is the intended design for un-annotated text. Learning roles
from raw text is the research goal, not an accident.

What survives from #116 is narrower and still real: `_ROLE_MAP` covers 3 of 7
roles, so the blocks-world curriculum's six `goal` annotations are silently
dropped by `continue`. That is a genuine dead path. It is not "three mechanisms
inert from one line."

## Step 2: the raw comparison does not work, exactly as pre-registered

| depth | area | M | α | retrieval | chance | margin |
|---|---|---|---|---|---|---|
| `SENTENCES` | `ROLE_PATIENT` | 36 | 0.360 | 0.375 | 0.028 | 1.21 |
| `SENTENCES` | `ROLE_AGENT` | 46 | 0.460 | 0.359 | 0.022 | 1.28 |
| `FULL_TRAIN` | `ROLE_PATIENT` | 6 | 0.060 | 0.800 | 0.200 | 2.19 |
| `FULL_TRAIN` | `ROLE_AGENT` | 11 | 0.110 | 0.667 | 0.167 | 3.89 |

FULL_TRAIN looks twice as good and **is not comparable**: α differs 6×, chance
differs 7×, and margin inflates when there are fewer competitors to be second.
Reading 0.800 against 0.375 as "supervised is better" would be a route claim
resting entirely on load.

## Step 3: matched readout difficulty

Candidate set subsampled to a fixed size, 40 random subsets, so both depths are
judged at the same task:

| depth | area | M stored | ret@6 (chance 0.167) | ret@11 (chance 0.091) |
|---|---|---|---|---|
| `FULL_TRAIN` | `ROLE_AGENT` | 11 | **0.667** | — |
| `SENTENCES` | `ROLE_PATIENT` | 36 | **0.729** | 0.639 |
| `SENTENCES` | `ROLE_AGENT` | 46 | **0.721** | 0.591 |

**The unsupervised route matches or beats the supervised one at equal readout
difficulty — while carrying 3–4× the writing load.**

So the "degraded route" hypothesis is refuted. The curriculum is not running a
broken path; it is running a working path past its capacity.

## The capacity curve, obtained for free

The same subsampling gives a discriminability curve on the real parser:

| candidates | retrieval | chance |
|---|---|---|
| 6 | 0.729 | 0.167 |
| 11 | 0.639 | 0.091 |
| 36 | 0.375 | 0.028 |

**The bindings are good; the area cannot keep 36 of them separable.** At 6
competitors three quarters are recoverable. This is #122's question answered on
the production substrate rather than a toy one — though it varies READOUT load
with WRITING load held fixed, which is the complement of the sweep #122 asks
for, not a substitute for it.

## Limits

`FULL_TRAIN ROLE_PATIENT` has fewer than 6 words present in both the core and
role lexicons, so `ret@6` is undefined there and the supervised side of the
matched comparison rests on **one** cell (`ROLE_AGENT`, 0.667). That is thin.
The direction is consistent across both SENTENCES areas and the gap is small, so
the defensible claim is "not worse", not "better".

2 seeds. And the retrieval probe drives `core → role` directly rather than
through the incremental circuit, so all of these are upper bounds on what a live
parse sees.
