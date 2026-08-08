# Passives buy 0.55 on held-out reversible sentences — and the assemblies are not what decides

## The measurement

Two corpora identical except for voice alternation, same brain seeds, scored on
40 **held-out reversible** sentences: both participants animate, so either could
be the agent and no lexical or plausibility shortcut can answer; the specific
verb+participant triple never occurred in training. Chance is **0.5** — two
nouns, two roles, two assignments.

| voice | `alternating` | `active_only` | paired delta |
|---|---|---|---|
| active | 0.8500 | 0.8500 | 0.0000 |
| passive | **0.6000** | **0.0500** | **+0.5500** (clears 0) |

Read the `active_only` passive score carefully: **0.05 is far BELOW chance.**
That is not a parser guessing. It is a parser that is *systematically inverted*
— which is exactly what a position counter does to a passive, and it is the
sharpest confirmation available that the all-active corpus was teaching position
rather than role. The premise of the whole arc was right.

And the active voice does not regress: 0.85 in both arms. The failure mode I
was watching for — a passive-bearing corpus teaching the determiner to reverse
everything — did not happen.

## The finding that matters more

Every seed returned a **bit-identical** value. Zero-width CIs in all four cells.

That has two opposite explanations, and they must be distinguished before the
number means anything: either the seeds never reached the substrate (in which
case the ensemble is one seed repeated five times and the interval is fiction),
or the substrate genuinely does not make this decision.

Measured: **10 distinct substrates** across the run — different `dog`
assemblies, `NOUN_CORE.w` ranging 2588–2676 — while the metric does not move
at all.

**So the role decision is not being made by the assemblies.** It is made by the
symbolic route: the constituent-order prior plus the learned gating rule. The
substrate is carrying the lexicon, and the structure is being decided above it.

This is the honest frame for everything in the last two commits: the *pipeline*
now handles voice alternation correctly, and that is a real and large gain. The
*assembly calculus* has not been shown to. For a project whose goal is to
compete with deep learning on compositional generalisation, that distinction is
the whole game, and this is direct evidence for #33 (whether the symbolic role
route can be retired for the neural one) — currently it cannot, because the
neural one is not what is answering.

## Two dead readouts caught before they became results

Both would have produced a confident wrong answer, and both were caught by
guards rather than by inspection.

**1. The off switch did not switch off.** I disabled passives by setting
`PASSIVE_EVERY` to an enormous number. But `n_eligible % N == 0` is TRUE at
`n_eligible == 0`, so the first eligible clause still emitted a passive and the
control arm contained one. The experiment's own corpus check ("alternating has
9 passives, active_only has 1") aborted before scoring. `PASSIVE_EVERY = 0` is
now an explicit off switch in the generator.

**2. The readout was a different parser.** I scored through `probe_parse`,
which runs the RULE-PROGRAM route (`NemoParser`). It skips words it has no
program for — "is", "by", the participle — and returns `roles: None` even for
"the dog chases the cat", which the neural route reads correctly. **Both arms,
both voices, scored 0.0000**: a clean null that was entirely an artifact of
asking the wrong parser. `same-name-two-meanings` again — two things called
`roles`, two things called "parse".

The tell was that ACTIVE scored 0.0000 when I had watched active parsing work
an hour earlier. A null that contradicts something already measured is a broken
instrument, not a result.

## Limits

* 40 probes, 5 seeds, one vocabulary preset, one corpus generator. The seed
  dimension turns out to buy nothing here (see above), so the effective n is
  the 40 probes.
* 0.60 on passives means **40% of held-out passives are still wrong**, and 0.85
  on actives means 15% of actives are. Neither is close to solved.
* Reversibility is enforced by `features.animate` on both participants. That
  makes the pair semantically symmetric in the lexicon's terms, not in the
  world's — "the mouse enters the child" is reversible by this test and odd by
  any other.
* The `alternating` arm was trained on a corpus containing the MARKER; the
  probes are held-out sentences using the same marker. This measures
  generalisation to new participants and verbs, NOT to a new construction.
