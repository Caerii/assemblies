# Re-binding fixes ~21% of failures and breaks more than it fixes

## The result

3 seeds, `deepcopy`-paired arms off one parser, `tail_rounds=1` — the same
protocol `train_roles` uses, so this measures REPETITION and not a different
operation.

| area | arm | overall | re-bound | untouched | spread | margin |
|---|---|---|---|---|---|---|
| `ROLE_PATIENT` | baseline | **0.389** | — | — | 0.165 | 1.19 |
| | guided | 0.315 | 0.212 | 0.483 | 0.178 | 1.14 |
| | random | 0.343 | 0.409 | 0.244 | 0.180 | 1.15 |
| `ROLE_AGENT` | baseline | **0.355** | — | — | 0.140 | 1.42 |
| | guided | 0.355 | 0.180 | 0.681 | 0.224 | 1.34 |
| | random | 0.362 | 0.381 | 0.327 | 0.207 | 1.37 |

**Neither arm beats baseline. Guided does not beat random.** The error signal
buys nothing.

## What the numbers say happened

Re-binding a failed word **does** fix it about 21% of the time — the guided
arm's re-bound subset goes from 0% correct by construction to 0.212. That part
works.

It just costs more than it gains. Taking `ROLE_PATIENT` at ~36 words:

```
failures re-bound       ~22 words,  21% now correct    ~ +4.6
previously correct      ~14 words,  now 0.483 correct  ~ -7.2
                                                  overall falls
```

**Fixing one failed binding breaks roughly 1.5 previously-correct ones.** That
is the capacity ceiling, measured directly rather than inferred.

And it is not free in the other currency either: **spread rises in every arm** —
`ROLE_AGENT` from 0.140 to 0.207–0.224, about +50%. Extra binding buys no
accuracy and costs distinctness, which is the reinforcement tradeoff this repo
already documented, now located on the role areas specifically.

## One comparison in that table is not valid, and it is mine

The `re-bound` and `untouched` columns **cannot be compared across arms.** The
guided arm selects failures, so its untouched set is by construction the words
that were already correct — which is why guided-untouched reads 0.483/0.681
against random's 0.244/0.327. That is selection, not an effect.

Only **overall** is comparable between arms, and only guided-vs-random is
attributable to the error signal, since both arms carry the same re-snapshot
recency bias. Both readings say the same thing, so the conclusion survives — but
the untouched column looks like a striking guided win and is not one.

## The quantitative point worth keeping

The ceiling shows up at **α = Mk/n = 36 × 30 / 3000 = 0.36**, against a critical
load of **α\* ≈ 1.15** measured elsewhere in this repo.

So binding-into-a-shared-target saturates at roughly **a third** of the load the
storage regime tolerates. Those are different protocols and the comparison is
indicative rather than exact — α\* was measured for assemblies stored in an
area, not for many sources bound to distinct targets inside one — but a 3×
shortfall is large enough to be worth a direct measurement.

**That is the composition line's question, not the ERP line's.** If a role area
saturates at α ≈ 0.36, the constraint on how much grammar this substrate can
hold is a capacity law nobody has written down, and it binds long before
vocabulary does (the lexicon has no known ceiling).

## What this closes

**Do not wire the error signal into training.** It selects correctly — failures
really are fixable at ~21% — but the area cannot absorb the fix. A mechanism
whose selectivity is real and whose net effect is negative is worse than none,
because it looks principled.

The honest next question is not "how do we re-bind better" but **"what is the
capacity of a shared role target, and does the parser sit past it?"** The
evidence here says yes, at α ≈ 0.36.

## Limits

`tail_rounds=1` only — one extra binding per selected word. More rounds might
find a different operating point, though spread is already climbing, so the
likely direction is more merge rather than more accuracy. 3 seeds. And the
retrieval probe drives `core → role` directly rather than through the
incremental circuit, so every number here is an upper bound on what a live parse
sees.
