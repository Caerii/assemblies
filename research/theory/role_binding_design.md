# Role binding: what the paper does, what the demo does, and why one retrieves

Written after the bridge experiment measured role retrieval at chance in six
different training configurations, and no amount of gain correction moved it.
The conclusion is that this was never a tuning problem. The demo parser's role
binding is **not the paper's mechanism**, and the mechanism it uses has no
addressable key, so there is nothing for a readout to cue on.

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

## 2. What the demo parser does instead

`assembly_calculus/parser.py` — whose own docstring calls it "a composition
demo" — does the thing the paper's design avoids:

```python
for word, role_area in zip(sentence, role_sequence):
    project(brain, stim_map[word], lex_area, rounds)
    brain.areas[lex_area].fix_assembly()
    for _ in range(rounds):
        brain.project({}, {lex_area: [role_area], role_area: [role_area]})
    role_lexicons[role_area][word] = _snap(brain, role_area)
    brain._engine.reset_area_connections(role_area)
```

Every word is projected into a **shared** role area and an assembly per word is
stored there, to be recovered later by similarity. Three separate problems
follow, and only the first two are the ones I had been chasing:

1. **The reset zeroes the connectome**, so k-WTA falls to its index tie-break
   and every word yields the same winners. Fixed by suppressing the reset for
   `ROLE_*` only — it is genuinely load-bearing for LEX, where each word has its
   own grounding stimulus so the tie-break is never reached.
2. **Self-recurrence in a shared area** is this project's documented collapse
   channel. Removing it, plus the gain derived from the phase map, takes
   distinctness to 1.000 and overlap to an order of magnitude *below* the chance
   floor.
3. **There is no cue.** And this one is not a bug to fix — it is the design.

## 3. Why retrieval reads at chance: no addressable key

Readout drives `LEX -> ROLE` and asks which stored role assembly the result
matches. But the *only* input is the word's LEX assembly, and the LEX→ROLE
fiber plus the shared recurrent fiber are trained by **every** word. So the
question "retrieve dog-as-agent" and "retrieve cat-as-agent" are posed with
nothing that distinguishes them except a lexical pattern the shared fibers have
already smeared together.

Contrast the synthetic sweeps, where retrieval reached **1.000** at depth 5.
There, each item's level-`L` assembly was formed by

    merge(C[L-1], P[L] -> C[L],  stim_b = p{L}_{item})

i.e. **two parents**: the content (the chain so far) and a per-item partner
stimulus. That partner is a *key*. Retrieval works because the read presents the
same key.

So the difference is not gain, depth, load or density — all of which I mapped.
It is **arity**:

> The synthetic task binds with **two** parents, one of which acts as an
> addressable key. The parser's role binding projects from **one** parent and
> therefore stores content with no key to retrieve it by.

That is why sweeping β from 0.1 to 0.004 changed nothing about retrieval while
changing distinctness by two orders of magnitude. Distinctness is a property of
storage; retrievability requires a key.

## 4. Two proper designs, and they are not the same project

**(A) Gated — the paper's.** Role assignment is a *fiber* decision. Grammar
disinhibits `LEX -> ROLE_AGENT` for a subject, the role area receives only that
projection, and the parse is read from the activated-fiber set. Already
implemented in `language/parser.py`. Measured separately as a **perfect
positional template** — 1.000 on reversible sentences, 0.000 on irreversible —
i.e. flawless structure with *no lexical sensitivity whatsoever*.

**(B) Keyed merge — what the synthetic results support.** Make role binding a
genuine `merge` of two parents: the word's LEX assembly and a **role cue**
(a per-role stimulus, or the gating state reified as a drive). Then a role area
holds `bind(word, role)`, retrieval presents the role cue, and the phase map
applies directly with everything it says about capacity, gain and depth.

These are complementary rather than competing, which is the substance of the
open question in task #33: gating supplies the structure the lexical route
lacks, and the lexical route supplies the content sensitivity gating lacks.

## 5. Concrete recommendation

Do **not** keep tuning the demo's role path; it has no key and no amount of β
will give it one. Instead:

1. **Change the mechanism, not the parameter.** Re-implement role binding as
   `merge(LEX, ROLE_CUE -> ROLE_AREA)` with a per-role cue stimulus present at
   *both* training and readout. This is a small change and it makes the
   parser's role areas the same object the phase map was built on.
2. **Then the map applies quantitatively.** At n=1000, k=50 the map gives
   `M_max ~ 1.15 n/k ~ 23` items per role area, with per-merge gain set from the
   corpus by `g = g_c^(1/(rounds * c_max))`. For a 3-word corpus that is
   comfortably inside capacity — consistent with distinctness already being fine
   once the reset is out of the way.
3. **Keep the reset fix regardless**, per call site: suppress for `ROLE_*`,
   retain for `LEX`.
4. **Judge with the paper's readout where the paper's mechanism is used.**
   `getWord(area, cue_area=...)` and `getActivatedFibers()` are the designed
   readouts for the gated route; assembly-similarity readout is only meaningful
   for the keyed-merge route.

## 6. What this retracts and what it leaves standing

**Retracted:** my framing that the parser's role failure was an instance of the
crowding phenomenon the phase map describes. It is not. Crowding was present and
is fixed, and retrieval was independently broken for an architectural reason.

**Standing:** the phase map itself, and the bridge's first two findings — that
the lexicon is immune for a mechanistic reason (feed-forward selection cannot be
reordered by uniform potentiation), and that the derived gain repairs
representational health in the role areas. What does not follow, and what I
briefly implied, is that repairing representational health repairs the task.
