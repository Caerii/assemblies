# Which paper-specified mechanisms are symbolic here, and why

Written after four separate findings turned out to be the same missing
primitive. Everything marked MEASURED was checked directly; everything marked
PROPOSED is a design argument and has not been implemented.

## The audit

| paper mechanism | this repo | status |
|---|---|---|
| role areas in mutual inhibition | registered, **never fires** | MEASURED: 1361 training + 12 parse `project()` calls reach `_apply_mutual_inhibition`, **0** co-target a group |
| role exclusivity | Python `inhibited` set in `_assign_roles_neural` | MEASURED: symbolic |
| word order / constituent order | `self.word_order_type`, a stored Python string | MEASURED (lesion study): corrupting the attribute takes reversible accuracy to 0.000 |
| lexical role readout | `role_lexicons` dict of stored assembly snapshots | MEASURED: clearing the dict forces accuracy to exactly 0.000 with zero variance; zeroing every synapse only costs ~0.46 |
| which role area a word targets | `structural_role_area(category, verb_seen)`, a lookup | MEASURED: routes a VERB to VP, which is what confounded the P600 contrast |

Two of these were found by accident while chasing unrelated bugs. That is the
reason for this document: they are not independent.

## The missing primitive

The reference `Brain` (`.reference/dmitropolsky-assemblies/brain.py`) has **no
inhibition of any kind** -- `grep -c inhibit` returns 0. Inhibition is not a
neural operation in NEMO. It lives entirely in the parser as two state maps:

    fiber_states[from_area][to_area]   # set of inhibition indices, per DIRECTED PAIR
    area_states[area]                  # set of inhibition indices, per AREA

open  <=>  the set is empty
INHIBIT    adds an index
DISINHIBIT discards an index

and the projection targets are then **derived** rather than named
(`parser.py::getProjectMap`):

    project a1 -> a2  iff  area_states[a1] empty
                      and  area_states[a2] empty
                      and  fiber_states[a1][a2] empty
                      and  a1 has winners

So parsing in NEMO is: apply the word's rules to the state maps, derive the
project map, project. **The parser never names a target.** Which area a word
binds to is a consequence of which fibers are open.

This repo has neither half:

* **No fiber-level inhibition.** There is no way to express the reference's
  central move, `"dogs": open fibers LEX<->SUBJ and LEX<->OBJ but only SUBJ
  disinhibited`. Targets can only be chosen in Python.
* **No index/refcount semantics.** `inhibit_areas()` *clears winners* -- it is
  erasure, not a reversible gate, and it does not compose. The reference's set
  of indices lets independent rules inhibit the same fiber and only reopen it
  when all of them release. A boolean cannot do that, which is exactly why the
  bookkeeping ends up in Python.
* **No project-map derivation.** Callers pass explicit target dicts, so the
  control flow lives in `if`/`else` rather than in the model state.

And one primitive this repo **added** that the reference does not have:
`add_mutual_inhibition`, winner-take-all across areas, applied *after*
projection -- every target fires, then losers get `winners=[]` and `w=0`.

## Why that substitution fails

Gating and post-hoc suppression are different computations.

* Disinhibition needs **no cross-area comparison**: exclusivity holds because
  only one fiber was open.
* WTA-after-the-fact needs a **reliable comparison between areas**, and
  MEASURED, that comparison is unreliable here: `total_activation` sums only
  the top-k winners, so an untrained area whose entire materialized population
  IS k keeps 100% of its small drive while a trained area's ~960 neurons dilute
  what the top-k captures. A 7x pre-k-WTA separation (7.60 / 6.61 vs ~1.05)
  collapses to a ~7% margin (1.128 / 1.069 vs ~1.047) that noise flips.

So the surrogate primitive both requires something the substrate cannot supply
reliably, and is never actually invoked. The Python `inhibited` set is doing the
work in its place.

## The same cause, four times

* **Role exclusivity is symbolic** because there is no fiber gate to make it
  neural.
* **Mutual inhibition is dormant** because with targets chosen directly, the
  parser never needs to co-target -- the primitive was added to substitute for
  fiber control and then bypassed.
* **The P600 contrast was area-confounded** because `structural_role_area` is a
  categorical stand-in for "which fiber is open", and it routes the intruding
  word by its OWN category, so a verb goes to VP and the violation is never
  measured where the syntax was binding.
* **There is no ELAN** because expected-vs-actual is not represented anywhere:
  with fiber state, "expected" is simply whichever role fiber the syntax has
  disinhibited.

## PROPOSED primitive set

Add to `Brain`, mirroring the reference:

    inhibit_fiber(src, dst, index=0)      disinhibit_fiber(src, dst, index=0)
    inhibit_area(area, index=0)           disinhibit_area(area, index=0)
    fiber_open(src, dst) -> bool          area_open(area) -> bool
    project_map() -> Dict[str, List[str]] # derived from state + winners

Keep `fix_assembly`/`unfix_assembly` as they are. Keep `inhibit_areas()` under
its current name but stop describing it as inhibition -- it is `clear_winners`.

`add_mutual_inhibition` should be either retired or explicitly documented as a
non-NEMO extension, because the reference achieves exclusivity by gating and
this repo's WTA is a different model with a measured reliability problem.

## PROPOSED composition

    for word in sentence:
        for rule in lexeme[word].pre_rules:   apply_rule(rule)   # mutate state
        brain.project({}, brain.project_map())                   # derived
        for rule in lexeme[word].post_rules:  apply_rule(rule)

Role exclusivity then emerges: only one role fiber is open, so only one role
area receives the word. Nothing compares areas, nothing kills losers, and no
Python set is required.

## What to check before building this

1. Whether the emergent parser's *acquisition* story survives. NEMO's rules are
   given per lexeme; this repo's selling point is that categories are LEARNED
   from grounding. Fiber rules would have to be learned too, or the model trades
   one hand-specification for another. **This is the real risk and it should be
   settled first.**
2. Whether `project_map()` derivation is affordable at 48 areas -- the reference
   loops all pairs each step.
3. Whether existing results survive. Several depend on the current path; the
   0.930 role accuracy is achieved *with* the symbolic set, so a neural
   replacement must be re-measured, not assumed equivalent.
