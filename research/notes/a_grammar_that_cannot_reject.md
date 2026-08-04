# The parser's rules were total, so no violation could ever surface

#24 / #104 / #107. The paper's empty-project detector now fires on the emergent
parser. Getting there turned up the reason it never could, and the reason is
the substrate's dominant theme reaching the syntax layer.

## What the paper does, and what we did instead

Mitropolsky, Collins & Papadimitriou (TACL 2021) detect syntactic violations
STRUCTURALLY, not with a graded score. Their example: after the intransitive
verb in "the dogs lived", area OBJ was never disinhibited, so the following
"cats" causes `project*` to fire nothing. An **empty-project error**. It is an
event, and it requires AREA inhibition -- an area that is simply not available.

Our emergent parser had none of that available to it:

    _get_syntactic_target   clamps with min(noun_count, len(seq) - 1) and
                            ALWAYS returns SUBJ or OBJ
    _apply_pre_rules        for a NOUN, unconditionally disinhibits core->target
    transitivity            not consulted anywhere on this path
    FiberCircuit            gates FIBERS only; there is no way to say
                            "this AREA does not exist right now"

So every word always had a route. **A grammar that cannot reject is not a
grammar**, and this is [[same-name-two-meanings]]'s sibling: the no-bottom
substrate, where k-WTA always returns k winners and every operation succeeds,
propagated upward into rules that always succeed too.

Worth stating plainly because I got it wrong out loud first: the emergent
parser DOES gate. It has `FiberCircuit` and a full pre/post rule program. What
it lacked was area inhibition and any notion of transitivity.

## What landed

    Brain.inhibit_area / inhibit_fiber      the calculus's two control
                                            primitives, honoured in
                                            _project_impl, lazily allocated
    _apply_post_rules(VERB)                 inhibits area OBJ when the verb is
                                            intransitive
    infer_transitive_verbs                  distributional, from role
                                            annotation train_roles already uses
    parse_errors.py                         empty-project, nonsense-assembly,
                                            try-project; thresholds DERIVED
                                            from (n, k), never tuned

Measured on a normally-trained parser (n=1000, k=30, the standard corpus):

    transitivity UNKNOWN   "the dog sleeps the cat"   0 errors    (default, unchanged)
    transitivity LEARNED   "the dog sleeps the cat"   2 errors    VIOLATION
    transitivity LEARNED   "the dog chases the cat"   0 errors    control clean

Unknown verbs keep the object slot OPEN. The other default would turn a missing
lexicon into a stream of confident violations, which is an apparatus defect
reading as a linguistic result -- the failure this whole line of work exists to
avoid.

## Two things the measurements corrected mid-build

**The first test sentence was nonsense to the parser.** "the dogs lived cats"
classified as DET/UNKNOWN/UNKNOWN/UNKNOWN -- none of those words are in the
corpus. The detector read 0 errors and I nearly recorded that as "it does not
fire". The vocabulary is `dog`/`sleeps`/`chases`/`cat`. **A null result on
out-of-vocabulary input is not a null result about the mechanism.**

**Prepositional phrases produced a FALSE POSITIVE at exactly the same
magnitude.** With OBJ inhibited after an intransitive verb, "the dog sleeps on
the table" reported TWO errors -- the same count as the real violation. A
detector that cannot separate a grammatical PP from an ungrammatical object is
worse than none. Guarded by skipping the check while a PP is open.

That guard SUPPRESSES, it does not ROUTE: the circuit declares no `core -> PP`
fiber, so the noun still binds nowhere. Recorded as #107 and pinned as
xfail(strict), because a mitigation that looks like a fix is how a gap becomes
permanent.

## And a lesson about the pin itself

The xfail first asserted `len(brain.areas[PP].winners) > 0` -- and XPASSED. PP
does have winners; they are the PREPOSITION's, put there by `PREP_CORE -> PP`.
The assertion was true and said nothing about the noun. Rewritten to assert the
STRUCTURE (`circuit.is_active(NOUN_CORE, PP)` raises while the fiber is
undeclared), which cannot pass by accident.

Same shape as [[fake-perfect-probe-signatures]]: a number that a DIFFERENT
mechanism already satisfies is not evidence for the one you are testing.

## Why this matters for #104

The graded P600 detector is dead -- margin 11.9x above anything observable --
and [[erp_scale_is_an_implementation_detail]] shows its scale tracks `area.w`,
a lazy-materialisation artifact. The signals here cannot fail that way: two are
boolean and the third is a set comparison, and every threshold is derived from
(n, k) rather than fitted to a brain. They are the honest replacement, and
empty-project is now the first one that demonstrably fires end to end.
