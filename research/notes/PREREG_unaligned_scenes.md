# PREREG: cross-situational learning -- reference and structure solved together

Registered before building. The next rung named in `core/scene.py` and
`grounded-role-learning`: today each word's grounding lists its OWN features,
so word -> referent alignment is given and only the role mapping is learned.
A child gets a scene and a sentence, and has to work out which word names
which participant BEFORE any role can be read off the causal order.

## What changes

    today      sentence: words, each with its feature bundle attached
               scene:    participants (feature bundles) in causal order
               learned:  causal order -> linguistic order (word order)

    here       sentence: words ONLY
               scene:    participants (feature bundles) in causal order,
                         plus the action's features, UNLABELLED
               learned:  (1) which word denotes which participant
                         (2) causal order -> linguistic order, using (1)

Nothing in the scene names a token. Alignment must come from
CROSS-SITUATIONAL statistics: across scenes, "dog" co-occurs with [DOG, ANIMAL]
every time and with [BALL, OBJECT] only sometimes.

## Why the substrate should do this without new machinery

Alignment is a conjunction learned by frequency -- [[hebbian-mass-follows-
frequency]] -- and the anchor result says grounding IS the stimulus anchor
([[capacity-is-an-anchor-ratio]]). Fire the word's phonological stimulus and
ALL of the scene's feature stimuli into the lexical area together; the
consistent conjunction accrues mass, the incidental ones do not. This is the
same mechanism the lexicon already uses, minus the per-word feature list.
Distractor participants are the load: a scene with P participants fires P
feature bundles per word, so the word's own referent is 1/P of the grounding
drive on any one exposure and 1/1 across exposures.

## Design

Corpus: `grounded_corpus.build()` with `SceneEvent`s, then a NEW strip that
deletes each word's `grounding` (not only `roles`) and leaves the scene.
Words keep phonology only. The scene supplies `participants` (bundles) and
`action` features; the mapping word -> bundle is the target and is NEVER read
by the learner -- it is read by the SCORER, from the deleted groundings.

Learner: for each sentence, for each word, one `project` of
{phon(word), feat(f) for every f in the scene} into LEX with recurrence, as
the lexicon does today; no role information in this phase. Alignment
readout, probe-isolated ([[probe-isolation-required]]): cue phon(word) alone,
read LEX, and score each participant bundle by the overlap of the recalled
assembly with the bundle's own feature-driven assembly. Argmax = the aligned
referent.

Then roles: `roles_from_scene` currently matches a word's OWN features to the
participants; here it is fed the ALIGNED bundle instead. Then
`word_order_induction` unchanged.

## Bars, stated now

Seeds (42, 1, 2, 3, 4); 198-sentence corpus; content words with >= 3
exposures.

    U1  ALIGNMENT   fraction of content words whose argmax referent equals the
                    deleted grounding.
                    PASS  >= 0.85 mean, and the confidence bound beats 2x
                          chance (chance = 1 / mean participants per scene,
                          ~0.4 on this corpus)
                    FAIL  <= 0.60
    U2  ORDER       word order induced from ALIGNED roles.
                    PASS  >= 5/6 orders on >= 4/5 seeds (the aligned run
                          recovered 6/6)
    U3  ABLATION    same learner with scene features SHUFFLED across
                    sentences (word and scene decorrelated).
                    Alignment must fall to chance (<= 1.5x); if it does not,
                    the scorer is leaking the answer and U1 is void.
    U4  COMPLEMENT  the superordinate-collision bug's successor: two-person
                    events ([BOY, PERSON] vs [GIRL, PERSON]). Alignment
                    accuracy on those words reported separately; it is
                    expected to be the weakest cell, and the distinctive
                    feature should carry it (overlap SIZE, not first match).

## What is NOT claimed

* Nothing about syntax beyond word order; nothing about function words.
* The scene still gives causal order. Learning the causal structure itself
  (who is agent) from perception is a separate acquisition problem, deferred
  as `CAUSAL_ROLE_ORDER` says.
* No claim about real corpora; the grounding vocabulary is the lesion
  corpus's.

## Interpretation, stated now

* U1, U2, U3 pass -> reference and structure ARE solvable together on this
  substrate by the mechanism the lexicon already uses; the grounded pipeline
  is annotation-free end to end. Adopt; register the next rung (causal order
  from perception, or unaligned + multi-event scenes).
* U1 fails, U3 passes -> distractor load defeats frequency at this corpus
  size; measure alignment against exposures per word before adding anything
  (the anchor law predicts accuracy rises with the referent's share of
  grounding drive, i.e. with fewer participants per scene).
* U3 fails -> the scorer leaks; nothing else is read.
