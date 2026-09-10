# PREREG: the transducer as a temporal memory

Registered 2026-09-09, before implementing or running.

## Why this design

Every local-rule model that predicts long, high-order sequences uses the
same architecture: the state is the current element's population with a
context-specific subset chosen because the previous state predicted it,
and the transition from previous context cells to those cells is learned
one-shot by a Hebbian rule (HTM; the spiking model of Bouhadjar et al.
2022; Predictive Attractor Models, NeurIPS 2024; Distributed Hebbian
Temporal Memory). None merges contexts by their future with a local rule;
that has been done only with EM or gradients. Our arc is already the
cell of these models, a conjunction of the word's population with a
context. What differs is the context and the selection: our context is
an induced state area that rehashes the prefix every step, and our
selection is a plain k-WTA over the sum of two drives with no prediction
step. The successor-state construction (PREREG_successor_state.md) tried
to merge by future and did nothing; this registration takes the design
the literature converged on.

## The construction

Two changes to `HashedTransducer`, both switchable:

1. **State = previous arc.** The STATE area's winners are set to the
   arc's winners after each tick (no arc -> state projection); the
   state -> arc fiber becomes a lateral arc(t-1) -> arc(t) fiber, learned
   Hebbian prev x new as now. n_state = n_arc.
2. **Predicted neurons win.** Before the arc's k-WTA, the lateral drive
   alone names a predicted set P (the top k of the state -> arc drive,
   thresholded at a fraction of its maximum); the arc drive of neurons in
   P is multiplied by (1 + g). With g = 0 this is the plain conjunction;
   with g large only predicted neurons among the word's population can
   win, the HTM rule. g is swept over {0, 1, 4}.

OUT is read from the arc and teacher-forced as before. Nothing else
changes; at g = 0 with the state projected rather than copied, the
transducer is bit-identical to the registered one (gate).

## Cells and bars

**A. The chain corpus, gap 2** (PREREG_agreement_corpus.md, Amendment 1;
oracle - bigram 0.214). 20 seeds, n = n_arc = 10,000, k = 200.

    TM-1  copy-state alone (g = 0): MRR - bigram (paired) and the
          state-blind delta, REPORTED.
    TM-2  predicted-win at some g in {1, 4}: MRR - bigram lower bound
          >= 0.085 (40% of the gap).  PREDICTION: uncertain, leaning
          FAIL. The literature's models are episodic: they encode each
          sequence uniquely and do not generalize to unseen distractor
          combinations, and the test sentences contain distractor pairs
          the training set does not. A pass would mean the arcs for a
          given number overlap enough across contexts for OUT to read it.
    TM-3  the state is informative wherever TM-2 is measured: full minus
          blind lower bound > 0.

**B. High-order sequences**, the literature's own benchmark, on the
hashed organ with 20 brains, words as symbols, no OUT grounding beyond
the target word:

    set I    two sequences sharing a middle: A D B E and F D B C
    set II   six five-element sequences with shared elements
             (Bouhadjar et al. 2022, sequence set II)
    set III  two twelve-element sequences identical in the middle ten
             (order 10)

Each set is presented 40 times; then each sequence is replayed frozen
and the next word predicted at every position.

    TM-4  set I: after the shared prefix D B, the correct continuation
          (E after A D B, C after F D B) ranks first on >= 18 of 20
          brains at g = 4; at g = 0 with the induced state, reported.
          PREDICTION: PASSES at g = 4.
    TM-5  set III: the order-10 disambiguation holds on >= 15 of 20
          brains at g = 4.  PREDICTION: PASSES; this is what the spiking
          model does with 30 presentations.
    TM-6  learning speed: the presentation at which set I is first
          predicted perfectly, reported (the spiking model: ~30).

Adoption if TM-4 and TM-5 pass: a register entry that the assembly
substrate implements a temporal memory -- high-order sequence prediction
by context-specific arc cells selected by prediction, learned by local
rules -- with its measured order and speed, and the statement that it is
episodic (TM-2's outcome) alongside. TM-2 passing would be a larger
result and would be re-registered before being claimed.

## Result, cells B (2026-09-09, 20 brains, n = n_arc = 4000, k = 100, 40 presentations)

    set   arm                 rank-1 all   ambiguous positions   brains perfect at 40   first perfect
    I     induced (registered)  1.000        1.000 (2 positions)    20/20                  1
    I     copy, g = 0           1.000        1.000                  20/20                  1
    I     copy, g = 4           1.000        1.000                  20/20                  1
    II    induced               0.998        0.995 (10 positions)   19/20                  1
    II    copy, g = 0           0.998        0.995                  19/20                  2
    II    copy, g = 4           0.858        0.770                   0/20                  2-4
    III   induced               0.957        0.525 (2 positions)     1/20                  1-2
    III   copy, g = 0           0.966        0.625                   5/20                  2-6
    III   copy, g = 4           0.957        0.525                   1/20                  13-16

    TM-4  set I at g = 4: 20/20                                        PASS
    TM-5  set III at g = 4, scored at presentation 40: 1/20            FAIL
    TM-6  set I first perfect: presentation 1 on every brain

**What the per-presentation curves show, and what the bar missed.** The
order-10 accuracy at the ambiguous positions, by presentation:

    induced        0.60 then 1.00 from presentation 2 through 31, then 0.42-0.53 (chance) from 32
    copy, g = 0    rises to 1.00 by presentation 7, holds to 26, decays from 29
    copy, g = 4    rises from 13, peaks 0.97 at 20, decays from 24

The registered transducer predicts an order-10 sequence exactly for
thirty consecutive presentations and loses it at presentation 32. That is
the clip edge measured on the S5 organ (PREREG_s5_cliff_anatomy.md,
Addenda 5 and 8: relocation between 28 and 30 presentations, c* =
ln(w_max) / ln(1 + beta) = 31.4 for a synapse potentiated once per
presentation); here each transition is potentiated once per presentation
and the collapse lands at 31-32. The registration fixed 40 presentations
from the spiking model's convergence time, which was the wrong number for
this substrate, and TM-5 fails for that reason alone. Scored inside the
window the induced state carries order 10 on every brain, and the
predicted-win rule adds nothing at order 2 or 10 and costs accuracy on
set II. The prefix hash is a temporal memory of unbounded order for
memorized sequences; what it cannot do is generalize across contexts
(cells A, the chain corpus), which is the same statement the literature
makes about its own models.

Nothing is adopted from cells B as registered. An amendment with the
presentation count inside the window (20) is registered below before
any re-run.

![accuracy at the order-10 positions against presentation, three arms, with the clip edge](../figures/organ_order10_window.png)

*How to read it: the fraction of twenty brains predicting the right
continuation after the ten shared words, against how many times the two
sequences were presented. Blue is the registered transducer, green the
state-as-previous-arc variant, red that variant with predicted neurons
winning at gain 4. The dotted line is c* = ln(20) / ln(1.1) = 31.4, where a
synapse potentiated once per presentation reaches the clip.*

## Amendment 1 (2026-09-09, before re-running): presentations inside the window

    TM-5'  set III at 20 presentations, induced state and copy g = 0:
           order-10 disambiguation on >= 15/20 brains at the final
           presentation. PREDICTION: PASSES on both (the curves above hold
           1.00 from presentation 7 to 26). The re-run is the same script
           with PRESENTATIONS = 20 and is a confirmation of the window, not
           a new claim; TM-5 as registered stays FAIL.

### Amendment 1 -- Result (2026-09-09, 20 presentations, 20 brains)

    set III at 20 presentations     order-10 positions   brains perfect
    induced state                   1.000                20/20
    copy, g = 0                     1.000                20/20
    copy, g = 4                     0.975                19/20

    TM-5'  >= 15/20 on the induced state and copy g = 0                  PASS
    Sets I and II at 20 presentations: 20/20 on every arm.

The window confirmed: what failed at 40 presentations is exact at 20, on
every brain, for all three constructions. The registered transducer is a
temporal memory of order at least 10 for memorized sequences, inside the
presentation window set by the clip.
