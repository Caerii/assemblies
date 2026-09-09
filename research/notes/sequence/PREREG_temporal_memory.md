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
