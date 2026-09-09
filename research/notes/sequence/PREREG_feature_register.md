# PREREG: a feature register, the structured slot

Registered 2026-09-09, before implementing or running.

## Why

Two learning-rule constructions failed to carry one bit across two
distractors (PREREG_successor_state.md; PREREG_temporal_memory.md, cells
A pending), and the induced state, which predicts memorized sequences at
unbounded order, carries 0.011 of a 0.214 gap on unseen combinations. The
literature's local models are episodic in the same way. The language-organ
line has always taken the other route: structure supplies what learning
cannot, as roles, categories and grounded features. The intuition to
follow is that agreement is a REGISTER problem: one slot that holds the
subject's number until it is overwritten, written only by words that
carry that feature in the relevant role, and read by the arc.

## The construction

A REGISTER area `REG` (n_reg = n, k) with two feature stimuli, `feat_sg`
and `feat_pl`, and a fiber `REG -> ARC` at the organ's density, learned
Hebbian prev x new like the other organ fibers. A table `feature_of`
maps each word to a feature (sg, pl) or to none. On a tick, if the
current word has a feature, `REG` is projected from that feature's
stimulus (one round) and holds the result; if it has none, `REG` keeps
its winners. The arc is then the conjunction of LEX, STATE and REG. OUT
reads the arc and is teacher-forced as before. Nothing else changes.

The GATE is the table: which words write the register. On the chain
corpus the agreeing classes (AUX, VERB, PRON, TAG) write it and the
distractor nouns do not. This gate is supplied by structure, the way a
parser supplies a role; whether it can be learned is the next question,
not this one.

## Cells and bars

Chain corpus, gap 2 (PREREG_agreement_corpus.md, Amendment 1; oracle -
bigram 0.214), 20 seeds, n = n_arc = n_reg = 10,000, k = 200, the
registered transducer (induced state) plus the register.

    FR-1  GATED REGISTER closes the gap: MRR - bigram (paired) lower
          bound >= 0.15 (70% of 0.214).  PREDICTION: PASSES. The register
          is the oracle's number bit; the phase is in the word and state.
    FR-2  THE REGISTER CARRIES IT, not the state: with the state area held
          empty (state-blind) and the register kept, MRR - bigram lower
          bound >= 0.15.  PREDICTION: PASSES.
    FR-3  REGISTER-BLIND falls back: with the register held empty at test,
          MRR - bigram upper bound <= 0.05.  PREDICTION: PASSES.
    FR-4  THE GATE DOES THE WORK: an UNGATED register, written by every
          word that has a number (nouns included), MRR - bigram upper
          bound <= 0.05 -- the distractor overwrites the subject.
          PREDICTION: PASSES.
    FR-5  reported: agreement-site accuracy (the correct number class
          ranks first at AUX/VERB/PRON/TAG positions), gated vs ungated.

Adoption if FR-1, FR-3 and FR-4 pass: a register entry that a structured
slot -- one feature area, category-gated writes, read by the conjunction
-- carries an agreement feature across distractors on the assembly
substrate where three learning-rule constructions did not, and that the
gate, not the memory, is what the substrate lacks. FR-1 passing with FR-4
failing would mean the arc reads the register's LAST write regardless of
gating, which would be a different mechanism and is reported as such.
