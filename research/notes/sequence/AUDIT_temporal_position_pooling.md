# Temporal arc position-pooling audit

Code inspected at f662e09 on 2026-09-10, before any new temporal GPU run.
SEQ-TEMPORAL-CARRY's TM-9 mechanism interpretation is suspended. The independently
collected prediction MRR and high-order task observations are not changed by this
audit. The historical 0.111/0.220 contrasts are not valid distractor-specific data.

## Code path and counterexample

seq_a3_transducer.a3_hashed collects every processed token's arc under its position.
_distractor_overlaps excludes position0 only, then averages all remaining pairs
using subject labels from position0. It never checks noun/distractor identity.
ntp_agree.generate_chain interleaves NOUN distractors with AUX/VERB/PRON/TAG words
whose number is explicitly the subject's number. The processed noninitial positions
therefore include directly informative words, not just distractors. Equal row counts
also stand in for sentence alignment, and unequal counts are silently omitted.

Construct four sentences, two per subject number. Give every distractor the same
arc, so its same-number minus different-number overlap is0. Give agreement-word
arcs disjoint number-specific sets. The historical two-position pool reports contrast
0.5 although the distractor contrast is0. This proves the metric can pass a mechanism
bar without any distractor representation. It does not determine the true corrected
contrast in the recorded brains.

## Required correction

Keep historical arithmetic explicitly named for audit, but refuse the former
_distractor_overlaps path and collect_arcs requests before GPU construction. Do not
silently write corrected values under the old TM-9 protocol. A replacement must
retain sentence identity, token, subject label, position and arc IDs; select actual
distractor tokens; report each position separately; reject dropped/misaligned rows;
and separate readout fitting from held-out evaluation. Register its controls/bars
before measuring. The old serialized mechanism summaries cannot resolve this
selection error into a corrected distractor-only number.

The performance observation still motivates representation-versus-readout research.
The prior statement that g=0 already carries a0.11 distractor contrast does not.
No new scientific outcome is adopted from the constructed software counterexample.


## Replacement observation contract

research/experiments/temporal_observations.py defines the CPU analysis boundary.
Build an expected manifest from the complete agreement-chain corpus and declared
gap. Derive subject number from AUX and token role from the fixed chain vocabulary;
validate the role sequence and agreement words. Exclude only the final prediction
target. Each frame must name sentence_id, position, token, subject_number and exactly
k unique in-range neuron IDs. The frame inventory must equal the manifest: reject
missing, duplicate, extra or mismatched rows rather than silently dropping sentences.

Report every processed position separately, with its derived role and distractor
flag. For each position retain same/different-subject pair counts and overlap means,
and their contrast. Both pair types must exist. These are within-brain pair means,
not independent seed replicates or confidence intervals. Retain frames and corpus
in the eventual run artifact; this analyzer returns summaries, not a storage format.
Frame ordering may change without changing results because identities are explicit.

Constructed agreement-only signal must yield zero contrast at every distractor;
subject-specific distractor arcs must change those contrasts. This validates the
instrument on known cases, not its scientific conclusion on trained neural systems.
GPU frame capture, learning-state checks, registration and empirical reruns remain
required. The old pooled collector stays blocked until that new path is validated.
