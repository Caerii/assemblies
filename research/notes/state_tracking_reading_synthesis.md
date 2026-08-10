# State tracking in the Assembly Calculus: the reading synthesis behind Phase A

**Written 2026-08-10, before any Phase A code.** Sources read in full or in
targeted depth: Dabagia, Papadimitriou & Vempala, *Computation with Sequences
of Assemblies in a Model of the Brain* (arXiv:2306.03812, Neural Computation
37(1); local PDF `research/literature/papers/dabagia2025_sequences.pdf`);
Mitropolsky & Papadimitriou, *Simulated Language Acquisition* (2507.11788,
local PDF); the vendored reference dynamics
`neural_assemblies/reference/nemo_numpy/{areas,fsm_network}.py` (mdabagia/nemo
port); the package's own `assembly_calculus/{fsm,ops,scaffold,pfa}.py`; and
this repo's failure record: `conjunctive_arc_measured.md`,
`arc_conjunction_has_no_operating_point.md`, `PREREG_context_beyond_bigram.md`,
`PREREG_context_recurrence_off.md`, plus the #56/#93 task history.

The purpose of this note is to replace four scattered open items (#87
prediction redesign, #92 FSM arc, #14 context buffer, #93 chain operating
point) with ONE architectural account and one registered build plan.

---

## 1. What the theory actually says

**Time lives in directed inter-assembly synapses.** (Thm 1.) A sequence is
memorized when Hebbian plasticity writes asymmetric bridges x_i -> x_{i+1}
in the direction of presentation. Recall = the chain plays itself forward
from any element. No clock, no buffer, no position code — the ORDER is the
weights.

**The regime has hard floors, and they are not our parser's.** Every theorem
assumes:

- `kp >= 3 ln n` — at n=10^5 that is kp ≈ 35. The paper's own FSM demo runs
  n=5000, k=70, **p=0.4** (kp=28). Our production parser runs kp=1.5. #93's
  complaint ("chains run at p=0.05, the bottom of the paper's own sweep") is
  not a nitpick; we have been running sequence protocols an order of
  magnitude below the theory's floor.
- **β sits in a WINDOW**: an upper bound so the caps formed on presentation 1
  never change across training (the proofs' load-bearing invariant — Lemma
  9/10), and for the FSM a LOWER bound `β >= sqrt(ln n / kp)` so transition
  writes bite in finite time. Too little plasticity and the machine never
  forms; too much and assemblies disperse. #56's non-monotone repetition
  result (2.33/3 at reps=3 falling to 1.00 by reps=8) is exactly what
  window violation looks like from below.
- **Per-round homeostasis** ("each neuron's incoming weights sum to 1") is
  assumed by EVERY sequence theorem. Not norm_init-once — continuous
  renormalization. Our E1/E9/deferred-scaling infrastructure is the local
  analog, and the CHILDES inversion taught us its failure mode; here the
  theory says some form of it is a prerequisite, not an option.
- Overlap discipline Δ <= k/(2 ln n)² between the items to be sequenced.

**State tracking is TRANSITION dynamics, not accumulation.** (Thm 4.) The
FSM construction is three areas — input I, state S, arc A — where the pair
(S_q, I_σ) co-fires into the arc to form a CONJUNCTION assembly A_{q,σ},
and A_{q,σ} projects back to S_{δ(q,σ)}. Learned by simply firing the triple
{S_q, I_σ} -> A_{q,σ} -> S_{δ(q,σ)} about 7/β times per transition —
teacher-forced by observation, no global supervisor. The clock that makes S
and A alternate is a pair of long-range interneuron populations wired
crosswise (Lemma 3), and in the authors' OWN demo code the clock is not
interneurons at all but **refraction** — `RefractedArea` in mdabagia/nemo
gives every fired arc neuron an adaptive bias (`bias[fired] += input * β`)
so the arc must move on. Our engine already has this primitive
(`refractory_period`, used by `ordered_recall`).

**Prediction is a transducer, and that is the whole of #87.** (Remark 5.)
Add one output area B; during training fire B_{θ(q,σ)} together with
S_{δ(q,σ)}. Afterward the (state, symbol) pair produces the OUTPUT two
steps later. Next-token prediction in an assembly LM is exactly this: the
output alphabet is the lexicon, θ is the corpus's next-word statistics, and
the "state" is however much context the state area carries. A bigram model
is the degenerate one-state case — which is precisely what our
PREREG-next-token study measured (0.2074 ≈ 81.6% of the span to the bigram
optimum, and nothing beyond it): **we built the stateless transducer and
then asked it for state.**

**Turing completeness is FSM + a tape of assembly chains.** (Thm 7 / Lemma
6.) Two three-area cycles (one per tape half) with LRI-gated add/delete, a
symbol area, ten areas total, n >= 2·max{T², |Q|²|Σ|²}. Nothing exotic: the
tape squares are a linked sequence of assemblies, i.e. Thm 1 again. We do
not need this for the LM milestone, but it bounds the ambition honestly:
regular structure now, bounded-stack (center-embedding, per 2206.13217)
later, tape only if we ever need it.

**Empirically the theory is conservative.** Sequence capacity in simulation
exceeds n/k (the disjoint-cap bound); FSM classification accuracy saturates
long before per-transition recall does; degradation with string length is
graceful. The mechanisms have headroom beyond the proofs.

---

## 2. Why OUR attempts failed — one account, three symptoms

Every failed sequence/state experiment in this repo violated the theory's
preconditions in an identifiable way, and no failure requires a new
mechanism to explain:

**#92 — "the arc drops the state" = frequency swamping (the #52 law) inside
the conjunction area.** The corrected measurement
(`arc_conjunction_has_no_operating_point.md`) shows A_{q,σ} degenerating to
A_σ, and names the mechanism: MOOD fired into ARC at every constituent
while any given state fired only when it preceded — ~3× the potentiation,
and the frequent conjunct takes the k-WTA. This is *identical* to what #52
measured this week in role space (gain's sign follows exposure
concentration; Hebbian mass follows frequency). The paper never hits this
because Thm 4's training presents every transition EQUALLY often, and the
reference's `FFArea.normalize()` (per-input-fiber column normalization —
each input area's weight matrix normalized separately) exists precisely to
equalize drive across input fibers. We ran a conjunction area with
unbalanced conjunct exposure and no per-fiber homeostasis. The arc did not
fail; our feeding of it did.

**#14 / PREREG II-III — the CONTEXT buffer collapse = the wrong
representation of state.** A recurrent accumulator area merged prefixes
(overlap 0.7566 across different prefixes, performance HALVED), and closing
recurrence during training recovered distinctness but learned nothing
beyond bigram — because an accumulator has no transition structure to
learn. The theory's answer is that context is not a bag that fills; it is a
STATE that transitions. The state area in the reference is *feed-forward*
(driven only by the arc) — it has no self-recurrence at all, so the
repo's documented collapse channel (recurrence during training) never even
applies. State persistence between steps is winner retention, which our
areas already do.

**#56 / #93 — single-area chains at parser kp = the wrong regime.** Intra-
area chaining (`sequence_memorize` + `ordered_recall`) fights the
within-assembly attractor with refraction and tops out at 2/3 with a
non-monotone training curve, run at kp 10-20× below the theory's floor. The
arc architecture routes transitions ACROSS areas, so there is no
self-attractor to fight; and the floor says the SEQ areas need their own
local p (per-fiber p exists on `numpy_exact` via `add_connectivity`; per-
fiber β everywhere via the #52 single-owner `set_base_beta`/overlay
machinery).

Two more corroborating pieces: the July observation that raising
`refractory_period` HURT context-prefix discrimination is consistent —
refraction belongs in the ARC (whose job is to move on), never in areas
whose identity must persist. And #97's engine fix (numpy_exact silently
skipping plasticity into a fixed area) was a hard prerequisite for
teacher-forced transition writes — the reference's `state.fire(next_state)`
idiom is exactly a plasticity-on write into a pinned target, which our
engine used to no-op.

**Bonus: #91's null is now scoped by the same reading.** SLA 2025's own
operating point is n=10^5, β=0.06, k_LEX=50, k_Ci=20, scene firing
throughout, τ repetitions per word, and m context areas C_i giving every
word a distributed semantic signature. Our toy ran n=10^4, twelve words, no
C_i, and its (now-corrected) stability probe matches the paper's Property 1
("fire it by itself"). The honest retest is at paper fidelity, and the C_i
areas the retest needs are the same "more context structure" investment as
Phase A.

---

## 3. The Phase A design this reading licenses

**One new organ, three areas, regime-local parameters:**

- `SEQ_STATE` — feed-forward area (no self-fiber). Holds the current
  context-state assembly. Written to only by the arc (and by teacher
  forcing during training).
- `SEQ_ARC` — the conjunction area. Inputs: SEQ_STATE and the current
  word's LEX/core assembly, and NOTHING else (the #92 lesson: "a
  conjunction cannot form in an area whose winners are already decided by a
  third input"). `refractory_period > 0` (the reference's clock). Sized
  generously relative to state/symbol (reference used n_arc 5-10× state).
  **Per-fiber input normalization ON** — the structural anti-swamping
  guarantee, so corpus frequency imbalance (Zipfian by nature, per #150)
  cannot hand the arc to the frequent conjunct. This is the reference's
  per-matrix `normalize()`, scoped per-fiber — NOT the per-column-over-
  everything scaling that inverted on CHILDES.
- `SEQ_OUT` — the transducer output (prediction). During training the
  observed next word's assembly is teacher-forced here alongside the state
  update; at test the arc drives it and the landing-channel readout scores
  it.
- Fibers `STATE->ARC`, `WORD->ARC`, `ARC->STATE`, `ARC->OUT` get
  regime-local p and β (window per §1), via existing per-fiber machinery.
  The rest of the parser is untouched — this is an organ beside the
  comprehension stack, not a rewrite of it.

**Training loop (per sentence):** reset state to q0; per word, fire
(STATE, word) -> ARC settles (refraction advances it), teacher-force
next-state and next-word: `bind`-idiom writes ARC->STATE' and ARC->OUT.
What IS the next state during acquisition? Start with the paper-faithful
answer for word order (state = last constituent/category — a small,
observable state alphabet, exactly #92's original setting), and for
prediction let SEQ_STATE be free-running (the state the arc itself
produces) with teacher forcing only on SEQ_OUT — measuring whether
useful state structure emerges is itself a registered question, with the
FSM-supervised arm as its control.

**Acceptance gates — all four instruments already exist, none need to be
built:**

1. `prediction_landing_surprise` on SEQ_OUT separates attested from
   unattested continuations (the #28 instrument, currently reading chance
   on an organ with no signal; bar >= 0.75 as in #121).
2. Next-token beats the **bigram optimum 0.2338** (PREREG II's H2, the bar
   that failed) — the definition of "holds context beyond one symbol."
3. #92's arc, retrained with balanced/normalized conjunct input: across-
   STATE arc overlap LOW on neuron IDs (the corrected probe), and the
   multi-mood word-order exam recovers its 24/32 baseline THROUGH the arc.
4. Paper parity: the mod-3 FSM (11 symbols, 5 states, 33 transitions) at
   the paper's own parameters, through OUR Brain engine, matching the
   vendored `reference/nemo_numpy/run_mod3_fsm_numpy` golden — the
   "FSM-from-sequences end-to-end" gap `docs/literature.md` already names.

**Pre-registered risks, from our own laws:** (a) high-kp SEQ areas put β on
the helpful side of the kp law (good) but raise crowding stakes — the #52
concentration rule predicts the arc needs its input balance far more than
it needs gain; (b) per-fiber normalization is a THIRD normalization policy
next to norm_init and deferred scaling — it must land under the one-owner
discipline (a single writer, scoped to SEQ_ARC's input fibers) or it will
recreate the beta-store two-writers bug; (c) torch/numpy recruitment
divergence at unequal area sizes (#62) matters here because SEQ areas are
deliberately size-asymmetric — the parity harness should run both engines
early.

**Sequencing within Phase A:** A1 build organ + mod-3 parity gate (pure
paper ground, no language); A2 re-run #92's word-order FSM through it
(balanced training + fixed readout); A3 wire SEQ_OUT to the lexicon and run
the prediction gates (#28 instrument + bigram bar); A4 only then revisit
#91 at paper fidelity with C_i areas, which by then exist as
infrastructure. Each step has a golden or a pre-registered bar before any
code.

---

## 4. What this changes in the standing phase plan

Phase A stops being "redesign prediction" and becomes "build the transition
organ" — prediction, FSM word order, context-beyond-bigram, and the #91
retest all fall out of one build. Phase B (per-form normalization for
CHILDES) gains a second customer: the arc's per-fiber normalization and the
morphology scaling fix are the same primitive family and should share an
owner. Phase C (#91 retest) is absorbed into A4. Phase D (DL baseline)
still waits for A3, because before A3 we have no next-token capability to
compare.

The one-sentence version for the whiteboard: **stop trying to make areas
remember; make transitions write synapses.** Time is a weight, state is an
assembly, context is a state machine — and every instrument needed to hold
this build to account is already on the shelf.
