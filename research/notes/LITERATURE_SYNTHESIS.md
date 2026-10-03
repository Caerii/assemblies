# The assembly-calculus literature and this repository: a synthesis

*2026-10-01. Read against the fifteen papers in
[`research/literature/index.json`](../literature/index.json). Nine were read
in full in this session (CCN 2019, TACL 2021, center-embedding 2022, the
language-organ architecture 2023, coin-flipping 2024, acquisition 2025,
E%-WTA 2026, speech 2026, causal 2026); for PNAS 2020, ITCS 2019, COLT 2022
and the sequences paper (2025) the page images did not render, and their
claims are taken from the repository's extracted claim registry ([REPRODUCTION_MATRIX.md](../literature/REPRODUCTION_MATRIX.md))
and the secondary accounts in the later papers. The planning paper (AAAI 2022)
and the dendritic-gating paper (Onasch et al. 2025) were read at abstract level
only. Statements marked "this repository" cite registered results in
[PREREG_refraction_memory.md](memory/PREREG_refraction_memory.md) and
[PREREG_word_capacity.md](aligner/PREREG_word_capacity.md).*

## 1. The field in one page

**The model.** Every paper shares one dynamical system: areas of n excitatory
neurons, Erdős–Rényi connectivity with probability p inside and between areas
("fibers"), k-cap (the k most driven neurons fire), and multiplicative Hebbian
plasticity w ← w(1 + beta) when presynaptic firing precedes postsynaptic.
Control is by inhibiting and disinhibiting areas and fibers. CCN 2019 names the
primitive random projection and cap (RP&C), adds homeostatic normalisation of
incoming weights, and proposes project, associate, pattern completion,
reciprocal project and merge, arguing merge underlies syntax. ITCS 2019 and
PNAS 2020 prove that projection converges to a stable assembly with high
probability for beta above a threshold, that association preserves overlap,
and (PNAS) that the calculus with control simulates O(sqrt n)-space
computation. The later "NEMO" papers keep the system and add long-range
inhibitory interneurons (LRIs) as control elements in place of programs, and
inter-area mutual inhibition.

**The threads.**

| Thread | Papers | What is shown |
|---|---|---|
| Foundations | CCN 2019, ITCS 2019, PNAS 2020 | RP&C; projection convergence above a beta threshold; association, merge; Turing completeness |
| Learning | COLT 2022 | assemblies classify well-separated distributions online |
| Sequences and control | Dabagia et al. 2025 | sequences of assemblies, finite-state machines, refractory adaptation for transition disambiguation, LRIs |
| Statistical learning | coin-flipping 2024 | an assembly "coin flip" samples a softmax of its input weights; a saturating rule min{alpha, e^(lambda(1+beta-w))} learns empirical frequencies; Markov chains and a trigram model; needs k ~ 500 |
| Syntax | TACL 2021, center-embedding 2022 | a dependency parser whose grammar is the lexicon's inhibit/disinhibit actions; constituency (S, VP areas); center embedding with a working-memory "touch" pass, giving the theorem fallback automata = context-free languages |
| Acquisition | architecture 2023, acquisition 2025 | nouns and verbs learned from grounded sentences (~10-20 sentences per word, linear in lexicon size); word order and generation through MOOD and role areas; the three rarest word orders are the hardest to learn |
| Planning | AAAI 2022 | blocks-world planning executed by assembly programs |
| Biological variants | E%-WTA 2026, dendritic gating 2025 | variable-size caps from gamma-cycle selection with inhibitory synapses; dendritic gating of plasticity for stable overlapping assemblies without forgetting |
| Applications | speech 2026, causal 2026 | refractory areas detect phone (F1 0.69) and word (0.61) boundaries without learning; per-class recurrent areas classify TIMIT phones (47.5%) and speech commands (45.1%); supervised causal direction on a known DAG (all 12 links) |

**Parameters the literature runs at.** PNAS: n = 10^4, k = 100, p = 0.01,
beta = 0.05 (k p = 1). CCN: n = 10^7, k = 10^4, p = 10^-3, beta = 0.05
(k p = 10). NEMO language papers: n = 10^5-10^6, k = 20-1000, p = 0.01-0.05,
beta = 0.05-0.1 (k p = 1-10, set per fiber by hand). Coin-flipping: n = 25,000
to 100,000, k = 500, p = 0.1 (k p = 50). E%-WTA: beta <= 0.01 best. Speech:
beta = 3e-4 to 3.6e-3 after Bayesian search.

## 2. Where the threads meet, read through this repository's results

### 2.1 Capacity is the field's blind spot, and refraction answers it

No paper measures how many assemblies one area holds. The reproduction matrix
lists ITCS's capacity scaling as missing. The two papers that look at
multiple assemblies both flag interference as the limit: E%-WTA reports that
later assemblies destroy earlier ones and retrieval becomes partial, and
dendritic gating exists to prevent forgetting. Every demonstration elsewhere
stores a handful of assemblies.

This repository measured it. A plain Hebbian area realises about 2% of the
associative capacity its synapses support, because repeat winners become
hubs and items merge (Amendment 9). A per-neuron usage penalty --
refraction -- raises that to about 50% and completes 14 to 57 times as many
distinct items as the Hebbian area at every one of ten cells (Amendment 14).

Refraction is already in the literature, for other reasons: the sequences
paper introduces refractory adaptation to disambiguate transitions, and the
speech paper uses refractory suppression to turn input change into assembly
change for boundary detection. The synthesis is that **the same mechanism the
literature adopted for time is also what makes an area a memory**. Three
anti-interference mechanisms now exist in the field -- a usage penalty
(refraction), variable-size inhibitory selection (E%-WTA, which reports lower
overlap), and gated plasticity (dendritic gating) -- and none has been
compared with another on a common capacity protocol.

### 2.2 The plasticity parameter: speed, capacity, and the convergence threshold

The literature's beta values span two orders of magnitude, chosen for
convergence speed: CCN notes that larger beta converges faster and makes
proofs simpler, and the acquisition paper finds learning accelerates roughly
inverse-exponentially in beta. Nobody measures what beta costs.

This repository found the cost. The beta that maximises distinct completion
obeys ln(1 + beta*) = 0.29 / sqrt(k p / 2) at p = 0.5, confirmed on new brains
and at two cells it had not seen (Amendment 13), and the field's typical 0.1
is 1.3 to 2.7 times too strong. With connectivity noise the law holds for
p = 0.125 to 0.75: ln(1 + beta*) = 0.285 sqrt(2 (1 - p)) / sqrt(k p / 2)
(Amendment 16, every bar passed, the dense-side ratio 0.707 against 0.71
predicted).

The connection to the theory: the projection-convergence thresholds of
ITCS 2019 and COLT 2022 scale as roughly sqrt(ln n / (p k)). Measured against
the form sqrt((1 - p) ln n / (p k)), the seven Amendment 13 optima are a
near-constant fraction of it, 0.20 with a coefficient of variation of 0.031,
tighter than the repository's own sqrt(k p / 2) law (0.052) and far tighter
than beta itself (0.218). (The threshold forms were taken from the two papers
in an earlier pass; they should be re-checked against the PDFs before being
quoted.) Read together: **capacity is maximised at the onset of reliable
completion, a fixed fraction of the convergence threshold the theory already
derives; speed wants beta above it.** That trade-off is why E%-WTA found small
beta best for recall while the acquisition paper found large beta best for
learning time. Each paper measured one side of the same trade.

### 2.3 Most published simulations run below the regime where the laws are clean

This repository's laws hold above the regime floor k p >= 3 ln n; below it,
cells fall 20-32% under their n/k pairs and fail to converge at low load
(Amendment 4). PNAS runs at k p = 1, the NEMO papers at 1-10, the lexicon of
this repository's word learner at 2.5; only the coin-flipping paper runs
above the floor (k p = 50, against 3 ln n = 30). The coin-flipping paper also
reports that faithful sampling "required substantially higher scale" (k = 500
against 30) and that sampling error falls with cap size. The common reading:
**the relevant width is the fan-in k p measured against ln n**, and the
published demonstrations are existence proofs at low load in a regime where
multi-item capacity would be small and noise-dominated. Scaling them is a
question of fan-in, not of neuron count alone.

Measured (PREREG_refraction_memory.md, Amendment 17, 20 brains): at the PNAS
2020 cell itself (n = 10^4, k = 100, p = 0.01) no learning rate from 0.1 to
0.4 of the convergence threshold completes a single item at the distinct
criterion; rank-1 identification peaks at 112 items. Below the floor at
p = 0.5 the memory does complete (about 1400 items at n = 4000, flat in k
from 10 to 40); at p = 0.05 it completes 160-200 items on twice the neurons.
The published operations are therefore identification-grade at the published
parameters, and below the floor sparseness costs capacity even at equal fan-in.

Amendment 20 sharpened this: across n = 10^4, p = 0.01, recall first appears
at k p = 4 while recognition holds 114-207 items from k p = 1; and read at its
own (much weaker) best write, recognition holds at least 32768 items where
recall holds ~1300 (n = 4000, k p <= 10). The published regimes are
recognition memories.

### 2.4 Per-fiber parameters are already used, by hand

The acquisition and architecture papers state that p and beta "may vary from
one area or fiber to the other" and draw some fibers bold with larger p and
beta. This repository's word learner sets one beta for its learning fiber,
and its best value depends on the lexicon's size and p (word capacity,
Amendment 4: the largest lexicon stores 517 words at beta = 0.05 against 324
at the registered 0.1). The learning-rate law is the principled version of
those hand choices for one area type; it is not yet validated for the
column-scaled learner, whose optimum also depends on load. **A per-fiber,
fan-in-scaled parameterisation (the muP analogue) is what would make the
language organs tunable at one size and transferable to another.**

### 2.5 Plasticity rules: the shape matters

The literature uses three rule shapes: multiplicative (1 + beta) with or
without a clip (most papers), the coin-flipping paper's saturating rule
whose increments shrink as weight grows, and the speech paper's ABS rule
with heterosynaptic depression. This repository found the rule's shape
decides the trade between identification and completion: a strong, clipped
(binary) write completes nearly everything it identifies; a weak write
identifies up to 4.4 (n/k)^2 items while completing none (Amendment 11). The
coin-flipping rule, which front-loads potentiation and then saturates, sits
near the binary end and should favour completion -- a testable prediction from
combining the two.

The rule's TIMING matters as much as its shape. Burst-timing-dependent
plasticity (Butts, Kanold & Shatz 2007, PLoS Biol 5:e61) potentiates
retinogeniculate synapses by the coincidence of bursts over a window of
about a second, order largely ignored; burst-dependent rules in cortex
(Payeur et al. 2021, Nat Neurosci 24:1010) separate potentiation by bursts
from depression by single events. In the refracted memory, a write that
acts on the item as a whole -- the round write's own counts, deferred to the
item's end, or a symmetric burst write -- stores no attractor at all, at any
rate over a 32-fold range, while the same counts written round by round store
451-1442 items (Amendment 25, [[WRITE-TIMING-DECIDES-ATTRACTOR]]). Without
the write's feedback, refraction relocates the assembly every round, and a
deferred write stores the item's trajectory instead (64-79% next-round
recovery). Gating the online write on bursts abolishes the memory too: the
write that converges an item is carried by first firings. A burst rule can
store items only if its window is short against the relocation period, or
if something else holds the assembly still while it writes.

### 2.6 Readout criteria decide what is claimed

Readouts across the literature: overlap with stored assemblies (PNAS),
lexicon look-up during parse readout (TACL), sampling frequency (coin
flipping), resonance score (speech), propagation overlap (causal). This
repository found that identification can grow while completion vanishes,
and that "completion" counts merged duplicates unless it also requires the
recall to be nearest its own item (Amendments 11, 12). A capacity or
accuracy claim needs its criterion stated, and completion needs the distinct
form.

### 2.7 Language: the grammar lives in control, the capacity in fibers

The TACL parser is a finite-state device whose grammar is the lexicon's
inhibit/disinhibit actions; center embedding needs a working-memory pass, and
formalising that pass gives exactly the context-free languages. The
acquisition model learns the lexicon and word order from grounded input with
about ten sentences per word. Both measure success and speed, not limits.
This repository's language results add the limits: a lexical area holds
about n/12 words, set by the fiber words are read through rather than by n/k
(word capacity, Amendment 3); its learning rate is mis-set for large
lexicons (Amendment 4). This repository also found that the parser's role separation survives
beta = 0 -- some of what looks learned is the random projection itself, which
the TACL readout cannot distinguish from learning.

### 2.8 Biology

Refraction is neural adaptation (sequences, speech, this repository);
E%-WTA ties cap size to gamma-cycle timing and inhibitory synapses; dendritic
gating ties plasticity to disinhibition; NEMO assumes synchrony and argues it
is not distortive. The capacity results add a functional role for adaptation
-- keeping stored items separate -- and a prediction: the best plasticity sits
just above the threshold where recall begins to work, and falls with the
square root of a neuron's fan-in.

## 3. What this repository adds, reproduces, and qualifies

| | |
|---|---|
| **Reproduces** | projection, separation, merge and the PNAS overlap metrics (pinned parity tests); the COLT separable-class toy; the TACL parser and word-order reference port; the acquisition model's noun/verb separation and word order (cross-situational learning needs homeostasis here) |
| **Adds** | the capacity of one area under four codes (Amendments 8-9); refraction as anti-merging and its 14-57x recall multiplier (14); identification versus completion and the rule-shape trade (11); the learning-rate law and its out-of-sample predictions (13); capacity invariance in p at equal fan-in (14); depth as over-writing (15); the GPU hashed substrate and the provenance runner that made 20-brain grids routine |
| **Qualifies** | "n/k alone" is a property of the operating write; published capacities need a criterion; beta values in the literature trade capacity for speed; most demonstrations run below k p = 3 ln n |

## 4. Agenda that the literature makes urgent

1. **The law in the literature's regimes.** Measure beta* at k p = 1-10 (PNAS,
   NEMO parameters). Does the optimum stay at a fixed fraction of the
   convergence threshold where fan-in is below ln n? *Answered by Amendment
   17:* above the floor, yes (0.18-0.19); below it, at p = 0.5 yes down to
   k p = 5, at p = 0.05 no (the fraction climbs to 0.30 at k p = 5), and at
   k p = 1 nothing completes. The optimum sits at the completion onset, the
   weakest write under which an item converges.
   *Measured directly by Amendment 18:* on a twelve-per-octave grid the
   onset is 0.16-0.18 theta at all ten cells (CV 0.04), dense and sparse,
   above and below the floor; what differs between regimes is how far the
   best write lies above it (one step dense, 1.6-1.8x sparse).
2. **One capacity protocol for every anti-interference mechanism.** Refraction,
   E%-WTA, dendritic gating, and the saturating coin-flip rule, each at its own
   best beta, on distinct completion.
3. **Per-fiber learning rates in the language organ**, set by the law and
   checked for transfer across two organ sizes; then the acquisition model's
   sentences-per-word under the tuned rates.
4. **Statistical learning at scale.** The coin-flipping softmax fidelity as a
   function of k p, on the hashed substrate.
5. **Theory.** The convergence thresholds are sufficient conditions; the
   measured optimum at 0.2 of one of them is a quantity a sparse-limit theory
   should derive.
