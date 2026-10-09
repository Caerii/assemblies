# Exploratory probes, 2026-10-07 to 2026-10-08: the sequence memory's budget

Every probe here is EXPLORATORY: not registered, run on four to twenty brains
with seeds outside every registration's (900-999), and reported only to
explain why a registration was designed as it was, never to support a claim.
Each one is disclosed in the registration it informed
([PREREG_refraction_memory.md](../../PREREG_refraction_memory.md), Amendments
37-43). The theory they feed is the notebook
[../../../../theory/assembly_statmech.tex](../../../../theory/assembly_statmech.tex)
(section "The sequence memory's budget") and the manuscript
[../../../../papers/drafts/sequence_budget/](../../../../papers/drafts/sequence_budget/).

Scripts are as run, with paths made relative to the repository. Run from the
repository root on a CUDA machine after `scripts\cuda-dev.cmd`, e.g.
`python research/notes/memory/probes/2026-10-08/probe_nk.py 15000,50,0.7`.
Where a log is missing the output was read from the terminal and is quoted in
the registration that disclosed it.

## Theory round (2026-10-07/08, before Amendment 37)

| Probe | Question | Result (as disclosed) | Informed |
|---|---|---|---|
| `probe_hebb.py` | Is the Hebbian length limit the write being captured? | The limit equals the step at which a new state overlaps an earlier one by 0.3, ratio 1.00-1.11 at six cells | sequence-limits memory note |
| `probe_onset.py`, `probe_onset_tau.py` | Capture vs replay as the limit, by tau | Capture limits only tau <= 4; at tau 16-256 replay fails first | Amendment 37 design |
| `probe_replay.py`, `probe_map.py` | The one-step overlap map F(o) under load | F - o peaks +0.07 at L 1600, +0.04 at 2300, < 0 at 3200 at (4000, 60): loss of the stable state | Conjecture (saddle-node) |
| `probe_kernel.py`, `probe_kernel2.py`, `probe_markov.py` | Is replay Markov in the overlap? | With a kernel from real transitions, yes (held-out horizon 132 vs 112); a random-filler kernel is 0.02-0.04 optimistic | hazard model |
| `probe_position.py`, `probe_ghost.py`, `probe_junk.py`, `probe_order.py` | Ghost tracks, junk attractors, position effects | None: the wrong part of a noisy state is fresh every step | hazard model |
| `probe_interf.py` | Is the interference Poisson? | Poisson at light load; outsider excess variance ~2x Poisson at L = 3200, counts 17% enriched | dispersion conjecture |
| `mf.py`, `mf2.py`, `mf3.py` | A reduced one-step model (selection, learned transition, interference, k-WTA) | Overlap maps to 0.02-0.03 | Lemma's limits |
| `mf_sn.py` -> `mf_sn.log` | The reduced model's critical load per cell | 0.23-0.34 against measured 0.12-0.17: factor 0.48 +- 0.05, correlation 0.65 | "the constant is measured, not derived" |
| `probe_survey.py` -> `survey.json`, `survey.log` | rho_50 at seven in-regime cells and one below the floor | 0.119-0.171 (mean 0.142); 0.086 below the floor | Amendment 37's prediction |

## After Amendment 37

| Probe | Question | Result | Informed |
|---|---|---|---|
| `oracle_big.py` -> `oracle_big.log` | Does the engine equal the independent oracle at n > 2^14? | Counts, bias and every recall equal at (18000, 60, 0.6) and (24000, 80, 0.5), 40 elements | Amendment 38 result |
| `probe_nk.py` -> `probe_nk.log` | Is Amendment 38's low cliff n/k or n? | n/k: (15000, 50, 0.7) ~0.088; (20000, 200, 0.3) ~0.137 | Amendment 39 |
| `probe_tau.py` -> `probe_tau.log` | Does tau move the cliff at n/k = 300? | 64: 0.088; 128, 256: ~0.115; 512: tiling deadline returns | Amendment 39 |

## After Amendment 40

| Probe | Question | Result | Informed |
|---|---|---|---|
| `probe_many.py` -> `probe_many.log` | Many short sequences vs one long one | Short ones fail graded, outlast the single cliff | Amendment 40's bars (rewritten before data) |
| `probe_lognk.py`, `probe_small.py`, `probe_dbg.py` (+ logs) | Does ln(n/k) replace ln n? Why does (4000, 400, 0.5) fail at step 0? | ln(n/k) refuted at n/k 10, 20; large kp/ln n leaves the linear regime | Amendment 41 design |
| `probe_taunk.py`, `probe_peak.py` (+ logs) | rho_50 against tau/(n/k) | Peak near n/k at n/k 20-33 (0.20 vs 0.12); none at n/k 300 | Amendment 41 |
| `probe_disp.py` -> `probe_disp.log` | Reuse dispersion against tau | Use-count var/mean 7-12 at n/k/4, 0.3-0.4 at n/k/2, 0.04-0.08 at n/k; a pair with 70 counts at n/k 300 | Amendment 41 disclosure |

## After Amendment 41

| Probe | Question | Result | Informed |
|---|---|---|---|
| `probe_vocab.py` -> `probe_vocab.log`, `probe_noise.log` | Recurring words; replay noise | Same-word overlap 0.04-0.10 (tokens); noise multiplies with load. Flaw: one word draw shared by all brains | Amendment 42 (per-brain draws) |
| `probe_uses.py` -> `probe_uses.log` | Uses per word or vocabulary size? | Neither predicts heavy reuse cleanly; reported, not judged | Amendment 42 |
| `probe_links.py`, `probe_links2.py` (+ logs) | The C -> S link budget | Set by the source area: doubling n_S changes A_50 by < 2%, doubling n_C multiplies it by 1.36-1.7 | links conjecture |
| `probe_disp2.py` -> `probe_disp2.log` | Does every knob act through interference dispersion? | No: rho_50 * kappa ranges 0.12-0.56 | conjecture failed, recorded |

## After Amendment 43

| Probe | Question | Result | Informed |
|---|---|---|---|
| `probe_survival.py` -> `probe_survival.log` | Where along a sequence does it first fail? | No failure in steps 1-5; hazard then FALLS along the sequence (~ j^-1.2 to -1.7); survivors' overlap climbs more slowly at higher load (critical slowing?) | two-phase conjecture refuted; notebook |
| `probe_signal.py` -> `probe_signal.log` | Is the tau peak on the signal side (successor selection)? | Successor structural input z = 1.3-2.0, slightly LOWER at the peak: no | signal conjecture refuted; notebook |
| `probe_grammar.py`, `probe_grammar2.py`, `probe_grammar3.py` (+ logs) | Does a consistent grammar make reuse cheap? | Reversed: b = 1, 2, 4 successors per word break replay at the WORD level (0.00-0.39) where random reuse holds (0.98); repeated transitions merge tokens (overlap to 0.42) | transition-repetition conjecture; next registration |

## After Amendment 44

| Probe | Question | Result | Informed |
|---|---|---|---|
| `probe_dose.py` -> `probe_dose.log` | Dose-response of repetition, (4000, 100, 0.35) | Tolerant: random reuse at R ~ 5 keeps 0.91; grammars fail only at b <= 4 | the edges move with the cell |
| `probe_dose2.py` -> `probe_dose2.log` | Dose-response at (12000, 80, 0.45) | R ladder 1.0 .. 4.0: 1.00 .. 0.00, edge near 2.5-3; at R ~ 2.3, U = 10/20/40: 0.95/0.86/0.19 | frequency matters at fixed R |
| `probe_product.py` -> `probe_product.log` | Is U x R the variable? | No: U-shaped at fixed U R (0.00/0.39/0.28; 0.18/0.89/0.21) | two mechanisms |
| `probe_edges.py` -> `probe_edges.log` | The two edges | Recurrence: 1.00/1.00/0.39/0.00/0.00 at U 10-80 (brains collapse); repetition at U = 10: 1.00/0.95/0.39/0.00/0.00 at R 1.0-5.7; f(R) g(U) predicts the interior points within ~0.1 | separable-budget conjecture; Amendment 45 |
| `probe_collapse.py` -> `probe_collapse.log` | What is a collapsed brain? | Failed replays funnel into 7-36 shared states (overlap 0.11-0.19 vs chance 0.0075); NOT hub neurons (usage ~1.1x mean) | spurious-attractor reading |
| `probe_control.py` -> `probe_control.log` | Can it be controlled? | Cross-validated lesion of the attractor neurons rescues partly failing brains (0.17 -> 0.82 ...); mild replay-time habituation (c 0.1, T 50) lifts collapsed brains 0.04 -> 0.46 and intact 0.74 -> 0.88; strong habituation kills replay | read-time control conjecture |
