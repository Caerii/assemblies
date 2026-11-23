# R1L — Twenty rounds of continual learning, twenty sensitivity stages

**Date:** 2026-11-23. **Research branch:** \`research/multimodal-continual-assemblies\`.
**Status:** 20-round synthetic stress test executed locally, 20-stage ablation campaign executed, exact compressed manifest committed and replayed. **NOT native \`Brain\` evidence.**

## Scope and methodology

This is a *synthetic task calibrator*, not a neural-language intelligence result. Two semantic role labels (SUBJ and OBJ) apply hidden modular shifts to k-sparse codes. An explicit shift-hypothesis learner may infer the role shift from observed cue/target examples; a plain Hebbian coactivation table and input-copy baseline cannot directly infer that shared operator.

Each **learning trial** performs precisely 20 sequential writes, alternating roles. Both hidden shifts change after update 10. Every round evaluates withheld, code-disjoint probes *before* and *after* writing the next labeled example. Rounds 7 and 16 deliberately supply a wrong training label to stress robustness. The evaluator hashes the learned state before and after probes, and fails on read-induced state mutation. The model never receives the private shift table or the withheld probe labels.

Model arms:

- \`global_rule\`: correct cyclic-shift hypothesis; retains all demonstrations;
- \`recent_rule\`: same hypothesis; retains most recent 3 examples per role;
- \`last_rule\`: same hypothesis; retains only the last example per role;
- \`hebbian_pairs\`: train-only pairwise additive coactivation, no shared-rule hypothesis;
- \`copy_input\`: no learning.

The hypotheses and memory budgets differ between models. This benchmark tests a stability/plasticity trade-off **within a known transformation family** and is not a fair fixed-budget architecture competition.

## Baseline twenty-round results

Default: \`n=128, k=8, seeds=0..19\`, five disjoint probes per role, \`change_after=10\`, poisoned rounds 7 and 16.

| Metric | Global history | Recent 3 | Last 1 | Hebbian pairs | Copy input |
|---|---:|---:|---:|---:|---:|
| Mean exact held-out recall, rounds 11–20 | **15.5%** | **79.5%** | **85.0%** | **0%** | **0%** |
| Exact current-regime recall, round 20 | **50%** | **100%** | **100%** | **0%** | **0%** |
| Recall of OLD regime, round 20 | **50%** | **0%** | **0%** | **0%** | **0%** |

The old-regime retention metric uses the earlier hidden rule on the same withheld probes: it is *not* recall of arbitrary episodic memory. Short windows rapidly adapt by overwriting previous evidence, while all-history scoring resists the rule change and retains older behavior.

Seed-level paired bootstrap over seeds 0–19 (2,000 resamples, seed 0), post-change exact accuracy:

- Recent 3 **minus global history:** +0.640 (descriptive percentile 95% interval **[0.615, 0.665]**)
- Recent 3 **minus Hebbian pairs:** +0.795 (**[0.775, 0.815]**)
- Recent 3 **minus last 1:** −0.055 (**[−0.075, −0.035]**)

These are descriptive intervals for this deterministic synthetic seed cohort, **not** population inference or evidence of real-world cognitive performance.

## Twenty controlled sensitivity stages

Each stage runs its own 20 sequential updates for 20 seeds. **20 stages × 20 seeds × 20 updates = 8,000 synthetic learning updates.** Held-out probes are disjoint from training codes.

| Stage | Variation | Recent-3 post-switch exact | Global post-switch exact |
|---|---|---:|---:|
| 00 | Baseline | 79.5% | 15.5% |
| 01 | No wrong labels | 75.0% | 7.0% |
| 02 | Wrong label before switch only | 79.5% | 17.5% |
| 03 | Wrong label after switch only | 75.0% | 5.0% |
| 04 | Window = 1 | 85.0% | 15.5% |
| 05 | Window = 2 | 82.0% | 15.5% |
| 06 | Window = 4 | 68.5% | 15.5% |
| 07 | Window = 5 | 54.0% | 15.5% |
| 08 | Window = 8 | 25.0% | 15.5% |
| 09 | Window = 10 | 15.5% | 15.5% |
| 10 | Rule change after round 6 | 77.5% | 53.2% |
| 11 | Rule change after round 14 | 47.5% | 0.0% |
| 12 | n=64, k=4 | 79.5% | 15.5% |
| 13 | n=64, k=8 | 79.0% | 15.0% |
| 14 | n=128, k=4 | 79.5% | 14.5% |
| 15 | n=128, k=16 | 78.5% | 16.0% |
| 16 | n=256, k=8 | 80.5% | 15.5% |
| 17 | n=256, k=16 | 80.5% | 15.5% |
| 18 | Wrong labels near boundary (9, 12) | 73.5% | 15.0% |
| 19 | Wrong labels exactly after switch (11, 12) | 62.0% | 0.0% |

The column labeled “Recent-3” refers to the experiment's *recent-rule arm*, whose window changes for stages 04–09. Those rows therefore report window-specific results, not a fixed window of three. Changing switch timing also changes the number of post-switch rounds; stage-to-stage averages are **not directly duration-matched**.

**Interpretation:** the observed regime strongly rewards discarding stale demonstrations after shift changes. But the "last 1" learner performs better on current-regime exact accuracy in the baseline, at the expense of vulnerability to single corrupted samples. No optimal window should be selected from these exploratory comparisons without a *new* held-out design.

## Executable code

\`\`\`bash
# Single 20-round synthetic stream across 20 seeds
python -m research.multimodal.continual_twenty \
  --output /tmp/r1l_20rounds.json

# Paired seed analysis
python -m research.multimodal.analyze_twenty \
  /tmp/r1l_20rounds.json --output /tmp/r1l_analysis.json

# Twenty sensitivity stages, each with 20 sequential updates and 20 seeds
python -m research.multimodal.campaign_twenty \
  --output /tmp/r1l_20stage_manifest.json

# Verify Git-archived compressed manifest; --recompute reruns all 8000 writes
python -m research.multimodal.verify_twenty_evidence --recompute

# Offline tests; GitHub Actions NOT required
python -m pytest -q research/multimodal
\`\`\`

## Evidence provenance

- Full single-baseline 20-seed, 20-round JSON SHA-256:
  \`ad39ea9c8b29cb39c05c417f4406cde9dd462655920fe9caf900eda08d9226a1\`
- Twenty-stage manifest JSON SHA-256:
  \`35b42065dfff39a16133ad6857302c6bc9e616ea2ab54e5e24d5c694a44980d1\`
- Compressed manifest \`json.xz\` SHA-256:
  \`7bb159ead79ba342cbe49d9773ee09480ee271bdf199e6234fbb848c8e929270\`
- Git-tracked base64 archive:
  \`research/multimodal/evidence/r1l-20stage-manifest.json.xz.base64\`.
  Its fetched Git blob SHA is \`6ecccb5fc4d324da332b909b5595de3eb6a602e2\`, matching the exact locally generated bytes including newline.
- Full source + tests (8 new \`*.py\` files) were compared against GitHub blob IDs; the locally tested versions match the branch. A local copy of \`teacher_control_matrix.py\` was found **stale** on an integrated test run; it initially caused 8 failures. After reconciling its package-relative \`learner_view\` import to the already-correct GitHub version, **82 integrated tests passed** in the local research-only mirror.

A full replay initially raised a false mismatch because Python tuple configuration values become JSON lists. The verifier was corrected to compare **canonical serialized JSON**, and the exact 8,000-update recomputation passed. Importantly this records a REAL failed validation and its remediation, rather than hiding it.

## Explicit research limitations

- These models do not execute the native \`Brain\`, real assembly Merge, refraction, or neuromorphic circuits.
- The role transform family (cyclic shift) is supplied to the successful models; their success is not spontaneous compositional discovery.
- The benchmark is run in one Python process without adversarial isolation. Small seeds could be brute-forced; this is a scientific leakage check, not a security sandbox.
- The 20 exploratory stages are not an independently preregistered confirmatory experiment.
- Geometry stages and varying switch timings change task difficulty and exposure budgets; the table is descriptive.
- Local research-only tests are NOT the entire repository’s native test suite; no Rust/CUDA/FPGA performance claims follow.

**Next:** execute R1b–R1j native gates in a complete Git checkout, then adapt this prequential protocol to a teacher-driven native \`Brain\` implementation. Compare its actual learned synaptic changes, strict exact held-out accuracy, forgetting and read-only state fingerprints against the strong symbolic rule baseline and the Hebbian control.
