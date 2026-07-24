# Science gaps & iteration toward compositional generalization

## Platonic target

A parser that **grounds** childhood input (CDS), **binds** roles compositionally on novel
holdout lemmas, **predicts** OOV continuations via bridge assemblies, and **systematically**
swaps agent/patient without collapsing — not one that merely passes three easy SVO probes.

## Known gaps (updated as we measure)

| Gap | Symptom | Status |
|-----|---------|--------|
| Easy probes saturate | `novel_easy` = 100% for all paths | Use `composition_battery` strain + bridge |
| Adaptive false alarms | 2000+ remedial sents when `acc=-1` | **Fixed** |
| CDS telegraphic-only at SENTENCES | strain 33% vs curriculum 100% | **Fixed** — CDS + generic SVO blend |
| Holdout lexicon leak | `bird` entered core_lexicons via distributional | **Fixed** — `lexicon_holdouts` + stage filter |
| Bridge eval misaligned | variable probe counts, open readout noise | **Fixed** — holdout bridge corpus + constrained readout |
| Bridge metric saturation | 3-way holdout-only readout → 100% all paths | **Improved** — +15 distractor words in readout pool |
| Dev strain seed flake | seeds 43–44 strain 33% (dev only) | **Open** — fuzzy load-bearing; investigate gates |
| `bridge_direct` still low | 0–43% on direct prefix probes | **Open** — open-vocab readout; training signal |

## Metrics (composition battery)

- **novel_strain** — holdout-heavy 6-probe role assignment (primary path discriminator)
- **systematicity** — agent/patient swap
- **bridge_oov_top5** — OOV next-token among holdouts + distractors (constrained readout)
- **bridge_direct_top5** — fixed prefix probes, open `predict_next` top-5
- **science_score** — strain 30%, systematicity 20%, bridge OOV 20%, bridge direct 15%, holdout bootstrap 15%

## Latest results (v5b battery, seed 42 — distractor readout pool)

| Path | science_score | train | novel_strain | bridge_oov | bridge_direct |
|------|---------------|-------|--------------|------------|---------------|
| **developmental** | **0.804** | 56s | 100% | **76.9%** | 0% |
| curriculum_only | 0.733 | 45s | 100% | 30.8% | 14.3% |
| chat_bootstrap | 0.696 | 75s | 100% | 23.1% | 0% |

Developmental path now leads on `science_score` and `bridge_oov` when holdouts stay OOV and bridge readout uses holdout+distractor pool.

### v5 (pre-distractor, seeds 42–44)

| Path | science_score | novel_strain | bridge_oov (holdout-only) |
|------|---------------|--------------|---------------------------|
| curriculum_only / chat | 0.850 | 100% | 100% (saturated) |
| developmental | 0.656 ± 0.17 | 67% | 100% (saturated) |

Seeds 43–44: developmental `novel_strain` = 33% (curriculum stays 100%). **Open.**

## Ablation (seed 42)

| Ablation | novel_strain | bridge_oov | train |
|----------|--------------|------------|-------|
| full | 100% | — | ~50s |
| **no_fuzzy** | **33%** | — | ~29s |
| no_babble | 100% | — | ~35s |
| no_adaptive | 100% | — | ~33s |

**Fuzzy surfaces are required for strain on seed 42.**

## Commands

```bash
Remove-Item Env:ASSEMBLIES_ENGINE -ErrorAction SilentlyContinue
$env:EMERGENT_DEV_CURRICULUM="1"
$env:EMERGENT_FAST_TRAINING="1"

uv run python research/experiments/compare_training_paths.py --seeds 42 43 44
uv run python research/experiments/ablate_developmental.py --seeds 42 43 44
uv run python examples/train_developmental.py --max-stage SENTENCES --fast
uv run pytest neural_assemblies/tests/test_acquisition.py -m "not slow" -q
```

## Next iterations

1. Re-run battery with distractor-pool `bridge_oov` (v5b) across seeds 42–46
2. Stabilize developmental strain on seeds 43–44 (fuzzy + SVO blend audit)
3. Ablations at seeds 43–44 for strain variance
4. Hard gate on `novel_strain` + `bridge_direct` once stable
5. Stage4 error-correction (`STAGE4_ERRORS`) as contrastive signal
