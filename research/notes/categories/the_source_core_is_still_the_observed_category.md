# RETRACTED: the source core is the MECHANISM, not a confound

**This note's original claim -- that `expected_slot` had missed a third consumer
and that fixing it moved the honest P600 from 0.7167 to 0.6056 -- is withdrawn.
The change was adopted and reverted the same day.** The original text is kept
below the line so the retraction can be checked.

## What was claimed

`measure_live_integration` reads "the category" three times: `role_area` (probe
TARGET), `phrase_category`, and `core` (probe SOURCE). #108 fixed the first two.
So the violation arm reads VERB_CORE -> ROLE_PATIENT while its control reads
NOUN_CORE -> ROLE_PATIENT -- which looked exactly like #108's area-identity
confound one level over, committed by the fix for that very defect class.

## Why it is wrong

`anchored_p600_live`'s own docstring states the design:

> a category violation routes a **wrongly-typed core** through an **untrained
> pathway** and delivers LESS

**The source core differing between arms IS the mechanism.** That is what a
category violation physically is in this architecture. #108's problem was
different *in kind*: `VP -> VP` is unmaterialized, so that probe read a fiber
that does not exist. A live area reached by an untrained pathway is not a dead
probe.

## The control was not condition-constant, and that is the real lesson

`erp_source_core_identity_control.py` forced a trained noun's category to VERB
and reported AUC 0.78 / 0.89 / 0.89, described in its own docstring as *"only the
category LABEL handed to the metric changed"*.

False. `_advance_incremental_word`:

```python
core_area = CATEGORY_TO_CORE.get(cat, self._word_core_area(word))
project(self.brain, phon, core_area, rounds=rounds)
```

`forced_category` is not an annotation -- it is a **projection target**. Forcing
VERB puts the noun in VERB_CORE and reads the untrained pathway. **The control
manufactured a genuine category violation and then reported the metric detecting
it as evidence of a confound.**

> A control is only condition-constant if you have checked what its knob *does*,
> not what its parameter is *called*.

Same shape as [[same-name-two-meanings]], and this time it cost an adoption.

## What survives, read the other way round

- **Forced-typing, AUC 0.78-0.89.** The pathway mechanism is load-bearing: type
  a trained noun as a verb, same sentence, same position, and the P600 rises.
- **Lesioning the source, 0.7167 -> 0.6056.** Matching the source makes the
  violation arm read a STALE SUBJECT assembly in NOUN_CORE instead of the
  critical word at all. That 0.1111 is a lower bound on how much of the effect
  travels through the designed route.

`protocol.expected_slot_source_core` is kept, **off**, as exactly that lesion.

## The honest P600 is 0.7167 again

| | p600 AUC | matched |
|---|---|---|
| originally reported | 0.9056 | nothing |
| `f79c4f5` | **0.7167** | probe TARGET area |
| retracted | ~~0.6056~~ | removed the mechanism |

Standing caveats unchanged: AUC granularity is 1/9, and the DEFAULT frames still
contain untrained words (`the_calibration_frames_are_untrained.md`).

## The control that is still owed

Run the same frames on an **untrained or shallow parser**, which changes no
typing at all. If the separation survives there it is structural; if it collapses
it is learned. That is the degenerate control this episode should have started
with, and it is what the trained frames' 1.0000 still needs.

---

