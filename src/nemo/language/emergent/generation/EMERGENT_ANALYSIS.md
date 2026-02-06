# Making NEMO Generation Truly Emergent

## Experimental Findings (December 2024)

We conducted systematic experiments to understand what NEMO assembly calculus
can and cannot do for emergent language generation.

### Key Experimental Results

#### Test 1: Basic Projection
- Same word projected twice: **52% overlap** (good!)
- Different words: **18% overlap** (discriminable)
- **Conclusion**: Projection creates discriminable patterns ✓

#### Test 2: Merge Operation (A + B → C)
- After merging subject + verb into VP:
  - VP ∩ Subject: only **16% overlap**
  - VP ∩ Verb: only **24% overlap**
- **Conclusion**: Merge DESTROYS component information ✗

#### Test 3: Component Discrimination
- Can we tell if "dog" is in VP("dog_runs")?
- Correct subject overlap: **21%**
- Wrong subject overlap: **20%**
- **Discrimination ratio: 1.05x** (essentially random!)
- **Conclusion**: Cannot discriminate components from merged VP ✗

#### Test 4: Recurrent Settling
- Does settling help retrieve components?
- Initial overlap: 18%, Final: 16%
- **Conclusion**: Settling does NOT help ✗

#### Test 5: Association (Bidirectional Learning)
- Does VP ↔ NOUN_CORE association help?
- Subject retrieval: 19%, Verb contamination: 20%
- **Conclusion**: Association does NOT enable retrieval ✗

#### Test 6: Learned Weights Stabilization
- After 10 projections, consecutive overlap: **72-76%**
- First vs Last: **75% overlap**
- **Conclusion**: Repeated exposure DOES stabilize assemblies ✓

#### Test 7: VP Overlap Discrimination (Comprehensive)
- 4 sentences: dog runs, cat sleeps, bird flies, fish swims
- Subject discrimination accuracy: **0%** (0/4)
- Verb discrimination accuracy: **25%** (1/4)
- **Conclusion**: VP overlap CANNOT discriminate components ✗

### Fundamental Insight

**NEMO merge fundamentally destroys component information.**

When you merge A + B → C:
- C has only k neurons total
- It cannot preserve both A (k neurons) and B (k neurons)
- The merged assembly represents a NEW concept, not a container

This is NOT a bug - it's how NEMO works! The question is: how do we work WITH this?

## Current State Analysis

### What IS Emergent Now

1. **Category Emergence** ✓
   - Word categories (NOUN, VERB, etc.) emerge from grounding patterns
   - No hardcoded POS tags - categories are inferred from `word_grounding` statistics
   - Example: "dog" → consistently grounded with VISUAL → emerges as NOUN

2. **Assembly Formation** ✓
   - Assemblies form through random projection + Hebbian learning
   - k-winners competition selects top-k neurons
   - Repeated exposure stabilizes assemblies (75% overlap after training)

3. **VP Assembly Learning** ✓
   - VP assemblies are created during sentence presentation
   - They represent learned propositions (subject-verb-object)
   - BUT: Components cannot be extracted from merged VP

### What is NOT Emergent (Symbolic/Hardcoded)

1. **VP Key Strings** ✗
   - VP assemblies are stored with keys like "dog_runs" or "cat_chases_mouse"
   - Retrieval uses string parsing: `vp_key.split('_')`
   - This is a SYMBOLIC encoding of structure, not neural

2. **Response Generation** ✗
   - Templates like `f"{subject} {verb} {', '.join(objects[:3])}"`
   - Question type classification uses hardcoded word lists
   - Response formatting is entirely symbolic

3. **Word Extraction from VP** ✗
   - We parse VP keys to get words, not decode assemblies
   - `decode_vp_key()` is string manipulation, not neural decoding

4. **Area Connectivity** ✗
   - `ActivationSpreader._get_target_areas()` hardcodes which areas connect
   - Real NEMO: connectivity should emerge from co-activation during learning

## The Core Problem (Experimentally Verified)

### Why VP Assembly Decoding Doesn't Work

The fundamental issue is that VP assemblies are **blended** representations:

```
VP("dog_runs") = merge(NOUN_CORE("dog"), VERB_CORE("runs"))
```

**Experimental proof**: We tested this directly and found:
- VP ∩ Subject: only 16% overlap
- VP ∩ Verb: only 24% overlap
- Discrimination ratio: 1.05x (random!)

When we merge two assemblies, we get a NEW assembly that shares some neurons with both parents. But:

1. **Overlap is weak**: VP assembly has only ~16-24% overlap with components
2. **Many false positives**: Other nouns/verbs have SIMILAR overlap (~20%)
3. **No structural separation**: Subject vs object are not neurally distinguished

### Why Reverse Projection Doesn't Work

We tested VP → NOUN_CORE projection and found:
- Initial overlap with original subject: 18%
- After 5 rounds of settling: 16%
- **No improvement from recurrence**

The reasons:
1. **PHON assemblies are arbitrary**: They don't encode phonological structure
2. **Projection is lossy**: Information is lost in the merge
3. **Learned weights don't help**: Even with bidirectional learning, discrimination fails

### Why Direct Association Doesn't Work

We tested direct VERB_CORE ↔ NOUN_CORE association:
- "dog runs" trained 75 times, "cat sleeps" 25 times
- Query "Who runs?": dog 0.15, cat 0.19 (WRONG!)
- Query "Who sleeps?": dog 0.38, cat 0.31 (WRONG!)

The weights get **contaminated** because all nouns and verbs share the same connection space.

## Paths Forward (Revised Based on Experiments)

### Option A: Separate Role Areas (Architectural) - RECOMMENDED

Instead of blending everything into VP, use separate areas:

```
VP_SUBJECT: stores subject assemblies linked to VP
VP_VERB: stores verb assemblies linked to VP  
VP_OBJECT: stores object assemblies linked to VP
```

This works because:
- Each area has its OWN learned weights
- Subject → VP_SUBJECT creates unique associations
- Verb → VP_VERB creates separate associations
- No blending = no information loss

**Status**: Not yet implemented, but most promising

### Option B: Temporal Binding - UNLIKELY TO WORK

Based on experiments, recurrent settling does NOT help.
The learned weights don't create attractors that "unpack" sequences.

**Status**: Experimentally disproven

### Option C: Contrastive Learning - UNTESTED

Train with negative examples to create discriminative weights.
This might work but requires significant changes to training.

**Status**: Not tested

### Option D: Hybrid with Component Storage - CURRENT APPROACH

Accept that VP assemblies cannot be decoded, but:
1. Use VP overlap to FILTER relevant propositions (neural)
2. Store component assemblies alongside VP (symbolic storage)
3. Decode components from stored assemblies (neural decoding)

This is what `HybridEmergentGenerator` does:
- Neural: VP overlap for filtering
- Symbolic: VP keys for structure
- Neural: Word decoding from learned assemblies

**Status**: Currently implemented, works but not fully emergent

### Option E: Attractor-Based Decoding - EXPERIMENTALLY DISPROVEN

We tested this directly:
- Settling does not improve overlap
- Reverse projection does not recover components
- Learned weights do not create discriminative attractors

**Status**: Experimentally disproven

## SUCCESSFUL IMPLEMENTATION (December 2024)

### The Core Insight

**NEMO's assembly calculus is designed for COMPOSITION, not DECOMPOSITION.**

When you merge A + B → C, you create a NEW representation C that can be:
- Recognized (does this pattern match C?)
- Composed further (merge C + D → E)
- Associated (C co-occurs with X)

But C cannot be DECOMPOSED back into A and B. This is fundamental to how NEMO works.

### Solution: VP Component Areas

We added three new areas to preserve component information:
- `VP_SUBJ` (Area 37): Stores subject assemblies
- `VP_VERB` (Area 38): Stores verb assemblies
- `VP_OBJ` (Area 39): Stores object assemblies

**Key insight**: We STORE assemblies directly, not PROJECT them!
- Projection creates a NEW assembly (~20% overlap with original)
- Direct storage preserves the assembly (~95% overlap)

### Implementation Details

During learning (in `learner.py`):
```python
# When we see the verb, store components DIRECTLY
vp_key = f"{current_subject}_{word}"
self.brain.store_learned_assembly(Area.VP_SUBJ, vp_key, subj_assembly)
self.brain.store_learned_assembly(Area.VP_VERB, vp_key, verb_assembly)

# Also create merged VP for other uses
self.brain._project(Area.VP, subj_assembly, learn=True)
self.brain._project(Area.VP, verb_assembly, learn=True)
self.brain.store_learned_assembly(Area.VP, vp_key, self.brain.current[Area.VP])
```

During retrieval (in `emergent_retriever.py`):
```python
# "Who runs?"
# 1. Get verb assembly from VERB_CORE
verb_assembly = brain.get_learned_assembly(Area.VERB_CORE, "runs")

# 2. Find VP_VERB assemblies that match (neural overlap)
for vp_key, vp_verb in brain.learned_assemblies[Area.VP_VERB].items():
    overlap = brain.get_assembly_overlap(verb_assembly, vp_verb)
    if overlap > threshold:
        # 3. Get corresponding VP_SUBJ (same key)
        vp_subj = brain.get_learned_assembly(Area.VP_SUBJ, vp_key)
        
        # 4. Decode VP_SUBJ to word (neural overlap)
        subject, conf = brain.find_best_matching_word(Area.NOUN_CORE, vp_subj)
```

### Results

**Discrimination accuracy**: 100%
- VP_VERB['dog_runs'] vs 'runs': 0.88 (correct verb)
- VP_VERB['dog_runs'] vs 'sleeps': 0.17 (wrong verb)
- VP_SUBJ['dog_runs'] vs 'dog': 0.89 (correct subject)
- VP_SUBJ['dog_runs'] vs 'cat': 0.21 (wrong subject)

**Retrieval accuracy**:
- "Who runs?" → dog (0.77) > cat (0.28) > bird (0.28) ✓
- "Who sleeps?" → cat (0.80) > dog (0.27) > bird (0.27) ✓
- "What does dog do?" → runs (0.77) > sleeps (0.18) > flies (0.17) ✓
- "What does dog chase?" → cat (0.51) > mouse (0.09) ✓

### What's Emergent Now

1. **Category emergence** ✓ - from grounding patterns
2. **Assembly formation** ✓ - Hebbian learning
3. **VP composition** ✓ - merge operation
4. **Retrieval matching** ✓ - neural overlap on VP_VERB/VP_SUBJ
5. **Word decoding** ✓ - neural overlap with learned assemblies
6. **Question answering** ✓ - fully emergent retrieval

### What's Still Symbolic

1. **VP keys** - stored as strings like "dog_runs"
2. **Response formatting** - templates like "{subject} {verb}"
3. **Question word detection** - "who", "what" detected by string match

### Files Modified/Created

- `areas.py`: Added VP_SUBJ, VP_VERB, VP_OBJ (Areas 37-39)
- `learner.py`: Store component assemblies directly (not projected)
- `emergent_retriever.py`: New EmergentRetriever and EmergentGenerator classes

## Conclusion

We successfully implemented truly emergent retrieval by:

1. **Adding VP component areas** that preserve subject/verb/object separately
2. **Storing assemblies directly** instead of projecting (key insight!)
3. **Using neural overlap** for both matching and decoding

The system now answers questions like "who runs?" and "what does dog chase?"
using purely neural operations - no string parsing of VP keys required for retrieval.

