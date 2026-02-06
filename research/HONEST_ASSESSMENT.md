# Honest Assessment of Assembly Calculus

## ELI5: What Is This?

Imagine a room with 1000 light bulbs. When you show a picture of a CAT, 50 specific bulbs light up. When you show DOG, a different 50 bulbs light up.

These 50 bulbs = an "assembly" = the system's representation of that concept.

**The rules:**
1. Only 50 bulbs can be on at once (winner-take-all)
2. Bulbs that fire together wire together (Hebbian learning)
3. Show CAT many times → same 50 bulbs always light up

## What It Actually Is

**A lookup table, not a brain.**

```
Input CAT → Output neurons [0,1,2,3...]
Input DOG → Output neurons [100,101,102...]
```

The "learning" just makes the lookup more reliable.
The "associations" just link one lookup to another.

## The Holes in Previous Claims

### Hole #1: Assemblies Are NOT Attractors

**Claimed:** "Assemblies are stable patterns stored in weights"

**Reality:** Assemblies ONLY exist while you keep showing the input!

```
CAT neurons after training: [0,1,2,3,4...]
After self-projection (no CAT input): overlap = 6%
```

The CAT assembly DISAPPEARS when you stop showing CAT. This is stimulus-response, not memory.

### Hole #2: Sequence Model Was Cheating

**What I did:**
1. Show A, record which neurons fire
2. Show B, record which neurons fire
3. Learn: when A fires, activate B

**The problem:** I told the system when to show A, when to show B, and that A comes before B. 

The system didn't LEARN the sequence from data. I PROGRAMMED the sequence into the weights.

### Hole #3: Biological Validation Was Circular

**What I did:**
1. Set k=50 neurons active (my choice)
2. Set n=1000 neurons total (my choice)
3. Calculated sparsity = 5%
4. Said "This matches biology!"

**The problem:** I DESIGNED it to match. This proves nothing.

## Honest Capabilities

### What It CAN Do (with continuous input)

| Capability | How It Works | Limitation |
|------------|--------------|------------|
| Pattern Recognition | Show CAT → get CAT assembly | Need to show input |
| Associations | CAT picture → word "cat" | Need to show picture first |
| Composition | RED + APPLE → RED-APPLE | Need both inputs active |
| Sequence Prediction | A → predict B | Need to provide A |

### What It CANNOT Do

| Capability | Why Not |
|------------|---------|
| Recall without input | Assemblies disappear without stimulus |
| Imagination | Can't generate novel patterns |
| Autonomous sequences | Can't play A→B→C without prompting |
| Pattern completion | Can't fill in missing parts |

## The Real Question: Is This Useful?

### YES - With Continuous Input
- Real-time perception (always have visual input)
- Language processing (always have word input)
- Sensorimotor control (always have sensory input)

### NO - For Autonomous Cognition
- Planning (thinking without external input)
- Memory recall (remembering without cues)
- Dreaming (running without any input)

**The brain does BOTH. This system only does the first.**

## What Would Make It Better?

To get true attractor dynamics, we would need:

1. **Simultaneous intra-area learning**: When stimulus activates area, strengthen connections WITHIN the area (not just stimulus→area)

2. **Asymmetric weight growth**: Internal assembly weights must grow faster than external weights

3. **Active inhibition**: Suppress non-assembly neurons, don't just let them lose competition

4. **Recurrent stability**: Design learning rule so self-projection maintains the assembly

## Conclusion

Assembly Calculus (as implemented) is a **stimulus-driven pattern matcher**. It's useful for real-time processing with continuous input, but it's not a memory system or an autonomous cognitive architecture.

The framework may be theoretically capable of more, but this implementation realizes only the stimulus-response aspect.

