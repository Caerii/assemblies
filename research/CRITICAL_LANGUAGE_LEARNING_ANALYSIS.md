# Critical Analysis: What's Wrong with Our Language Learning Approach

## The Core Problem

**We are NOT doing Assembly Calculus language learning. We are doing a hybrid system with lots of hardcoded knowledge that Assembly Calculus is supposed to LEARN.**

## What We Hardcoded (That Should Be Learned)

### 1. **POS Lexicon** (Lines 250-291)
```python
self.pos_lexicon = {
    'the': 'DET', 'dog': 'NOUN', 'runs': 'VERB', ...
}
```
**Problem**: A real system should LEARN that "dog" is a NOUN by observing:
- "the dog" (DET + X → X is likely NOUN)
- "dog runs" (X + VERB → X is likely NOUN/subject)
- "sees the dog" (VERB + DET + X → X is likely NOUN/object)

**We pre-labeled everything!**

### 2. **Transitivity Classification** (Lines 373-376)
```python
self.transitive_verbs = {'sees', 'has', 'wants', 'eats', ...}
self.intransitive_verbs = {'runs', 'sleeps', 'flies', ...}
```
**Problem**: This should be LEARNED from:
- "the dog runs" → runs is intransitive (no object follows)
- "the dog eats the food" → eats is transitive (object follows)

### 3. **Semantic Categories** (Lines 378-382)
```python
self.edible = {'food', 'milk', 'apple', 'water'}
self.drinkable = {'milk', 'water'}
self.animate = {'dog', 'cat', ...}
```
**Problem**: This is world knowledge that should be grounded through experience, not hardcoded.

### 4. **Grammar State Machine** (Lines 620-632)
```python
transitions = {
    'START': [('DET', 'AFTER_DET'), ...],
    'AFTER_DET': [('ADJ', 'AFTER_ADJ'), ('NOUN', 'AFTER_SUBJ_NOUN')],
    ...
}
```
**Problem**: This is a complete English grammar! The system should INDUCE this from examples.

### 5. **Syntactic Role Assignment** (Lines 404-411)
```python
if position == 0:  # First noun/pronoun is likely subject
    self.brain.project(pos, 'SUBJ', learn=learn)
else:  # Later nouns are likely objects
    self.brain.project(pos, 'OBJ', learn=learn)
```
**Problem**: We're using position heuristics. Real learning would discover that:
- Before verb = subject
- After verb = object
- This varies by language (SOV, VSO, etc.)

## What the Brain Actually Learns

Looking at the actual Hebbian learning:

```python
# Line 154: Hebbian weight update
W[winners.unsqueeze(1), src_active.unsqueeze(0)] += self.beta
```

The brain learns:
1. **Word → POS associations**: When we activate "dog" and project to NOUN, it strengthens dog→NOUN connections
2. **Word sequences**: LEX→LEX connections for bigrams
3. **Word → Role associations**: dog→SUBJ when dog appears first

**But these are SUPERVISED by our hardcoded labels!**

## What True Assembly Calculus Learning Looks Like

From the original papers and `parser.py`:

### The Parser Approach (parser.py)
```python
# Areas are activated based on WHAT FIRES, not labels
# The readout rules determine meaning from activation patterns
```

Key insight: In true AC, we don't tell the system "dog is a NOUN". Instead:
1. "dog" fires in LEX
2. It projects to some area (call it A)
3. Over many examples, A develops structure where similar words cluster
4. We later INTERPRET A as "the noun area" based on what clusters there

### What We Should Do Instead

1. **No POS labels during training**
   - Just expose to sentences
   - Let the brain self-organize

2. **Emergent categories**
   - Words that appear in similar contexts should develop similar representations
   - "dog", "cat", "bird" should cluster because they all follow "the" and precede verbs

3. **Emergent grammar**
   - Word order should be learned from sequence statistics
   - Not from a hardcoded state machine

4. **Emergent semantics**
   - "eats" should become associated with "food", "milk" through co-occurrence
   - Not from a hardcoded `edible` set

## The Real Assembly Calculus Approach

From `parser.py` and the papers:

```python
class ParserBrain(Brain):
    def __init__(self):
        # Areas exist, but their MEANING emerges from learning
        self.add_area("LEX", n, k, beta)
        self.add_area("SUBJ", n, k, beta)
        self.add_area("VERB", n, k, beta)
        self.add_area("OBJ", n, k, beta)
        
        # Fibers connect areas (learnable)
        self.add_fiber("LEX", "SUBJ")
        self.add_fiber("LEX", "VERB")
        self.add_fiber("LEX", "OBJ")
        
    def parse(self, sentence):
        # Control which areas are active at each step
        # But don't tell the system WHAT category each word is
        for word in sentence:
            self.activate_word(word)
            # Let the brain figure out where to project
```

The key difference:
- **Our approach**: "dog" → label it NOUN → project to NOUN area
- **AC approach**: "dog" → project to all areas → competition determines where it lands

## What We Need to Fix

### 1. Remove POS Labels
Don't use `self.pos_lexicon`. Let categories emerge.

### 2. Learn Transitivity
Track: "after this verb, did an object appear?" Learn this statistically.

### 3. Learn Semantics from Co-occurrence
"eats" + "food" co-occur → strengthen association. No hardcoded sets.

### 4. Emergent Grammar
Use the brain's LEX→LEX connections to learn sequences. The state machine should be IMPLICIT in the weights, not explicit in code.

### 5. Competition-Based Category Assignment
When a word activates, let it project to ALL category areas simultaneously. The one with strongest response "wins" and gets strengthened.

## The Honest Truth

What we built is essentially:
- A **neural bigram model** (LEX→LEX)
- With **supervised POS tagging** (hardcoded labels)
- And a **rule-based generator** (state machine)

This is NOT Assembly Calculus. It's a hybrid that uses AC-style neurons but bypasses the learning.

## Next Steps for Real AC Learning

1. **Unsupervised category learning**: Expose to sentences, see if noun/verb clusters emerge
2. **Distributional semantics**: Words in similar contexts should have similar representations
3. **Grammar induction**: Can the system learn SVO order from examples alone?
4. **Minimal supervision**: Maybe just sentence boundaries, nothing else

The question is: **Can pure Hebbian learning with winner-take-all competition actually induce grammatical categories?**

This is the real scientific question we should be testing.

---

## Update: Experimental Results

### Experiment: True Unsupervised Learning

We ran `true_assembly_learner.py` with:
- NO POS labels
- NO grammar rules
- Just sentences → Hebbian learning → winner-take-all

### Results: FAILURE

1. **All category areas learned the same thing**
   - CAT0, CAT1, CAT2, CAT3, CAT4 all have identical word rankings
   - No differentiation emerged

2. **Word similarity is uniformly 1.0**
   - "dog" vs "cat" = 1.0 (correct, same POS)
   - "dog" vs "runs" = 1.0 (WRONG, different POS)
   - No separation between categories

3. **Sequence prediction partially works**
   - "the" → nouns (4/5 correct)
   - "he" → verbs (1/5 correct, failed)

### Why It Failed

The fundamental issue: **All areas receive the same input and have no reason to differentiate.**

In the original parser.py, differentiation comes from:
1. **INHIBIT/DISINHIBIT rules** - Word-specific control of which fibers are active
2. **Area inhibition** - Only certain areas are active at each step
3. **Fixed assemblies** - Some patterns are locked and don't change

**But these rules are HARDCODED per word category!**

```python
def generic_noun(index):
    return {
        "PRE_RULES": [
            FiberRule(DISINHIBIT, LEX, SUBJ, 0),  # Nouns can be subjects
            FiberRule(DISINHIBIT, LEX, OBJ, 0),   # Nouns can be objects
        ],
        ...
    }
```

This is NOT learning - it's encoding linguistic knowledge into the system.

### The Hard Question

**Can Assembly Calculus learn grammar WITHOUT pre-defined rules?**

The papers don't address this. They assume:
1. We know which words are nouns/verbs
2. We know the grammar rules
3. AC just implements the parsing

The LEARNING aspect of AC is about:
- Forming stable assemblies
- Associating assemblies across areas
- Pattern completion

NOT about:
- Inducing grammatical categories
- Learning word classes from distribution
- Discovering syntax from examples

### What Would True Unsupervised Learning Need?

1. **Competitive inhibition between areas**
   - If CAT0 fires strongly, CAT1-4 should be suppressed
   - This forces differentiation

2. **Temporal structure**
   - Position in sentence matters
   - First content word → SUBJ, after verb → OBJ

3. **Distributional learning**
   - Words in similar contexts cluster
   - "dog", "cat", "bird" all follow "the" → same category

4. **Sparse coding**
   - Not all areas fire for all words
   - Selective activation based on learned patterns

### Conclusion

**Assembly Calculus as described in the papers is NOT an unsupervised grammar learner.**

It is a framework for:
- Representing linguistic structures with neural assemblies
- Parsing sentences using pre-defined rules
- Binding words to syntactic roles

The "learning" is Hebbian association, not category induction.

To do true unsupervised language learning, we would need to add:
- Competitive learning between category areas
- Temporal/positional encoding
- Distributional similarity metrics
- Possibly attention-like mechanisms

This is a research direction, not a solved problem.

---

## Experiment 2: Competitive Inhibition

Added inter-area competition: only the "winning" category area learns strongly.

### Results: Still Failed

- CAT0 dominates (10x scores of others) but still has mixed POS
- Word similarity still 1.0 for everything
- No separation between noun/verb/etc.

### The Fundamental Issue

The problem is that **frequency dominates**. "the" appears in every sentence, so it has the highest score everywhere. "dog" appears often, so it's next. The system learns **word frequency**, not **word category**.

To learn categories, we would need:
1. **Positional encoding** - "the" always comes first, "runs" comes after noun
2. **Context-dependent activation** - Same word activates differently based on neighbors
3. **Negative examples** - "the runs" is ungrammatical, should be penalized

### What the Original Assembly Calculus Actually Does

Looking at `parser.py`, the system:
1. Has **pre-defined word categories** (LEXEME_DICT)
2. Has **pre-defined rules** per category (generic_noun, generic_verb, etc.)
3. Uses **control signals** to enable/disable fibers at each step

The "learning" is:
- Forming assemblies for each word
- Associating words with syntactic roles
- Pattern completion for parsing

NOT:
- Discovering that "dog" is a noun
- Learning that nouns follow determiners
- Inducing grammar from examples

### Honest Conclusion

**Assembly Calculus is a representational framework, not a learning algorithm.**

It provides a neural substrate for:
- Representing words as assemblies
- Binding words to roles
- Parsing sentences with pre-defined grammar

But it does NOT provide:
- Unsupervised category learning
- Grammar induction
- Distributional semantics

To use AC for language, you must **already know** the grammar and categories. AC just implements them neurally.

### What Would Be Needed for True Learning

1. **Temporal difference learning** - Learn from prediction errors
2. **Contrastive learning** - Distinguish grammatical from ungrammatical
3. **Attention mechanisms** - Context-dependent representations
4. **Hierarchical structure** - Compositionality

These are features of modern deep learning, not classical Hebbian learning.

### The Research Question

Can we extend Assembly Calculus with:
- Competitive learning (like SOM or k-means)
- Temporal structure (like RNNs/Transformers)
- Error-driven learning (like backprop)

While maintaining biological plausibility?

This is an open research question.

---

## Update: NEMO Paper Implementation (July 2025)

After reading Mitropolsky & Papadimitriou's 2025 paper, we implemented their approach.

### Key Insights from the Paper

1. **Grounded Learning**: Words are presented with sensory context (Visual for nouns, Motor for verbs)
2. **Separate Lex Areas**: Lex1 (strong connection to Visual), Lex2 (strong connection to Motor)
3. **Stability-Based Classification**: Nouns form stable assemblies in Lex1, wobbly in Lex2
4. **No POS Labels**: Categories EMERGE from differential grounding

### Results

With 200 grounded sentences:
- **Noun accuracy: 93.3%**
- **Verb accuracy: 69.2%**
- **Overall: 82.1%**

The system learned to distinguish nouns from verbs purely from:
- Nouns co-occurring with Visual activation
- Verbs co-occurring with Motor activation

### What This Teaches Us

1. **Pure Hebbian learning CAN induce categories** - but only with the right architecture
2. **Grounding is essential** - you can't learn categories from text alone
3. **Differential connectivity matters** - Lex1↔Visual vs Lex2↔Motor
4. **Stability is the signal** - not just activation strength

### What's Still Missing

1. Semantic pathway (Phon → Lex → Visual) doesn't work well yet
2. Word order learning (Phase 2 of paper)
3. Sentence generation

### Conclusion

Assembly Calculus CAN learn language, but requires:
1. Biologically-inspired architecture (separate areas with differential connectivity)
2. Grounded input (sensory context during learning)
3. Proper firing dynamics (recurrence, stability measurement)

This is fundamentally different from our earlier attempts which tried to learn from text alone.

