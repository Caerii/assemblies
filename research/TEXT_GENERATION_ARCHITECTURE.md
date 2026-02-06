# Text Generation with Assembly Calculus

## Overview

To generate text using Assembly Calculus, we need to represent words/tokens as neural assemblies and learn sequential associations between them.

## Core Components Needed

### 1. **Lexicon Area** - Word/Token Representations
Each word or token gets its own distinct assembly:
- Area with n neurons, k = √n active
- Each word → unique assembly of k neurons
- Vocabulary size limited by capacity: ~n/k distinct assemblies

```
Example for vocab of 10,000 words:
- n = 1,000,000 neurons
- k = 1,000 active per word
- Capacity = ~1,000 distinct assemblies (need larger n!)

Better: n = 100,000,000, k = 10,000
- Capacity = ~10,000 distinct words ✓
```

### 2. **Sequence Memory** - Temporal Associations
Learn that word A is followed by word B:
- Use cross-area projection: Lexicon → Sequence → Lexicon
- Or use recurrent connections within Lexicon area
- Hebbian learning strengthens A→B transitions

```
Training "the cat sat":
1. Activate "the" assembly
2. Project to sequence area
3. Activate "cat" assembly  
4. Associate sequence→cat (Hebbian)
5. Repeat for "cat"→"sat"
```

### 3. **Context Area** - Longer-range Dependencies
Track context beyond immediate predecessor:
- Separate area that accumulates context
- Multiple words project into context
- Context influences next word prediction

### 4. **Pattern Completion** - Generation
Given partial input, complete to full word:
- Activate partial assembly (e.g., from context)
- Run recurrent dynamics
- Assembly completes to nearest learned word

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      CONTEXT AREA                            │
│  (accumulates meaning over multiple words)                   │
│  n = 10M, k = 3162                                          │
└─────────────────────────────────────────────────────────────┘
           ↑                              ↓
           │ project                      │ influence
           │                              │
┌─────────────────────────────────────────────────────────────┐
│                      LEXICON AREA                            │
│  (each word = one assembly)                                  │
│  n = 100M, k = 10000                                        │
│                                                              │
│  "the" → [neurons 1,5,9,...]                                │
│  "cat" → [neurons 2,7,12,...]                               │
│  "sat" → [neurons 3,8,15,...]                               │
└─────────────────────────────────────────────────────────────┘
           ↑                              ↓
           │ input                        │ output
           │                              │
┌─────────────────────────────────────────────────────────────┐
│                    SEQUENCE AREA                             │
│  (learns temporal transitions)                               │
│  n = 10M, k = 3162                                          │
│                                                              │
│  Learns: "the"→"cat", "cat"→"sat"                          │
└─────────────────────────────────────────────────────────────┘
```

## What We Already Have

### ✅ Working Components
1. **Projection** - Create assemblies from stimuli
2. **Association** - Link assemblies across areas (with fix_assembly)
3. **Pattern Completion** - Recover full assembly from partial (with k=√n)
4. **GPU Acceleration** - PyTorch (40x) and custom CUDA (6x at billion-scale)
5. **Billion-scale Simulation** - 1B neurons, 3.8GB memory, 45 steps/sec

### ❌ Missing Components
1. **Token Embedding** - Map text tokens to stimulus patterns
2. **Sequence Learning** - Train on text corpus
3. **Sampling/Decoding** - Convert assemblies back to tokens
4. **Temperature/Top-p** - Control generation randomness

## Implementation Plan

### Phase 1: Token ↔ Assembly Mapping
```python
class AssemblyTokenizer:
    def __init__(self, vocab_size, n_neurons, k_active):
        self.vocab_size = vocab_size
        self.n = n_neurons
        self.k = k_active
        # Each token gets a fixed random stimulus pattern
        self.token_to_stimulus = {}
        
    def encode(self, token) -> np.ndarray:
        """Convert token to stimulus pattern (k active neurons)"""
        if token not in self.token_to_stimulus:
            # Generate deterministic pattern from token hash
            rng = np.random.default_rng(hash(token) % 2**32)
            self.token_to_stimulus[token] = rng.choice(self.n, self.k, replace=False)
        return self.token_to_stimulus[token]
    
    def decode(self, assembly) -> str:
        """Find closest token to given assembly"""
        best_token = None
        best_overlap = 0
        for token, stim in self.token_to_stimulus.items():
            overlap = len(set(assembly) & set(stim))
            if overlap > best_overlap:
                best_overlap = overlap
                best_token = token
        return best_token
```

### Phase 2: Sequence Learning
```python
class SequenceLearner:
    def __init__(self, brain, tokenizer):
        self.brain = brain
        self.tokenizer = tokenizer
        
    def train_sequence(self, tokens):
        """Train on a sequence of tokens"""
        prev_assembly = None
        
        for token in tokens:
            # Get stimulus for this token
            stimulus = self.tokenizer.encode(token)
            
            # Project stimulus to create/activate assembly
            self.brain.project_stimulus(stimulus)
            current_assembly = self.brain.get_winners()
            
            if prev_assembly is not None:
                # Associate previous → current (Hebbian)
                self.brain.associate(prev_assembly, current_assembly)
            
            prev_assembly = current_assembly
```

### Phase 3: Text Generation
```python
class TextGenerator:
    def __init__(self, brain, tokenizer, sequence_learner):
        self.brain = brain
        self.tokenizer = tokenizer
        
    def generate(self, prompt_tokens, max_length=50):
        """Generate text continuation"""
        generated = list(prompt_tokens)
        
        # Initialize with prompt
        for token in prompt_tokens:
            stimulus = self.tokenizer.encode(token)
            self.brain.project_stimulus(stimulus)
        
        # Generate new tokens
        for _ in range(max_length):
            # Run pattern completion to get next assembly
            self.brain.step()  # Recurrent dynamics
            
            # Decode assembly to token
            assembly = self.brain.get_winners()
            next_token = self.tokenizer.decode(assembly)
            
            if next_token == "<EOS>":
                break
                
            generated.append(next_token)
            
            # Project new token for next iteration
            stimulus = self.tokenizer.encode(next_token)
            self.brain.project_stimulus(stimulus)
        
        return generated
```

## Memory Requirements

| Component | Neurons | k | Memory (Sparse) |
|-----------|---------|---|-----------------|
| Lexicon (10K vocab) | 100M | 10K | 400 MB |
| Sequence | 10M | 3.2K | 40 MB |
| Context | 10M | 3.2K | 40 MB |
| **Total** | **120M** | - | **~500 MB** |

This fits easily on a single GPU!

## Key Challenges

1. **Vocabulary Size**: Need n >> vocab_size × k for distinct assemblies
2. **Long-range Dependencies**: Context area may not capture enough
3. **Training Speed**: Hebbian learning is slow compared to backprop
4. **Quality**: Assembly-based generation likely won't match transformers

## Comparison to Transformers

| Aspect | Assembly Calculus | Transformers |
|--------|-------------------|--------------|
| Memory | O(vocab × k²) | O(vocab × d²) |
| Attention | Implicit via associations | Explicit QKV |
| Training | Hebbian (local) | Backprop (global) |
| Parallelism | High (sparse ops) | High (matrix ops) |
| Interpretability | High (assemblies = concepts) | Low (distributed) |
| Quality | Unknown (research) | State-of-art |

## Next Steps

1. **Implement AssemblyTokenizer** with fixed stimulus patterns
2. **Create SequenceLearner** using our existing brain.py
3. **Train on simple corpus** (e.g., children's books)
4. **Evaluate generation quality**
5. **Compare to simple n-gram baselines**

