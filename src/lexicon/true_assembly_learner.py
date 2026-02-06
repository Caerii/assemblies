"""
TRUE Assembly Calculus Language Learner

NO HARDCODED KNOWLEDGE:
- No POS labels
- No grammar rules
- No semantic categories

The system must LEARN everything from:
1. Sentence exposure (sequences of words)
2. Hebbian learning (what fires together wires together)
3. Competition (winner-take-all in each area)

Key question: Can grammatical categories EMERGE from this?
"""

import torch
import numpy as np
import time
from typing import Dict, List, Tuple
from collections import defaultdict

# Check CUDA
assert torch.cuda.is_available(), "CUDA required"
DEVICE = torch.device('cuda')
print(f"Using GPU: {torch.cuda.get_device_name()}")


class TrueAssemblyBrain:
    """
    A brain that learns ONLY through Hebbian plasticity.
    
    No labels. No rules. Just neurons and connections.
    
    Architecture:
    - LEX: Where words are represented
    - CAT1, CAT2, CAT3: Category areas (will they become POS?)
    - SEQ: Sequence area (for word order)
    - ROLE1, ROLE2: Role areas (will they become SUBJ/OBJ?)
    
    All areas connect to all areas. Learning determines what sticks.
    """
    
    def __init__(self, n: int = 2000, k: int = 50, beta: float = 0.1, 
                 n_categories: int = 5, verbose: bool = True):
        self.n = n
        self.k = k
        self.beta = beta
        self.verbose = verbose
        
        # Areas - generic names, meaning will EMERGE
        self.areas = ['LEX'] + [f'CAT{i}' for i in range(n_categories)] + ['SEQ']
        
        # ALL areas connect to ALL areas (including self)
        # Learning will determine which connections are meaningful
        self.W: Dict[Tuple[str, str], torch.Tensor] = {}
        for src in self.areas:
            for dst in self.areas:
                self.W[(src, dst)] = torch.randn(n, n, device=DEVICE) * 0.01
        
        # Current activations
        self.activations: Dict[str, torch.Tensor] = {}
        
        # Word representations - will be LEARNED, not assigned
        self.word_to_indices: Dict[str, torch.Tensor] = {}
        
        # Statistics for analysis (not used in learning!)
        self.word_contexts: Dict[str, List[List[str]]] = defaultdict(list)
        
        if verbose:
            n_connections = len(self.W)
            mem_mb = n_connections * n * n * 4 / 1e6
            print(f"TrueAssemblyBrain: n={n}, k={k}, beta={beta}")
            print(f"  Areas: {self.areas}")
            print(f"  Connections: {n_connections} ({mem_mb:.1f} MB)")
    
    def get_word_assembly(self, word: str) -> torch.Tensor:
        """Get or create assembly for a word"""
        if word not in self.word_to_indices:
            # Random initial assembly
            self.word_to_indices[word] = torch.randperm(self.n, device=DEVICE)[:self.k]
        return self.word_to_indices[word]
    
    def activate(self, area: str, indices: torch.Tensor):
        """Activate specific neurons in an area"""
        act = torch.zeros(self.n, device=DEVICE)
        act[indices] = 1.0
        self.activations[area] = act
    
    def project_all(self, src_area: str, learn: bool = True) -> Dict[str, torch.Tensor]:
        """
        Project from src to ALL other areas simultaneously.
        Each area competes internally (winner-take-all).
        
        NEW: Areas also compete with each other!
        Only the STRONGEST responding area gets to learn.
        """
        src_act = self.activations.get(src_area)
        if src_act is None:
            return {}
        
        results = {}
        src_active = (src_act > 0).nonzero(as_tuple=True)[0]
        
        # First pass: compute response strength for each area
        area_responses = {}
        area_inputs = {}
        
        for dst_area in self.areas:
            W = self.W[(src_area, dst_area)]
            input_to_dst = W @ src_act
            
            # Add current activation if exists (recurrence)
            if dst_area in self.activations:
                input_to_dst += self.activations[dst_area] * 0.5
            
            area_inputs[dst_area] = input_to_dst
            # Response = sum of top-k activations
            top_vals, _ = torch.topk(input_to_dst, self.k)
            area_responses[dst_area] = top_vals.sum().item()
        
        # Find winning area (excluding LEX and SEQ)
        cat_areas = [a for a in self.areas if a.startswith('CAT')]
        if cat_areas:
            cat_responses = {a: area_responses[a] for a in cat_areas}
            winning_cat = max(cat_responses, key=cat_responses.get)
        else:
            winning_cat = None
        
        # Second pass: update activations, but only winning CAT learns
        for dst_area in self.areas:
            input_to_dst = area_inputs[dst_area]
            
            # Winner-take-all within area
            _, winners = torch.topk(input_to_dst, self.k)
            
            # Update activation
            new_act = torch.zeros(self.n, device=DEVICE)
            new_act[winners] = 1.0
            self.activations[dst_area] = new_act
            
            # Hebbian learning - but with COMPETITION
            if learn and len(src_active) > 0:
                W = self.W[(src_area, dst_area)]
                
                if dst_area.startswith('CAT'):
                    # Only winning category learns strongly
                    if dst_area == winning_cat:
                        W[winners.unsqueeze(1), src_active.unsqueeze(0)] += self.beta
                    else:
                        # Losing categories learn weakly (or not at all)
                        W[winners.unsqueeze(1), src_active.unsqueeze(0)] += self.beta * 0.1
                else:
                    # Non-category areas (LEX, SEQ) always learn
                    W[winners.unsqueeze(1), src_active.unsqueeze(0)] += self.beta
            
            results[dst_area] = winners
        
        return results
    
    def clear(self):
        """Clear all activations"""
        self.activations.clear()
    
    def get_category_response(self, word: str) -> Dict[str, float]:
        """
        Measure how strongly each category area responds to a word.
        This is for ANALYSIS only, not used in learning.
        """
        self.clear()
        indices = self.get_word_assembly(word)
        self.activate('LEX', indices)
        
        responses = {}
        for area in self.areas:
            if area == 'LEX':
                continue
            W = self.W[('LEX', area)]
            # Sum of weights from word's neurons to this area
            response = W[:, indices].sum().item()
            responses[area] = response
        
        return responses


class TrueLanguageLearner:
    """
    Language learner with NO hardcoded linguistic knowledge.
    
    Training:
    1. See sentence as sequence of words
    2. Activate each word in LEX
    3. Project to all areas
    4. Hebbian learning strengthens co-activations
    
    That's it. No labels, no rules.
    """
    
    def __init__(self, n: int = 2000, k: int = 50, n_categories: int = 5, verbose: bool = True):
        self.brain = TrueAssemblyBrain(n=n, k=k, n_categories=n_categories, verbose=verbose)
        self.verbose = verbose
        
        # Training corpus - NO LABELS, just sentences
        self.corpus = [
            # Simple patterns
            "the dog runs",
            "the cat sleeps", 
            "the bird flies",
            "a dog runs",
            "a cat sleeps",
            "a bird flies",
            
            # More subjects
            "the man walks",
            "the woman runs",
            "the child plays",
            "a man walks",
            "a woman runs",
            
            # With objects
            "the dog eats food",
            "the cat drinks milk",
            "the man sees the dog",
            "the woman has a cat",
            "the child wants the ball",
            
            # Adjectives
            "the big dog runs",
            "the small cat sleeps",
            "a big dog runs",
            "the fast bird flies",
            
            # Pronouns
            "he runs",
            "she sleeps",
            "it flies",
            "he sees the dog",
            "she has a cat",
            
            # Variations
            "dogs run",
            "cats sleep",
            "birds fly",
            "the dogs run",
            "the cats sleep",
        ]
        
        # For analysis only - ground truth POS (NOT used in learning!)
        self._ground_truth_pos = {
            'the': 'DET', 'a': 'DET',
            'dog': 'NOUN', 'cat': 'NOUN', 'bird': 'NOUN', 'man': 'NOUN', 
            'woman': 'NOUN', 'child': 'NOUN', 'food': 'NOUN', 'milk': 'NOUN',
            'ball': 'NOUN', 'dogs': 'NOUN', 'cats': 'NOUN', 'birds': 'NOUN',
            'runs': 'VERB', 'sleeps': 'VERB', 'flies': 'VERB', 'walks': 'VERB',
            'plays': 'VERB', 'eats': 'VERB', 'drinks': 'VERB', 'sees': 'VERB',
            'has': 'VERB', 'wants': 'VERB', 'run': 'VERB', 'sleep': 'VERB', 'fly': 'VERB',
            'big': 'ADJ', 'small': 'ADJ', 'fast': 'ADJ',
            'he': 'PRON', 'she': 'PRON', 'it': 'PRON',
        }
        
        self.sentences_seen = 0
    
    def process_sentence(self, sentence: str, learn: bool = True):
        """
        Process a sentence with NO labels.
        Just activate words in sequence and let Hebbian learning do its thing.
        """
        words = sentence.lower().strip().split()
        self.brain.clear()
        
        prev_word = None
        
        for i, word in enumerate(words):
            # Get word assembly
            indices = self.brain.get_word_assembly(word)
            
            # Activate in LEX
            self.brain.activate('LEX', indices)
            
            # Project to ALL areas - let competition decide
            self.brain.project_all('LEX', learn=learn)
            
            # Track context for analysis
            context = words[max(0, i-2):i] + words[i+1:min(len(words), i+3)]
            self.brain.word_contexts[word].append(context)
            
            # Sequence learning: previous word influences current
            if prev_word is not None and learn:
                prev_indices = self.brain.word_to_indices.get(prev_word)
                curr_indices = indices
                
                if prev_indices is not None:
                    # Strengthen sequence connection in LEX
                    W = self.brain.W[('LEX', 'LEX')]
                    W[curr_indices.unsqueeze(1), prev_indices.unsqueeze(0)] += self.brain.beta
                    
                    # Also in SEQ area
                    W_seq = self.brain.W[('LEX', 'SEQ')]
                    # The SEQ area should learn position patterns
            
            prev_word = word
        
        if learn:
            self.sentences_seen += 1
    
    def train(self, n_epochs: int = 10, verbose: bool = True):
        """Train on corpus"""
        if verbose:
            print(f"\nTraining for {n_epochs} epochs...")
        
        start = time.perf_counter()
        
        for epoch in range(n_epochs):
            sentences = self.corpus.copy()
            np.random.shuffle(sentences)
            
            for sentence in sentences:
                self.process_sentence(sentence, learn=True)
            
            if verbose and (epoch + 1) % max(1, n_epochs // 5) == 0:
                print(f"  Epoch {epoch + 1}/{n_epochs}")
        
        elapsed = time.perf_counter() - start
        if verbose:
            print(f"Training complete in {elapsed:.2f}s")
            print(f"  Sentences seen: {self.sentences_seen}")
    
    def analyze_categories(self, verbose: bool = True) -> Dict[str, Dict[str, List[str]]]:
        """
        Analyze what the category areas have learned.
        
        For each category area, find which words respond most strongly.
        Then check if these correspond to linguistic categories.
        """
        if verbose:
            print("\n" + "=" * 60)
            print("CATEGORY ANALYSIS")
            print("=" * 60)
        
        # Get response of each word to each category
        word_responses: Dict[str, Dict[str, float]] = {}
        
        for word in self.brain.word_to_indices.keys():
            word_responses[word] = self.brain.get_category_response(word)
        
        # For each category, find top words
        category_words: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        
        for word, responses in word_responses.items():
            for cat, score in responses.items():
                category_words[cat].append((word, score))
        
        # Sort by score
        for cat in category_words:
            category_words[cat].sort(key=lambda x: -x[1])
        
        # Analyze: do the clusters correspond to POS?
        results = {}
        
        for cat in sorted(category_words.keys()):
            if cat == 'SEQ':
                continue
            
            top_words = category_words[cat][:10]
            
            if verbose:
                print(f"\n{cat}:")
                for word, score in top_words:
                    gt_pos = self._ground_truth_pos.get(word, '?')
                    print(f"  {word:12} (score: {score:6.1f}) - GT: {gt_pos}")
            
            # Check POS distribution in this category
            pos_counts: Dict[str, int] = defaultdict(int)
            for word, _ in top_words:
                pos = self._ground_truth_pos.get(word, 'UNK')
                pos_counts[pos] += 1
            
            dominant_pos = max(pos_counts.items(), key=lambda x: x[1])[0] if pos_counts else 'UNK'
            purity = pos_counts[dominant_pos] / len(top_words) if top_words else 0
            
            results[cat] = {
                'top_words': [w for w, _ in top_words],
                'dominant_pos': dominant_pos,
                'purity': purity,
                'pos_distribution': dict(pos_counts)
            }
            
            if verbose:
                print(f"  → Dominant POS: {dominant_pos} (purity: {purity:.0%})")
        
        return results
    
    def analyze_word_similarity(self, verbose: bool = True):
        """
        Check if words with similar distributions have similar representations.
        
        Key test: Do "dog" and "cat" cluster together? (both are nouns)
        Do "runs" and "sleeps" cluster together? (both are verbs)
        """
        if verbose:
            print("\n" + "=" * 60)
            print("WORD SIMILARITY ANALYSIS")
            print("=" * 60)
        
        # Get representation for each word (concatenate responses to all categories)
        word_vectors: Dict[str, np.ndarray] = {}
        
        for word in self.brain.word_to_indices.keys():
            responses = self.brain.get_category_response(word)
            vec = np.array([responses.get(f'CAT{i}', 0) for i in range(5)])
            word_vectors[word] = vec
        
        # Compute similarities
        def cosine_sim(v1, v2):
            norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if norm1 == 0 or norm2 == 0:
                return 0
            return np.dot(v1, v2) / (norm1 * norm2)
        
        # Test pairs
        test_pairs = [
            # Should be similar (same POS)
            ('dog', 'cat', 'NOUN-NOUN'),
            ('runs', 'sleeps', 'VERB-VERB'),
            ('big', 'small', 'ADJ-ADJ'),
            ('the', 'a', 'DET-DET'),
            
            # Should be different (different POS)
            ('dog', 'runs', 'NOUN-VERB'),
            ('the', 'dog', 'DET-NOUN'),
            ('big', 'runs', 'ADJ-VERB'),
        ]
        
        if verbose:
            print(f"\n{'Word1':>10} {'Word2':>10} {'Similarity':>12} {'Expected':>12}")
            print("-" * 50)
        
        same_pos_sims = []
        diff_pos_sims = []
        
        for w1, w2, label in test_pairs:
            if w1 in word_vectors and w2 in word_vectors:
                sim = cosine_sim(word_vectors[w1], word_vectors[w2])
                
                if 'same' in label.lower() or label.count('-') == 1 and label.split('-')[0] == label.split('-')[1]:
                    same_pos_sims.append(sim)
                    expected = 'HIGH'
                else:
                    diff_pos_sims.append(sim)
                    expected = 'LOW'
                
                if verbose:
                    print(f"{w1:>10} {w2:>10} {sim:>12.3f} {expected:>12}")
        
        if verbose:
            avg_same = np.mean(same_pos_sims) if same_pos_sims else 0
            avg_diff = np.mean(diff_pos_sims) if diff_pos_sims else 0
            print(f"\nAverage same-POS similarity: {avg_same:.3f}")
            print(f"Average diff-POS similarity: {avg_diff:.3f}")
            print(f"Separation: {avg_same - avg_diff:.3f}")
            
            if avg_same > avg_diff:
                print("✓ Words with same POS are MORE similar (good!)")
            else:
                print("✗ No clear POS clustering (need more training?)")
    
    def test_prediction(self, verbose: bool = True):
        """
        Test if the system can predict what comes next.
        Uses LEX→LEX connections learned from sequences.
        """
        if verbose:
            print("\n" + "=" * 60)
            print("SEQUENCE PREDICTION TEST")
            print("=" * 60)
        
        test_contexts = [
            (["the"], "Expect: noun (dog, cat, ...)"),
            (["the", "dog"], "Expect: verb (runs, eats, ...)"),
            (["the", "big"], "Expect: noun (dog, cat, ...)"),
            (["he"], "Expect: verb (runs, sees, ...)"),
        ]
        
        for context, description in test_contexts:
            self.brain.clear()
            
            # Activate context words
            for word in context:
                if word in self.brain.word_to_indices:
                    indices = self.brain.word_to_indices[word]
                    self.brain.activate('LEX', indices)
                    self.brain.project_all('LEX', learn=False)
            
            # Get LEX activation
            lex_act = self.brain.activations.get('LEX')
            if lex_act is None:
                continue
            
            # Score each word by overlap
            active = set((lex_act > 0).nonzero(as_tuple=True)[0].cpu().numpy())
            
            scores = []
            for word, indices in self.brain.word_to_indices.items():
                if word in context:
                    continue
                word_neurons = set(indices.cpu().numpy())
                overlap = len(active & word_neurons) / self.brain.k
                scores.append((word, overlap))
            
            scores.sort(key=lambda x: -x[1])
            top5 = scores[:5]
            
            if verbose:
                context_str = " ".join(context)
                pred_str = ", ".join([f"{w}:{s:.2f}" for w, s in top5])
                print(f"\n  Context: '{context_str}'")
                print(f"  {description}")
                print(f"  Predictions: [{pred_str}]")
                
                # Check if predictions match expected POS
                expected_pos = 'NOUN' if 'noun' in description.lower() else 'VERB'
                correct = sum(1 for w, _ in top5 
                             if self._ground_truth_pos.get(w) == expected_pos)
                print(f"  Correct POS in top-5: {correct}/5")


def main():
    print("=" * 70)
    print("TRUE ASSEMBLY CALCULUS LANGUAGE LEARNER")
    print("=" * 70)
    print("\nNO HARDCODED KNOWLEDGE - Everything must be LEARNED")
    
    # Create learner
    learner = TrueLanguageLearner(n=2000, k=50, n_categories=5, verbose=True)
    
    # Train
    learner.train(n_epochs=50, verbose=True)
    
    # Analyze what emerged
    learner.analyze_categories(verbose=True)
    learner.analyze_word_similarity(verbose=True)
    learner.test_prediction(verbose=True)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Words learned: {len(learner.brain.word_to_indices)}")
    print(f"  Sentences seen: {learner.sentences_seen}")
    
    mem = torch.cuda.memory_allocated() / 1e6
    print(f"  GPU memory: {mem:.1f} MB")
    
    print("\n  KEY QUESTION: Did grammatical categories EMERGE?")
    print("  Look at the category analysis above to see if CAT areas")
    print("  correspond to POS (NOUN, VERB, etc.)")


if __name__ == "__main__":
    main()

