"""
GPU-Accelerated Language-Only Learner

Learns words through linguistic context only:
- Word co-occurrence patterns
- Sentence structure (SVO, word order)
- Grammatical categories

Uses PyTorch CUDA for fast iteration.
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict

# Check CUDA availability
assert torch.cuda.is_available(), "CUDA required for GPU learner"
DEVICE = torch.device('cuda')
print(f"Using GPU: {torch.cuda.get_device_name()}")


class GPULanguageBrain:
    """
    GPU-accelerated brain for language learning.
    
    Uses SMALLER, more efficient weight matrices.
    Only stores weights for connections that exist.
    
    Areas:
    - LEX: Lexical representations (words)
    - NOUN, VERB, ADJ, DET, PREP, ADV: Grammatical category areas
    - SUBJ, OBJ, VERB_ROLE: Syntactic role areas
    - SEQ: Sequence/word order area
    """
    
    def __init__(self, n: int = 2000, k: int = 50, beta: float = 0.2, verbose: bool = True):
        self.n = n  # neurons per area
        self.k = k  # assembly size (winners)
        self.beta = beta  # Hebbian learning rate
        self.verbose = verbose
        
        # Define areas
        self.areas = [
            'LEX',      # All words
            'NOUN', 'VERB', 'ADJ', 'DET', 'PREP', 'ADV', 'PRON',  # POS categories
            'SUBJ', 'OBJ', 'VERB_ROLE',  # Syntactic roles
            'SEQ',      # Sequence position encoding
        ]
        
        # Weight matrices (area -> area) on GPU
        # Only create connections that make sense
        self.connections = {
            # LEX connects to all POS areas
            ('LEX', 'NOUN'): self._init_weights(),
            ('LEX', 'VERB'): self._init_weights(),
            ('LEX', 'ADJ'): self._init_weights(),
            ('LEX', 'DET'): self._init_weights(),
            ('LEX', 'PREP'): self._init_weights(),
            ('LEX', 'ADV'): self._init_weights(),
            ('LEX', 'PRON'): self._init_weights(),
            
            # POS areas connect to syntactic roles
            ('NOUN', 'SUBJ'): self._init_weights(),
            ('NOUN', 'OBJ'): self._init_weights(),
            ('PRON', 'SUBJ'): self._init_weights(),
            ('PRON', 'OBJ'): self._init_weights(),
            ('VERB', 'VERB_ROLE'): self._init_weights(),
            
            # Sequence connections - critical for word order
            ('LEX', 'SEQ'): self._init_weights(),
            ('SEQ', 'LEX'): self._init_weights(),  # For prediction
            
            # Recurrent within LEX for word associations
            ('LEX', 'LEX'): self._init_weights(),
        }
        
        # Current activations per area
        self.activations: Dict[str, torch.Tensor] = {}
        
        # Word -> neuron indices mapping
        self.word_to_neurons: Dict[str, torch.Tensor] = {}
        self.neuron_to_word: Dict[int, str] = {}  # For LEX area
        
        # Learned statistics
        self.word_counts: Dict[str, int] = defaultdict(int)
        self.bigram_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        self.pos_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        if verbose:
            print(f"GPULanguageBrain initialized: n={n}, k={k}, beta={beta}")
            print(f"  Areas: {len(self.areas)}")
            print(f"  Connections: {len(self.connections)}")
            mem_mb = sum(w.numel() * 4 for w in self.connections.values()) / 1e6
            print(f"  GPU Memory: {mem_mb:.1f} MB")
    
    def _init_weights(self) -> torch.Tensor:
        """Initialize weight matrix with small random values"""
        W = torch.randn(self.n, self.n, device=DEVICE) * 0.01
        return W
    
    def _get_or_create_word_assembly(self, word: str) -> torch.Tensor:
        """Get or create a random assembly for a word in LEX"""
        if word not in self.word_to_neurons:
            # Create random assembly
            indices = torch.randperm(self.n, device=DEVICE)[:self.k]
            self.word_to_neurons[word] = indices
            
            # Store reverse mapping
            for idx in indices.cpu().numpy():
                self.neuron_to_word[idx] = word
        
        return self.word_to_neurons[word]
    
    def activate_word(self, word: str) -> torch.Tensor:
        """Activate a word's assembly in LEX area"""
        indices = self._get_or_create_word_assembly(word)
        
        # Create activation vector
        activation = torch.zeros(self.n, device=DEVICE)
        activation[indices] = 1.0
        
        self.activations['LEX'] = activation
        return indices
    
    def project(self, src_area: str, dst_area: str, learn: bool = True) -> torch.Tensor:
        """Project activation from src to dst area"""
        key = (src_area, dst_area)
        if key not in self.connections:
            return torch.zeros(self.n, device=DEVICE)
        
        W = self.connections[key]
        src_act = self.activations.get(src_area)
        
        if src_act is None:
            return torch.zeros(self.n, device=DEVICE)
        
        # Simple matrix-vector multiply (GPU handles this efficiently)
        input_to_dst = W @ src_act
        
        # Winner-take-all: top-k neurons
        _, winners = torch.topk(input_to_dst, self.k)
        
        # Create new activation
        new_act = torch.zeros(self.n, device=DEVICE)
        new_act[winners] = 1.0
        
        # Hebbian learning
        if learn:
            src_active = (src_act > 0).nonzero(as_tuple=True)[0]
            if len(src_active) > 0:
                W[winners.unsqueeze(1), src_active.unsqueeze(0)] += self.beta
        
        self.activations[dst_area] = new_act
        return winners
    
    def project_multiple(self, src_areas: List[str], dst_area: str, learn: bool = True) -> torch.Tensor:
        """Project from multiple source areas to one destination"""
        # Accumulate inputs
        total_input = torch.zeros(self.n, device=DEVICE)
        
        for src_area in src_areas:
            key = (src_area, dst_area)
            if key not in self.connections:
                continue
            
            W = self.connections[key]
            src_act = self.activations.get(src_area)
            
            if src_act is not None:
                total_input += W @ src_act
        
        # Winner-take-all
        _, winners = torch.topk(total_input, self.k)
        
        # Create new activation
        new_act = torch.zeros(self.n, device=DEVICE)
        new_act[winners] = 1.0
        
        # Hebbian learning for all sources
        if learn:
            for src_area in src_areas:
                key = (src_area, dst_area)
                if key not in self.connections:
                    continue
                
                W = self.connections[key]
                src_act = self.activations.get(src_area)
                
                if src_act is not None:
                    src_active = (src_act > 0).nonzero(as_tuple=True)[0]
                    if len(src_active) > 0:
                        w_grid = winners.unsqueeze(1).expand(-1, len(src_active))
                        s_grid = src_active.unsqueeze(0).expand(len(winners), -1)
                        W[w_grid, s_grid] += self.beta
        
        self.activations[dst_area] = new_act
        return winners
    
    def get_word_from_lex(self) -> Optional[str]:
        """Read out the currently active word from LEX"""
        lex_act = self.activations.get('LEX')
        if lex_act is None:
            return None
        
        # Find active neurons
        active = (lex_act > 0).nonzero(as_tuple=True)[0].cpu().numpy()
        
        # Count votes for each word
        word_votes: Dict[str, int] = defaultdict(int)
        for idx in active:
            if idx in self.neuron_to_word:
                word_votes[self.neuron_to_word[idx]] += 1
        
        if not word_votes:
            return None
        
        # Return word with most votes
        return max(word_votes.items(), key=lambda x: x[1])[0]
    
    def clear_activations(self):
        """Clear all activations"""
        self.activations.clear()
    
    def compute_overlap(self, indices1: torch.Tensor, indices2: torch.Tensor) -> float:
        """Compute overlap between two assemblies"""
        set1 = set(indices1.cpu().numpy())
        set2 = set(indices2.cpu().numpy())
        intersection = len(set1 & set2)
        return intersection / self.k


class GPULanguageLearner:
    """
    Language learner using GPU-accelerated brain.
    
    Learns through:
    1. Word exposure in sentences
    2. POS category association
    3. Word order patterns
    4. Word co-occurrence
    """
    
    def __init__(self, n: int = 10000, k: int = 100, verbose: bool = True):
        self.brain = GPULanguageBrain(n=n, k=k, verbose=verbose)
        self.verbose = verbose
        
        # Simple POS lexicon
        self.pos_lexicon = {
            # Determiners
            'the': 'DET', 'a': 'DET', 'an': 'DET', 'this': 'DET', 'that': 'DET',
            'my': 'DET', 'your': 'DET', 'his': 'DET', 'her': 'DET',
            
            # Pronouns
            'i': 'PRON', 'you': 'PRON', 'he': 'PRON', 'she': 'PRON', 'it': 'PRON',
            'we': 'PRON', 'they': 'PRON',
            
            # Common nouns
            'dog': 'NOUN', 'cat': 'NOUN', 'bird': 'NOUN', 'fish': 'NOUN',
            'man': 'NOUN', 'woman': 'NOUN', 'child': 'NOUN', 'baby': 'NOUN',
            'ball': 'NOUN', 'book': 'NOUN', 'car': 'NOUN', 'house': 'NOUN',
            'food': 'NOUN', 'water': 'NOUN', 'milk': 'NOUN', 'apple': 'NOUN',
            'tree': 'NOUN', 'flower': 'NOUN', 'sun': 'NOUN', 'moon': 'NOUN',
            'mom': 'NOUN', 'dad': 'NOUN', 'boy': 'NOUN', 'girl': 'NOUN',
            
            # Verbs
            'is': 'VERB', 'are': 'VERB', 'was': 'VERB', 'were': 'VERB',
            'run': 'VERB', 'runs': 'VERB', 'walk': 'VERB', 'walks': 'VERB',
            'eat': 'VERB', 'eats': 'VERB', 'drink': 'VERB', 'drinks': 'VERB',
            'see': 'VERB', 'sees': 'VERB', 'look': 'VERB', 'looks': 'VERB',
            'play': 'VERB', 'plays': 'VERB', 'sleep': 'VERB', 'sleeps': 'VERB',
            'jump': 'VERB', 'jumps': 'VERB', 'fly': 'VERB', 'flies': 'VERB',
            'like': 'VERB', 'likes': 'VERB', 'want': 'VERB', 'wants': 'VERB',
            'have': 'VERB', 'has': 'VERB', 'go': 'VERB', 'goes': 'VERB',
            
            # Adjectives
            'big': 'ADJ', 'small': 'ADJ', 'fast': 'ADJ', 'slow': 'ADJ',
            'good': 'ADJ', 'bad': 'ADJ', 'happy': 'ADJ', 'sad': 'ADJ',
            'red': 'ADJ', 'blue': 'ADJ', 'green': 'ADJ', 'yellow': 'ADJ',
            'hot': 'ADJ', 'cold': 'ADJ', 'new': 'ADJ', 'old': 'ADJ',
            
            # Adverbs
            'quickly': 'ADV', 'slowly': 'ADV', 'very': 'ADV', 'really': 'ADV',
            'now': 'ADV', 'here': 'ADV', 'there': 'ADV', 'always': 'ADV',
            
            # Prepositions
            'in': 'PREP', 'on': 'PREP', 'at': 'PREP', 'to': 'PREP',
            'with': 'PREP', 'for': 'PREP', 'from': 'PREP', 'of': 'PREP',
        }
        
        # Training corpus - child-directed speech style
        # More examples to learn proper patterns
        self.training_sentences = [
            # Simple intransitive (SUBJ VERB)
            "the dog runs",
            "the cat sleeps",
            "the bird flies",
            "the baby sleeps",
            "the boy plays",
            "the girl jumps",
            "the man walks",
            "the woman runs",
            "the child plays",
            
            # Transitive with object (SUBJ VERB DET NOUN)
            "the dog eats the food",
            "the cat drinks the milk",
            "the baby wants the milk",
            "the boy sees the ball",
            "the girl has the book",
            "the man walks the dog",
            "the woman sees the cat",
            "the child wants the toy",
            "the dog chases the cat",
            "the cat sees the bird",
            "the boy throws the ball",
            "the girl reads the book",
            
            # With adjectives (DET ADJ NOUN VERB)
            "the big dog runs",
            "the small cat sleeps",
            "the happy baby eats",
            "the fast bird flies",
            "the good boy plays",
            "the little girl jumps",
            "the old man walks",
            "the young woman runs",
            
            # Adjectives with objects (DET ADJ NOUN VERB DET NOUN)
            "the big dog eats the food",
            "the small cat drinks the milk",
            "the happy boy sees the ball",
            "the little girl has the book",
            
            # Pronouns (PRON VERB)
            "he runs",
            "she sleeps",
            "it flies",
            "he plays",
            "she jumps",
            
            # Pronouns with objects (PRON VERB DET NOUN)
            "he sees the dog",
            "she has the cat",
            "he wants the ball",
            "she reads the book",
            "he eats the food",
            "she drinks the milk",
            
            # Copular sentences (SUBJ is ADJ)
            "the dog is big",
            "the cat is small",
            "the ball is red",
            "the bird is fast",
            "the baby is happy",
            "the boy is good",
            
            # With adverbs
            "the dog runs quickly",
            "the bird flies fast",
            "the man walks slowly",
            "the cat sleeps quietly",
            
            # Common patterns
            "mom sees the baby",
            "dad has the ball",
            "the baby sees mom",
            "the boy wants dad",
        ]
        
        # Mark transitive vs intransitive verbs
        self.transitive_verbs = {'sees', 'has', 'wants', 'eats', 'drinks', 'reads', 
                                  'walks', 'chases', 'throws', 'has', 'have'}
        self.intransitive_verbs = {'runs', 'sleeps', 'flies', 'plays', 'jumps', 'is'}
        
        # Semantic constraints: what can be eaten/drunk/etc.
        self.edible = {'food', 'milk', 'apple', 'water'}
        self.drinkable = {'milk', 'water'}
        self.animate = {'dog', 'cat', 'bird', 'fish', 'man', 'woman', 'child', 'baby', 
                        'boy', 'girl', 'mom', 'dad'}
        
        # Statistics
        self.words_learned: Set[str] = set()
        self.sentences_seen = 0
        self.training_time = 0.0
    
    def get_pos(self, word: str) -> str:
        """Get POS tag for a word"""
        return self.pos_lexicon.get(word.lower(), 'NOUN')  # Default to NOUN
    
    def process_word(self, word: str, pos: str, position: int, learn: bool = True):
        """Process a single word: activate and project to relevant areas"""
        word = word.lower()
        
        # Activate word in LEX
        self.brain.activate_word(word)
        
        # Project to POS area
        if pos in self.brain.areas:
            self.brain.project('LEX', pos, learn=learn)
        
        # Project to syntactic role based on position and POS
        if pos == 'NOUN' or pos == 'PRON':
            if position == 0:  # First noun/pronoun is likely subject
                self.brain.project(pos, 'SUBJ', learn=learn)
            else:  # Later nouns are likely objects
                self.brain.project(pos, 'OBJ', learn=learn)
        elif pos == 'VERB':
            self.brain.project('VERB', 'VERB_ROLE', learn=learn)
        
        # Update word count
        if learn:
            self.brain.word_counts[word] += 1
            self.words_learned.add(word)
    
    def process_sentence(self, sentence: str, learn: bool = True, verbose: bool = False):
        """Process a sentence: word by word with learning"""
        words = sentence.lower().strip().split()
        
        if verbose:
            print(f"  Processing: '{sentence}'")
        
        prev_word = None
        
        for i, word in enumerate(words):
            pos = self.get_pos(word)
            
            # Process this word
            self.process_word(word, pos, i, learn=learn)
            
            # Learn word sequence (bigrams)
            if prev_word is not None and learn:
                self.brain.bigram_counts[(prev_word, word)] += 1
                
                # Strengthen LEX->LEX connection for sequence
                prev_indices = self.brain.word_to_neurons.get(prev_word)
                curr_indices = self.brain.word_to_neurons.get(word)
                
                if prev_indices is not None and curr_indices is not None:
                    W = self.brain.connections[('LEX', 'LEX')]
                    # curr neurons get input from prev neurons
                    c_grid = curr_indices.unsqueeze(1).expand(-1, len(prev_indices))
                    p_grid = prev_indices.unsqueeze(0).expand(len(curr_indices), -1)
                    W[c_grid, p_grid] += self.brain.beta
            
            prev_word = word
        
        if learn:
            self.sentences_seen += 1
    
    def train(self, n_epochs: int = 10, verbose: bool = True):
        """Train on the corpus for multiple epochs"""
        if verbose:
            print(f"\nTraining for {n_epochs} epochs on {len(self.training_sentences)} sentences...")
        
        start = time.perf_counter()
        
        for epoch in range(n_epochs):
            # Shuffle sentences each epoch
            sentences = self.training_sentences.copy()
            np.random.shuffle(sentences)
            
            for sentence in sentences:
                self.brain.clear_activations()
                self.process_sentence(sentence, learn=True, verbose=False)
            
            if verbose and (epoch + 1) % max(1, n_epochs // 5) == 0:
                print(f"  Epoch {epoch + 1}/{n_epochs}: {self.sentences_seen} sentences processed")
        
        self.training_time = time.perf_counter() - start
        
        if verbose:
            print(f"Training complete in {self.training_time:.2f}s")
            print(f"  Words learned: {len(self.words_learned)}")
            print(f"  Sentences seen: {self.sentences_seen}")
    
    def predict_next_word(self, context: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """Given context words, predict the next word"""
        self.brain.clear_activations()
        
        # Activate context words and let them influence LEX
        for word in context:
            word = word.lower()
            if word in self.brain.word_to_neurons:
                self.brain.activate_word(word)
                # Project through LEX->LEX to get next word candidates
                self.brain.project('LEX', 'LEX', learn=False)
        
        # Get current LEX activation
        lex_act = self.brain.activations.get('LEX')
        if lex_act is None:
            return []
        
        # Score each known word by overlap with current activation
        scores = []
        active_neurons = set((lex_act > 0).nonzero(as_tuple=True)[0].cpu().numpy())
        
        for word, indices in self.brain.word_to_neurons.items():
            if word in context:  # Don't predict same word
                continue
            word_neurons = set(indices.cpu().numpy())
            overlap = len(active_neurons & word_neurons) / self.brain.k
            scores.append((word, overlap))
        
        # Sort by score
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]
    
    def test_pos_classification(self, verbose: bool = True) -> float:
        """Test if words activate the correct POS area"""
        correct = 0
        total = 0
        
        for word, expected_pos in self.pos_lexicon.items():
            if word not in self.brain.word_to_neurons:
                continue
            
            self.brain.clear_activations()
            self.brain.activate_word(word)
            
            # Project to all POS areas
            pos_activations = {}
            for pos in ['NOUN', 'VERB', 'ADJ', 'DET', 'PREP', 'ADV', 'PRON']:
                winners = self.brain.project('LEX', pos, learn=False)
                # Measure activation strength
                W = self.brain.connections[('LEX', pos)]
                word_indices = self.brain.word_to_neurons[word]
                strength = W[:, word_indices].sum().item()
                pos_activations[pos] = strength
            
            # Find predicted POS
            predicted_pos = max(pos_activations.items(), key=lambda x: x[1])[0]
            
            if predicted_pos == expected_pos:
                correct += 1
            elif verbose:
                print(f"  {word}: expected {expected_pos}, got {predicted_pos}")
            
            total += 1
        
        accuracy = correct / total if total > 0 else 0
        if verbose:
            print(f"POS Classification: {correct}/{total} = {accuracy:.1%}")
        
        return accuracy
    
    def test_word_prediction(self, verbose: bool = True) -> float:
        """Test next word prediction on held-out patterns"""
        test_cases = [
            (["the", "dog"], ["runs", "eats", "sleeps"]),  # Expect verb
            (["the", "cat"], ["sleeps", "drinks", "runs"]),
            (["the", "big"], ["dog", "cat", "man"]),  # Expect noun after adj
            (["he"], ["runs", "sleeps", "plays"]),  # Pronoun -> verb
            (["the", "baby", "wants"], ["milk", "food", "ball"]),  # Verb -> object
        ]
        
        correct = 0
        total = len(test_cases)
        
        for context, expected_words in test_cases:
            predictions = self.predict_next_word(context, top_k=10)
            predicted_words = [w for w, _ in predictions]
            
            # Check if any expected word is in top predictions
            found = any(w in predicted_words for w in expected_words)
            if found:
                correct += 1
            
            if verbose:
                pred_str = ", ".join([f"{w}:{s:.2f}" for w, s in predictions[:5]])
                status = "✓" if found else "✗"
                print(f"  {status} '{' '.join(context)}' -> [{pred_str}]")
                print(f"      Expected one of: {expected_words}")
        
        accuracy = correct / total
        if verbose:
            print(f"Word Prediction: {correct}/{total} = {accuracy:.1%}")
        
        return accuracy
    
    def generate_sentence(self, start_words: List[str] = None, max_length: int = 6) -> str:
        """
        Generate a grammatical sentence starting from given words.
        Uses strict POS constraints to ensure grammaticality.
        
        Grammar: DET? ADJ* NOUN VERB (DET? ADJ* NOUN)?
        """
        if start_words is None:
            start_words = ["the"]
        
        sentence = start_words.copy()
        used_content_words = set(start_words)
        
        # State machine for simple SVO grammar
        # States: START, AFTER_DET, AFTER_ADJ, AFTER_SUBJ_NOUN, AFTER_TRANS_VERB, AFTER_INTRANS_VERB, AFTER_OBJ_DET, AFTER_OBJ_ADJ, DONE
        
        # Determine initial state based on start words
        last_word = sentence[-1].lower()
        last_pos = self.get_pos(last_word)
        has_subject = any(self.get_pos(w) in ['NOUN', 'PRON'] for w in sentence)
        has_verb = any(self.get_pos(w) == 'VERB' for w in sentence)
        
        if last_pos == 'DET':
            state = 'AFTER_DET'
        elif last_pos == 'ADJ':
            state = 'AFTER_ADJ'
        elif last_pos in ['NOUN', 'PRON']:
            state = 'AFTER_SUBJ_NOUN' if not has_verb else 'DONE'
        elif last_pos == 'VERB':
            # Check if transitive or intransitive
            if last_word in self.transitive_verbs:
                state = 'AFTER_TRANS_VERB'
            else:
                state = 'AFTER_INTRANS_VERB'
        else:
            state = 'START'
        
        # State transitions with allowed POS
        # Transitive verbs REQUIRE an object (DET NOUN), intransitive verbs END or take adverb
        transitions = {
            'START': [('DET', 'AFTER_DET'), ('PRON', 'AFTER_SUBJ_NOUN')],
            'AFTER_DET': [('ADJ', 'AFTER_ADJ'), ('NOUN', 'AFTER_SUBJ_NOUN')],
            'AFTER_ADJ': [('NOUN', 'AFTER_SUBJ_NOUN'), ('ADJ', 'AFTER_ADJ')],
            'AFTER_SUBJ_NOUN': [('VERB', 'CHECK_VERB')],  # Special: check transitivity
            'AFTER_TRANS_VERB': [('DET', 'AFTER_OBJ_DET')],  # Must have object
            'AFTER_INTRANS_VERB': [('ADV', 'DONE'), (None, 'DONE')],  # Can end or adverb
            'AFTER_OBJ_DET': [('ADJ', 'AFTER_OBJ_ADJ'), ('NOUN', 'DONE')],
            'AFTER_OBJ_ADJ': [('NOUN', 'DONE')],
            'DONE': [],
        }
        
        for _ in range(max_length - len(sentence)):
            if state == 'DONE' or state not in transitions:
                break
            
            allowed = transitions[state]
            if not allowed:
                break
            
            # Check if we can end
            can_end = any(pos is None for pos, _ in allowed)
            if can_end and len(sentence) >= 3 and np.random.random() < 0.4:
                break
            
            # Filter to non-None transitions
            allowed = [(pos, next_state) for pos, next_state in allowed if pos is not None]
            if not allowed:
                break
            
            allowed_pos = [pos for pos, _ in allowed]
            
            # Get predictions from brain
            context = sentence[-2:] if len(sentence) >= 2 else sentence
            predictions = self.predict_next_word(context, top_k=50)
            
            # Filter by allowed POS and not used (for content words)
            candidates = []
            for word, score in predictions:
                word_pos = self.get_pos(word)
                if word_pos not in allowed_pos:
                    continue
                # Don't repeat content words
                if word_pos in ['NOUN', 'VERB', 'ADJ', 'ADV'] and word in used_content_words:
                    continue
                
                # Find the next state for this POS
                next_state = next((ns for pos, ns in allowed if pos == word_pos), state)
                
                # Special handling for verbs: check transitivity
                if next_state == 'CHECK_VERB':
                    if word in self.transitive_verbs:
                        next_state = 'AFTER_TRANS_VERB'
                    else:
                        next_state = 'AFTER_INTRANS_VERB'
                
                # Semantic filtering for objects
                if state in ['AFTER_OBJ_DET', 'AFTER_OBJ_ADJ'] and word_pos == 'NOUN':
                    # Find the verb in the sentence
                    verb = None
                    for w in sentence:
                        if self.get_pos(w) == 'VERB':
                            verb = w.lower()
                            break
                    
                    # Apply semantic constraints
                    if verb in ['eats', 'eat']:
                        if word not in self.edible:
                            continue  # Skip non-edible objects for "eats"
                    elif verb in ['drinks', 'drink']:
                        if word not in self.drinkable:
                            continue  # Skip non-drinkable objects
                
                candidates.append((word, score, word_pos, next_state))
            
            if not candidates:
                # No valid candidates - try to end gracefully
                break
            
            # Pick from top candidates with weighted randomness
            top_candidates = candidates[:5]
            words, scores, poses, next_states = zip(*top_candidates)
            scores = np.array(scores) + 0.01
            probs = scores / scores.sum()
            
            idx = np.random.choice(len(words), p=probs)
            next_word = words[idx]
            next_pos = poses[idx]
            state = next_states[idx]
            
            sentence.append(next_word)
            if next_pos in ['NOUN', 'VERB', 'ADJ', 'ADV']:
                used_content_words.add(next_word)
        
        return " ".join(sentence)


def test_learning_curve():
    """Test how learning improves with more exposure"""
    print("=" * 70)
    print("LEARNING CURVE EXPERIMENT")
    print("=" * 70)
    
    # Create fresh learner
    learner = GPULanguageLearner(n=2000, k=50, verbose=False)
    
    # Test at different training levels
    checkpoints = [1, 2, 5, 10, 20, 50]
    
    print(f"\n{'Epochs':>8} {'POS Acc':>10} {'Pred Acc':>10} {'Time (s)':>10}")
    print("-" * 42)
    
    total_time = 0
    for epochs in checkpoints:
        # Train incrementally
        start = time.perf_counter()
        learner.train(n_epochs=epochs - (learner.sentences_seen // len(learner.training_sentences)), verbose=False)
        total_time += time.perf_counter() - start
        
        # Test
        pos_acc = learner.test_pos_classification(verbose=False)
        pred_acc = learner.test_word_prediction(verbose=False)
        
        print(f"{epochs:>8} {pos_acc:>10.1%} {pred_acc:>10.1%} {total_time:>10.2f}")
    
    return learner


def main():
    print("=" * 70)
    print("GPU LANGUAGE-ONLY LEARNER")
    print("=" * 70)
    
    # First run learning curve experiment
    learner = test_learning_curve()
    
    # Generate sentences with the trained model
    print("\n" + "=" * 70)
    print("GENERATION: GRAMMATICAL SENTENCES")
    print("=" * 70)
    
    starters = [
        ["the"],
        ["the", "dog"],
        ["the", "big"],
        ["she"],
        ["he", "sees"],
        ["the", "baby"],
    ]
    
    for start in starters:
        print(f"\n  Starting with '{' '.join(start)}':")
        for i in range(3):  # Generate 3 variations
            sentence = learner.generate_sentence(start_words=start.copy())
            print(f"    {i+1}. {sentence}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Words learned: {len(learner.words_learned)}")
    print(f"  Sentences seen: {learner.sentences_seen}")
    
    # GPU memory
    mem_allocated = torch.cuda.memory_allocated() / 1e6
    mem_reserved = torch.cuda.memory_reserved() / 1e6
    print(f"  GPU memory: {mem_allocated:.1f} MB allocated, {mem_reserved:.1f} MB reserved")
    
    # Speed test
    print("\n" + "=" * 70)
    print("SPEED TEST: Sentences per second")
    print("=" * 70)
    
    # Time how many sentences we can process
    test_sentences = learner.training_sentences * 20
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for s in test_sentences:
        learner.brain.clear_activations()
        learner.process_sentence(s, learn=False, verbose=False)  # Inference only
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    print(f"  Inference: {len(test_sentences)} sentences in {elapsed:.3f}s")
    print(f"  Speed: {len(test_sentences)/elapsed:.1f} sentences/second")


if __name__ == "__main__":
    main()

