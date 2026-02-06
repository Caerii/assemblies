"""
Interactive Learner
===================

The main interactive NEMO system that:
1. Takes user input
2. Learns from each interaction
3. Generates responses using learned patterns
4. Continuously grows its knowledge

This extends EmergentLanguageLearner modularly.
"""

from typing import Optional, List
import re

from ..learner import EmergentLanguageLearner
from ..curriculum import get_training_curriculum
from .dialogue import DialogueState, Turn
from .grounding import GroundingInference
from .response import ResponseGenerator


class InteractiveLearner:
    """
    Interactive NEMO that learns from conversation.
    
    Wraps EmergentLanguageLearner and adds:
    - Dialogue state tracking
    - Grounding inference
    - Response generation
    - Continuous learning
    
    Usage:
        nemo = InteractiveLearner()
        nemo.bootstrap()  # Optional initial training
        
        while True:
            user_input = input("You: ")
            response = nemo.interact(user_input)
            print(f"NEMO: {response}")
    """
    
    def __init__(self, 
                 use_cuda: bool = True,
                 bootstrap_training: bool = True,
                 verbose: bool = False):
        """
        Initialize interactive NEMO.
        
        Args:
            use_cuda: Use CUDA backend for speed
            bootstrap_training: Do initial training on startup
            verbose: Print learning details
        """
        self.verbose = verbose
        
        # Core learner (the brain)
        # The CUDA backend is controlled at the brain level
        from ..brain import EmergentNemoBrain
        from ..params import EmergentParams
        
        params = EmergentParams()
        brain = EmergentNemoBrain(params, verbose=False, use_cuda_backend=use_cuda)
        self.learner = EmergentLanguageLearner(params=params, verbose=False)
        self.learner.brain = brain  # Replace with CUDA-enabled brain
        
        # Dialogue state (working memory)
        self.dialogue = DialogueState()
        
        # Modules for interactive learning
        self.grounding_inference = GroundingInference(self.learner)
        self.response_generator = ResponseGenerator(self.learner)
        
        # Statistics
        self.total_interactions = 0
        self.words_learned = 0
        self.patterns_learned = 0
        
        # Bootstrap if requested
        if bootstrap_training:
            self.bootstrap()
    
    def bootstrap(self, num_epochs: int = 2, include_dialogue: bool = True):
        """
        Bootstrap with initial training data.
        
        This gives NEMO a starting vocabulary and patterns
        so it can understand basic inputs.
        
        Args:
            num_epochs: Number of training epochs
            include_dialogue: Include dialogue Q-A patterns in training
        """
        if self.verbose:
            print("Bootstrapping NEMO with curriculum...")
        
        # Get training data from modular curriculum
        training_data = get_training_curriculum(
            include_dialogue=include_dialogue,
            stage=4,  # Full grammar stage
            seed=42
        )
        
        # Train for a few epochs
        for epoch in range(num_epochs):
            for sentence in training_data:
                self.learner.present_grounded_sentence(
                    sentence.words,
                    sentence.contexts,
                    sentence.roles,
                    sentence.mood
                )
        
        self.words_learned = len(self.learner.word_count)
        
        if self.verbose:
            print(f"Bootstrap complete: {self.words_learned} words learned")
    
    def interact(self, user_input: str) -> str:
        """
        Process user input and generate response.
        
        This is the main interaction loop:
        1. Tokenize input
        2. Classify input type
        3. Learn from input
        4. Generate response
        5. Learn from response (self-learning)
        
        Args:
            user_input: Raw text from user
        
        Returns:
            Response string
        """
        # Tokenize
        words = self._tokenize(user_input)
        
        if not words:
            return ""
        
        # Add to dialogue history
        input_type = self.response_generator.classify_input(words, self.dialogue)
        user_turn = self.dialogue.add_turn("user", user_input, words, input_type)
        
        # Learn from this input
        self._learn_from_input(words, user_turn)
        
        # Generate response
        response = self.response_generator.generate(
            words, input_type, self.dialogue
        )
        
        # Add response to dialogue
        response_words = self._tokenize(response)
        system_turn = self.dialogue.add_turn("system", response, response_words)
        
        # Learn from our own response (self-reinforcement)
        if response_words:
            self._learn_from_response(response_words, system_turn)
        
        self.total_interactions += 1
        
        return response
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        # Simple tokenization - split on whitespace and punctuation
        text = text.lower().strip()
        # Keep punctuation as separate tokens
        text = re.sub(r'([.,!?])', r' \1', text)
        words = text.split()
        # Remove empty strings and standalone punctuation
        words = [w for w in words if w and w not in {".", ",", "!", "?"}]
        return words
    
    def _learn_from_input(self, words: List[str], turn: Turn):
        """Learn from user input."""
        # Infer grounding for all words
        contexts = self.grounding_inference.infer_sentence_grounding(words)
        
        # Infer roles
        roles = self.grounding_inference.infer_roles(words)
        
        # Determine mood
        mood = "interrogative" if turn.turn_type == "question" else "declarative"
        
        # Present to learner
        self.learner.present_grounded_sentence(words, contexts, roles, mood)
        
        # Track what was learned
        new_words = [w for w in words if self.learner.word_count.get(w, 0) == 1]
        turn.learned_words = new_words
        
        # Track patterns
        pattern = self._extract_pattern(words)
        if pattern:
            turn.learned_patterns.append(pattern)
        
        # Update stats
        self.words_learned = len(self.learner.word_count)
        
        if self.verbose and new_words:
            print(f"  [Learned new words: {new_words}]")
    
    def _learn_from_response(self, words: List[str], turn: Turn):
        """
        Learn from our own response.
        
        This is a form of self-reinforcement:
        - Strengthens patterns we use in responses
        - Helps consolidate learned knowledge
        """
        if len(words) < 2:
            return
        
        # Only learn from substantive responses
        # (not just "ok" or "learned")
        if words[0] in ["ok", "learned", "i"]:
            return
        
        # Create grounding for response words
        contexts = self.grounding_inference.infer_sentence_grounding(words)
        roles = self.grounding_inference.infer_roles(words)
        
        # Present as declarative
        self.learner.present_grounded_sentence(words, contexts, roles, "declarative")
    
    def _extract_pattern(self, words: List[str]) -> Optional[str]:
        """Extract SVO pattern from sentence."""
        subject = None
        verb = None
        obj = None
        
        for word in words:
            cat, _ = self.learner.get_emergent_category(word)
            
            if cat == "VERB" and verb is None:
                verb = word
            elif cat in ["NOUN", "PRONOUN"]:
                if verb is None and subject is None:
                    subject = word
                elif verb is not None and obj is None:
                    obj = word
        
        if subject and verb:
            if obj:
                return f"{subject}_{verb}_{obj}"
            return f"{subject}_{verb}"
        return None
    
    def teach(self, word: str, category: str, examples: List[str]):
        """
        Explicitly teach NEMO a new word.
        
        Args:
            word: The word to teach
            category: Expected category (NOUN, VERB, etc.)
            examples: Example sentences using the word
        
        Example:
            nemo.teach("giraffe", "NOUN", [
                "the giraffe is tall",
                "giraffes eat leaves",
                "I see a giraffe"
            ])
        """
        from ..params import GroundingContext
        
        # Create grounding based on category
        ctx = GroundingContext()
        if category == "NOUN":
            ctx.visual = [word]
        elif category == "VERB":
            ctx.motor = [word]
        elif category == "ADJECTIVE":
            ctx.property = [word]
        
        # Present word in isolation first
        self.learner.present_word_with_grounding(word, ctx)
        
        # Then present in example sentences
        for example in examples:
            words = self._tokenize(example)
            contexts = self.grounding_inference.infer_sentence_grounding(words)
            # Override the target word's grounding
            for i, w in enumerate(words):
                if w == word:
                    contexts[i] = ctx
            
            roles = self.grounding_inference.infer_roles(words)
            self.learner.present_grounded_sentence(words, contexts, roles, "declarative")
        
        if self.verbose:
            print(f"Taught '{word}' as {category}")
    
    def query_knowledge(self) -> dict:
        """
        Query NEMO's current knowledge state.
        
        Returns summary of:
        - Vocabulary size and categories
        - Learned patterns
        - Dialogue statistics
        """
        vocab = self.learner.get_vocabulary_by_category()
        
        return {
            "vocabulary_size": len(self.learner.word_count),
            "categories": {cat: len(words) for cat, words in vocab.items()},
            "top_words": dict(sorted(
                self.learner.word_count.items(),
                key=lambda x: -x[1]
            )[:10]),
            "dialogue_turns": self.dialogue.total_turns,
            "total_interactions": self.total_interactions,
        }
    
    def reset_dialogue(self):
        """Start a new conversation (clear working memory)."""
        self.dialogue.clear()
    
    def save_state(self, filepath: str):
        """Save NEMO's learned state to file."""
        import pickle
        state = {
            "word_count": dict(self.learner.word_count),
            "word_grounding": dict(self.learner.word_grounding),
            "category_transitions": dict(self.learner.category_transitions),
            "total_interactions": self.total_interactions,
            "words_learned": self.words_learned,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
    
    def load_state(self, filepath: str):
        """Load NEMO's learned state from file."""
        import pickle
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.learner.word_count.update(state["word_count"])
        self.learner.word_grounding.update(state["word_grounding"])
        self.learner.category_transitions.update(state["category_transitions"])
        self.total_interactions = state.get("total_interactions", 0)
        self.words_learned = state.get("words_learned", len(self.learner.word_count))


def run_interactive_session():
    """Run an interactive NEMO session in the terminal."""
    print("=" * 60)
    print("NEMO Interactive Language Learner")
    print("=" * 60)
    print()
    print("Initializing...")
    
    nemo = InteractiveLearner(verbose=True)
    
    print()
    print("Ready! Type 'quit' to exit, 'status' for knowledge summary.")
    print("-" * 60)
    print()
    
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        
        if user_input.lower() == "status":
            knowledge = nemo.query_knowledge()
            print("\nNEMO Status:")
            print(f"  Vocabulary: {knowledge['vocabulary_size']} words")
            print(f"  Categories: {knowledge['categories']}")
            print(f"  Interactions: {knowledge['total_interactions']}")
            print()
            continue
        
        if user_input.lower() == "reset":
            nemo.reset_dialogue()
            print("(Dialogue reset)")
            continue
        
        response = nemo.interact(user_input)
        print(f"NEMO: {response}")
        print()


if __name__ == "__main__":
    run_interactive_session()

