"""
Shared Trained Model for Tests
==============================

Trains a model ONCE and caches it for all tests to reuse.
This dramatically speeds up test runs.
"""

import time
from typing import Optional

# Global cached model
_cached_learner: Optional['EmergentLanguageLearner'] = None
_cached_data = None
_training_time: float = 0.0


def get_trained_learner(epochs: int = 3, force_retrain: bool = False):
    """
    Get a trained learner, using cache if available.
    
    Args:
        epochs: Number of training epochs (only used if training fresh)
        force_retrain: If True, retrain even if cached
        
    Returns:
        Trained EmergentLanguageLearner
    """
    global _cached_learner, _cached_data, _training_time
    
    if _cached_learner is not None and not force_retrain:
        return _cached_learner
    
    from src.nemo.language.emergent import EmergentLanguageLearner, create_training_data
    
    print("\n" + "="*60)
    print("TRAINING SHARED MODEL (will be reused for all tests)")
    print("="*60)
    
    learner = EmergentLanguageLearner(verbose=True)
    data = create_training_data()
    
    print(f"\nTraining on {len(data)} sentences × {epochs} epochs...")
    
    start = time.perf_counter()
    for epoch in range(epochs):
        for s in data:
            learner.present_grounded_sentence(
                s.words, s.contexts, roles=s.roles, mood=s.mood
            )
        print(f"  Epoch {epoch + 1}/{epochs} complete")
    
    _training_time = time.perf_counter() - start
    
    print(f"\nTraining complete in {_training_time:.1f}s")
    print(f"  Vocabulary: {len(learner.word_count)} words")
    print(f"  Sentences: {learner.sentences_seen}")
    print("="*60 + "\n")
    
    _cached_learner = learner
    _cached_data = data
    
    return learner


def get_training_data():
    """Get the training data (cached)."""
    global _cached_data
    
    if _cached_data is None:
        from src.nemo.language.emergent import create_training_data
        _cached_data = create_training_data()
    
    return _cached_data


def get_training_time() -> float:
    """Get the time it took to train the model."""
    return _training_time


def clear_cache():
    """Clear the cached model (useful for testing fresh training)."""
    global _cached_learner, _cached_data, _training_time
    _cached_learner = None
    _cached_data = None
    _training_time = 0.0

