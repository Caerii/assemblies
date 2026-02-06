"""
Test Emergent Retrieval with VP Component Areas
================================================

Tests the new VP_SUBJ, VP_VERB, VP_OBJ areas for truly emergent retrieval.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from nemo.language.emergent.learner import EmergentLanguageLearner
from nemo.language.emergent.params import GroundingContext
from nemo.language.emergent.areas import Area
from nemo.language.emergent.generation.emergent_retriever import EmergentRetriever, EmergentGenerator


def create_grounding(visual=None, motor=None, social=None):
    """Helper to create grounding context."""
    return GroundingContext(
        visual=visual or [],
        motor=motor or [],
        social=social or []
    )


def test_vp_component_learning():
    """Test that VP component areas are populated during learning."""
    print("\n" + "="*70)
    print("TEST: VP Component Area Learning")
    print("="*70)
    
    learner = EmergentLanguageLearner(verbose=False)
    
    # Train some sentences
    sentences = [
        (['the', 'dog', 'runs'], 
         [create_grounding(), create_grounding(visual=['dog']), create_grounding(motor=['run'])],
         [None, 'agent', 'action']),
        (['the', 'cat', 'sleeps'],
         [create_grounding(), create_grounding(visual=['cat']), create_grounding(motor=['sleep'])],
         [None, 'agent', 'action']),
        (['the', 'bird', 'flies'],
         [create_grounding(), create_grounding(visual=['bird']), create_grounding(motor=['fly'])],
         [None, 'agent', 'action']),
    ]
    
    # Train multiple times
    print("Training sentences...")
    for epoch in range(5):
        for words, contexts, roles in sentences:
            learner.present_grounded_sentence(words, contexts, roles=roles)
    
    # Check what's in the VP component areas
    print("\nVP assemblies learned:")
    for vp_key in learner.brain.learned_assemblies[Area.VP].keys():
        print(f"  {vp_key}")
    
    print("\nVP_SUBJ assemblies learned:")
    for vp_key in learner.brain.learned_assemblies[Area.VP_SUBJ].keys():
        print(f"  {vp_key}")
    
    print("\nVP_VERB assemblies learned:")
    for vp_key in learner.brain.learned_assemblies[Area.VP_VERB].keys():
        print(f"  {vp_key}")
    
    # Check if we have the component assemblies
    has_vp_subj = len(learner.brain.learned_assemblies[Area.VP_SUBJ]) > 0
    has_vp_verb = len(learner.brain.learned_assemblies[Area.VP_VERB]) > 0
    
    print(f"\nVP_SUBJ populated: {has_vp_subj}")
    print(f"VP_VERB populated: {has_vp_verb}")
    
    return learner


def test_emergent_retrieval(learner):
    """Test emergent retrieval using VP component areas."""
    print("\n" + "="*70)
    print("TEST: Emergent Retrieval")
    print("="*70)
    
    retriever = EmergentRetriever(learner)
    
    # Test "Who runs?"
    print("\nQuery: 'Who runs?'")
    subjects = retriever.find_subjects_for_verb('runs', min_overlap=0.05)
    print(f"Subjects found: {subjects}")
    
    # Test "Who sleeps?"
    print("\nQuery: 'Who sleeps?'")
    subjects = retriever.find_subjects_for_verb('sleeps', min_overlap=0.05)
    print(f"Subjects found: {subjects}")
    
    # Test "What does dog do?"
    print("\nQuery: 'What does dog do?'")
    verbs = retriever.find_verbs_for_subject('dog', min_overlap=0.05)
    print(f"Verbs found: {verbs}")
    
    # Test "What does cat do?"
    print("\nQuery: 'What does cat do?'")
    verbs = retriever.find_verbs_for_subject('cat', min_overlap=0.05)
    print(f"Verbs found: {verbs}")
    
    # Test pattern existence
    print("\nPattern existence checks:")
    for subj, verb in [('dog', 'runs'), ('dog', 'sleeps'), ('cat', 'sleeps'), ('cat', 'runs')]:
        exists, conf = retriever.check_pattern_exists(subj, verb, min_overlap=0.05)
        print(f"  '{subj} {verb}': exists={exists}, confidence={conf:.3f}")


def test_emergent_generator(learner):
    """Test the full emergent generator."""
    print("\n" + "="*70)
    print("TEST: Emergent Generator")
    print("="*70)
    
    generator = EmergentGenerator(learner)
    
    queries = [
        ['who', 'runs'],
        ['who', 'sleeps'],
        ['what', 'does', 'the', 'dog', 'do'],
        ['what', 'does', 'the', 'cat', 'do'],
        ['does', 'the', 'dog', 'run'],
        ['does', 'the', 'cat', 'sleep'],
        ['does', 'the', 'dog', 'sleep'],
    ]
    
    for query in queries:
        response = generator.generate(query)
        print(f"\nQ: {' '.join(query)}")
        print(f"A: {response}")


def test_with_transitive_sentences():
    """Test with transitive sentences (subject-verb-object)."""
    print("\n" + "="*70)
    print("TEST: Transitive Sentences (Subject-Verb-Object)")
    print("="*70)
    
    learner = EmergentLanguageLearner(verbose=False)
    
    sentences = [
        (['the', 'dog', 'chases', 'the', 'cat'],
         [create_grounding(), create_grounding(visual=['dog']), create_grounding(motor=['chase']),
          create_grounding(), create_grounding(visual=['cat'])],
         [None, 'agent', 'action', None, 'patient']),
        (['the', 'cat', 'catches', 'the', 'mouse'],
         [create_grounding(), create_grounding(visual=['cat']), create_grounding(motor=['catch']),
          create_grounding(), create_grounding(visual=['mouse'])],
         [None, 'agent', 'action', None, 'patient']),
    ]
    
    print("Training transitive sentences...")
    for epoch in range(5):
        for words, contexts, roles in sentences:
            learner.present_grounded_sentence(words, contexts, roles=roles)
    
    print("\nVP_OBJ assemblies learned:")
    for vp_key in learner.brain.learned_assemblies[Area.VP_OBJ].keys():
        print(f"  {vp_key}")
    
    retriever = EmergentRetriever(learner)
    
    # Test object retrieval
    print("\nQuery: 'What does dog chase?'")
    objects = retriever.find_objects_for_subject_verb('dog', 'chases', min_overlap=0.05)
    print(f"Objects found: {objects}")
    
    print("\nQuery: 'What does cat catch?'")
    objects = retriever.find_objects_for_subject_verb('cat', 'catches', min_overlap=0.05)
    print(f"Objects found: {objects}")


if __name__ == "__main__":
    learner = test_vp_component_learning()
    test_emergent_retrieval(learner)
    test_emergent_generator(learner)
    test_with_transitive_sentences()


