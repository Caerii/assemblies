#!/usr/bin/env python
"""
Run an interactive NEMO session.

Usage:
    python -m nemo.language.emergent.interactive.run_session
"""

from .interactive_learner import InteractiveLearner


def main():
    print("=" * 60)
    print("NEMO Interactive Language Learner")
    print("=" * 60)
    print()
    print("Initializing NEMO...")
    
    nemo = InteractiveLearner(use_cuda=True, bootstrap_training=True, verbose=True)
    
    print()
    print("Commands:")
    print("  quit     - Exit the session")
    print("  status   - Show knowledge summary")
    print("  reset    - Clear dialogue history")
    print("  teach    - Teach a new word (teach <word> <category>)")
    print()
    print("NEMO learns from every interaction!")
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
        
        # Handle special commands
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        
        if user_input.lower() == "status":
            knowledge = nemo.query_knowledge()
            print()
            print("NEMO Knowledge Status:")
            print(f"  Vocabulary: {knowledge['vocabulary_size']} words")
            print(f"  Categories: {knowledge['categories']}")
            print(f"  Total interactions: {knowledge['total_interactions']}")
            print(f"  Top words: {list(knowledge['top_words'].keys())[:5]}")
            print()
            continue
        
        if user_input.lower() == "reset":
            nemo.reset_dialogue()
            print("(Dialogue history cleared)")
            continue
        
        if user_input.lower().startswith("teach "):
            parts = user_input.split()
            if len(parts) >= 3:
                word = parts[1]
                category = parts[2].upper()
                if category in ["NOUN", "VERB", "ADJECTIVE", "ADVERB"]:
                    nemo.teach(word, category, [f"the {word} is here"])
                    print(f"(Taught '{word}' as {category})")
                else:
                    print(f"(Unknown category: {category}. Use NOUN, VERB, ADJECTIVE, or ADVERB)")
            else:
                print("(Usage: teach <word> <category>)")
            continue
        
        # Normal interaction
        response = nemo.interact(user_input)
        print(f"NEMO: {response}")
        print()


if __name__ == "__main__":
    main()

