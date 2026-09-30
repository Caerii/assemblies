"""
Holdout bridge corpus — contexts for OOV next-token prediction.

Sentences use held-out lemmas (bird, finds, small) in positions that match
``collect_bridge_probes`` prefixes.  Lexicon training skips these words;
bridge / distributional phases still need observed transitions.
"""

# Subject-position holdout (prefix → trained continuation)
HOLDOUT_BRIDGE_CORPUS = [
  # bird as subject
    "the bird runs",
    "the bird sleeps",
    "the bird plays",
    "the bird sees the cat",
    "the bird finds the ball",
    "the bird eats the food",
    "the small bird plays",
    "the bird finds the small dog",
    # bird as object
    "the cat sees the bird",
    "the dog chases the bird",
    "the boy finds the bird",
    "the cat finds the small bird",
    "she sees the bird",
    # finds as verb
    "the boy finds the ball",
    "the girl finds the book",
    "the cat finds the food",
    "she finds the ball",
    "he finds the cat",
    "the dog finds the bird",
    # small as adjective
    "the small dog runs",
    "the small cat sleeps",
    "the small bird plays",
    "the boy finds the small dog",
    "the cat sees the small bird",
    "a small cat sees the bird",
    # multi-holdout frames (strain-aligned)
    "the bird finds the small dog",
    "the small bird finds the dog",
    "the dog finds the small bird",
    "she finds the small bird",
    "the small dog chases the bird",
    # Probe-aligned prefixes (balanced holdout geometry)
    "the small bird",
    "the bird finds",
    "the cat finds",
    "the dog finds",
    "she finds",
    "the boy finds",
    "the girl finds",
    "the small dog",
    "the small cat",
    "the bird chases",
    "the cat chases",
    "the dog sees",
    "the boy sees",
    "the bird finds the",
    "the cat finds the",
    "the dog finds the",
    "the boy finds the",
    "she finds the",
    "the bird finds the small",
    "the cat finds the small",
    "the dog finds the small",
]
