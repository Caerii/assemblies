"""Determiners - Articles, Demonstratives, Possessives, Quantifiers"""

DETERMINERS = [
    # Articles
    {"lemma": "the", "forms": {}, "freq": 6.9, "aoa": 1.5, "features": {"definite": True}},
    {"lemma": "a", "forms": {}, "freq": 6.5, "aoa": 1.5, "features": {"definite": False}},
    {"lemma": "an", "forms": {}, "freq": 5.8, "aoa": 1.8, "features": {"definite": False}},
    
    # Demonstratives
    {"lemma": "this", "forms": {"plural": "these"}, "freq": 5.5, "aoa": 2.0, "features": {"demonstrative": True, "proximal": True}},
    {"lemma": "that", "forms": {"plural": "those"}, "freq": 5.8, "aoa": 2.0, "features": {"demonstrative": True, "proximal": False}},
    {"lemma": "these", "forms": {}, "freq": 4.8, "aoa": 2.5, "features": {"demonstrative": True, "proximal": True, "plural": True}},
    {"lemma": "those", "forms": {}, "freq": 4.5, "aoa": 2.5, "features": {"demonstrative": True, "proximal": False, "plural": True}},
    
    # Possessives
    {"lemma": "my", "forms": {}, "freq": 5.5, "aoa": 1.8, "features": {"possessive": True, "person": 1}},
    {"lemma": "your", "forms": {}, "freq": 5.3, "aoa": 2.0, "features": {"possessive": True, "person": 2}},
    {"lemma": "his", "forms": {}, "freq": 5.0, "aoa": 2.2, "features": {"possessive": True, "person": 3, "gender": "m"}},
    {"lemma": "her", "forms": {}, "freq": 5.0, "aoa": 2.2, "features": {"possessive": True, "person": 3, "gender": "f"}},
    {"lemma": "its", "forms": {}, "freq": 4.5, "aoa": 3.0, "features": {"possessive": True, "person": 3, "gender": "n"}},
    {"lemma": "our", "forms": {}, "freq": 4.8, "aoa": 2.5, "features": {"possessive": True, "person": 1, "plural": True}},
    {"lemma": "their", "forms": {}, "freq": 4.8, "aoa": 2.8, "features": {"possessive": True, "person": 3, "plural": True}},
    
    # Quantifiers
    {"lemma": "some", "forms": {}, "freq": 5.2, "aoa": 2.5, "features": {"quantifier": True}},
    {"lemma": "any", "forms": {}, "freq": 4.8, "aoa": 3.0, "features": {"quantifier": True}},
    {"lemma": "no", "forms": {}, "freq": 5.5, "aoa": 1.8, "features": {"quantifier": True, "negative": True}},
    {"lemma": "every", "forms": {}, "freq": 4.5, "aoa": 3.5, "features": {"quantifier": True, "universal": True}},
    {"lemma": "each", "forms": {}, "freq": 4.2, "aoa": 4.0, "features": {"quantifier": True, "distributive": True}},
    {"lemma": "all", "forms": {}, "freq": 5.3, "aoa": 2.5, "features": {"quantifier": True, "universal": True}},
    {"lemma": "both", "forms": {}, "freq": 4.5, "aoa": 3.5, "features": {"quantifier": True, "dual": True}},
    {"lemma": "many", "forms": {}, "freq": 4.8, "aoa": 3.0, "features": {"quantifier": True, "count": True}},
    {"lemma": "much", "forms": {}, "freq": 4.5, "aoa": 3.0, "features": {"quantifier": True, "mass": True}},
    {"lemma": "few", "forms": {}, "freq": 4.2, "aoa": 3.5, "features": {"quantifier": True, "count": True, "small": True}},
    {"lemma": "little", "forms": {}, "freq": 4.8, "aoa": 2.5, "features": {"quantifier": True, "mass": True, "small": True}},
    {"lemma": "several", "forms": {}, "freq": 4.0, "aoa": 5.0, "features": {"quantifier": True}},
    {"lemma": "most", "forms": {}, "freq": 4.5, "aoa": 4.0, "features": {"quantifier": True}},
    {"lemma": "other", "forms": {"plural": "others"}, "freq": 4.8, "aoa": 3.0, "features": {"quantifier": True}},
    {"lemma": "another", "forms": {}, "freq": 4.5, "aoa": 3.5, "features": {"quantifier": True}},
    
    # Interrogative
    {"lemma": "what", "forms": {}, "freq": 5.5, "aoa": 2.0, "features": {"interrogative": True}},
    {"lemma": "which", "forms": {}, "freq": 4.8, "aoa": 3.0, "features": {"interrogative": True}},
    {"lemma": "whose", "forms": {}, "freq": 3.8, "aoa": 4.0, "features": {"interrogative": True, "possessive": True}},
]

