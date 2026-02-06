"""Pronouns - Personal, Demonstrative, Interrogative, Relative, Indefinite"""

PRONOUNS_PERSONAL = [
    {"lemma": "I", "forms": {"obj": "me", "poss": "my", "poss_pron": "mine", "refl": "myself"}, "freq": 6.0, "aoa": 1.0, "domains": ["FUNCTION_WORD"], "features": {"personal": True, "person": 1, "number": "sg"}},
    {"lemma": "you", "forms": {"obj": "you", "poss": "your", "poss_pron": "yours", "refl": "yourself"}, "freq": 5.8, "aoa": 1.0, "domains": ["FUNCTION_WORD"], "features": {"personal": True, "person": 2}},
    {"lemma": "he", "forms": {"obj": "him", "poss": "his", "poss_pron": "his", "refl": "himself"}, "freq": 5.5, "aoa": 1.5, "domains": ["FUNCTION_WORD"], "features": {"personal": True, "person": 3, "number": "sg", "gender": "m"}},
    {"lemma": "she", "forms": {"obj": "her", "poss": "her", "poss_pron": "hers", "refl": "herself"}, "freq": 5.5, "aoa": 1.5, "domains": ["FUNCTION_WORD"], "features": {"personal": True, "person": 3, "number": "sg", "gender": "f"}},
    {"lemma": "it", "forms": {"obj": "it", "poss": "its", "poss_pron": "its", "refl": "itself"}, "freq": 5.8, "aoa": 1.5, "domains": ["FUNCTION_WORD"], "features": {"personal": True, "person": 3, "number": "sg", "gender": "n"}},
    {"lemma": "we", "forms": {"obj": "us", "poss": "our", "poss_pron": "ours", "refl": "ourselves"}, "freq": 5.2, "aoa": 2.0, "domains": ["FUNCTION_WORD"], "features": {"personal": True, "person": 1, "number": "pl"}},
    {"lemma": "they", "forms": {"obj": "them", "poss": "their", "poss_pron": "theirs", "refl": "themselves"}, "freq": 5.5, "aoa": 2.0, "domains": ["FUNCTION_WORD"], "features": {"personal": True, "person": 3, "number": "pl"}},
]

PRONOUNS_DEMONSTRATIVE = [
    {"lemma": "this", "forms": {"plural": "these"}, "freq": 5.5, "aoa": 2.0, "domains": ["FUNCTION_WORD"], "features": {"demonstrative": True, "proximal": True, "number": "sg"}},
    {"lemma": "that", "forms": {"plural": "those"}, "freq": 5.8, "aoa": 2.0, "domains": ["FUNCTION_WORD"], "features": {"demonstrative": True, "distal": True, "number": "sg"}},
    {"lemma": "these", "forms": {}, "freq": 4.8, "aoa": 2.5, "domains": ["FUNCTION_WORD"], "features": {"demonstrative": True, "proximal": True, "number": "pl"}},
    {"lemma": "those", "forms": {}, "freq": 4.5, "aoa": 2.5, "domains": ["FUNCTION_WORD"], "features": {"demonstrative": True, "distal": True, "number": "pl"}},
]

PRONOUNS_INTERROGATIVE = [
    {"lemma": "who", "forms": {"obj": "whom", "poss": "whose"}, "freq": 5.2, "aoa": 2.0, "domains": ["FUNCTION_WORD"], "features": {"interrogative": True, "human": True}},
    {"lemma": "what", "forms": {}, "freq": 5.5, "aoa": 1.5, "domains": ["FUNCTION_WORD"], "features": {"interrogative": True}},
    {"lemma": "which", "forms": {}, "freq": 4.8, "aoa": 3.0, "domains": ["FUNCTION_WORD"], "features": {"interrogative": True, "selective": True}},
    {"lemma": "whose", "forms": {}, "freq": 3.8, "aoa": 4.0, "domains": ["FUNCTION_WORD"], "features": {"interrogative": True, "possessive": True}},
    {"lemma": "whom", "forms": {}, "freq": 3.5, "aoa": 6.0, "domains": ["FUNCTION_WORD"], "features": {"interrogative": True, "human": True, "object": True}},
]

PRONOUNS_RELATIVE = [
    {"lemma": "who", "forms": {"obj": "whom", "poss": "whose"}, "freq": 5.2, "aoa": 4.0, "domains": ["FUNCTION_WORD"], "features": {"relative": True, "human": True}},
    {"lemma": "which", "forms": {}, "freq": 4.8, "aoa": 4.0, "domains": ["FUNCTION_WORD"], "features": {"relative": True}},
    {"lemma": "that", "forms": {}, "freq": 5.8, "aoa": 3.0, "domains": ["FUNCTION_WORD"], "features": {"relative": True}},
    {"lemma": "whose", "forms": {}, "freq": 3.8, "aoa": 5.0, "domains": ["FUNCTION_WORD"], "features": {"relative": True, "possessive": True}},
    {"lemma": "where", "forms": {}, "freq": 5.0, "aoa": 3.5, "domains": ["FUNCTION_WORD"], "features": {"relative": True, "locative": True}},
    {"lemma": "when", "forms": {}, "freq": 5.2, "aoa": 3.0, "domains": ["FUNCTION_WORD"], "features": {"relative": True, "temporal": True}},
    {"lemma": "why", "forms": {}, "freq": 4.8, "aoa": 3.0, "domains": ["FUNCTION_WORD"], "features": {"relative": True, "reason": True}},
    {"lemma": "how", "forms": {}, "freq": 5.2, "aoa": 2.5, "domains": ["FUNCTION_WORD"], "features": {"relative": True, "manner": True}},
]

PRONOUNS_INDEFINITE = [
    {"lemma": "someone", "forms": {}, "freq": 4.5, "aoa": 3.0, "domains": ["FUNCTION_WORD"], "features": {"indefinite": True, "human": True, "affirmative": True}},
    {"lemma": "somebody", "forms": {}, "freq": 4.2, "aoa": 3.0, "domains": ["FUNCTION_WORD"], "features": {"indefinite": True, "human": True, "affirmative": True}},
    {"lemma": "something", "forms": {}, "freq": 5.0, "aoa": 2.5, "domains": ["FUNCTION_WORD"], "features": {"indefinite": True, "affirmative": True}},
    {"lemma": "somewhere", "forms": {}, "freq": 4.0, "aoa": 3.5, "domains": ["FUNCTION_WORD"], "features": {"indefinite": True, "locative": True, "affirmative": True}},
    {"lemma": "anyone", "forms": {}, "freq": 4.5, "aoa": 3.5, "domains": ["FUNCTION_WORD"], "features": {"indefinite": True, "human": True}},
    {"lemma": "anybody", "forms": {}, "freq": 4.0, "aoa": 3.5, "domains": ["FUNCTION_WORD"], "features": {"indefinite": True, "human": True}},
    {"lemma": "anything", "forms": {}, "freq": 4.8, "aoa": 3.0, "domains": ["FUNCTION_WORD"], "features": {"indefinite": True}},
    {"lemma": "anywhere", "forms": {}, "freq": 4.0, "aoa": 4.0, "domains": ["FUNCTION_WORD"], "features": {"indefinite": True, "locative": True}},
    {"lemma": "no one", "forms": {}, "freq": 4.2, "aoa": 3.5, "domains": ["FUNCTION_WORD"], "features": {"indefinite": True, "human": True, "negative": True}},
    {"lemma": "nobody", "forms": {}, "freq": 4.0, "aoa": 3.5, "domains": ["FUNCTION_WORD"], "features": {"indefinite": True, "human": True, "negative": True}},
    {"lemma": "nothing", "forms": {}, "freq": 4.8, "aoa": 3.0, "domains": ["FUNCTION_WORD"], "features": {"indefinite": True, "negative": True}},
    {"lemma": "nowhere", "forms": {}, "freq": 3.5, "aoa": 4.0, "domains": ["FUNCTION_WORD"], "features": {"indefinite": True, "locative": True, "negative": True}},
    {"lemma": "everyone", "forms": {}, "freq": 4.5, "aoa": 3.5, "domains": ["FUNCTION_WORD"], "features": {"indefinite": True, "human": True, "universal": True}},
    {"lemma": "everybody", "forms": {}, "freq": 4.2, "aoa": 3.5, "domains": ["FUNCTION_WORD"], "features": {"indefinite": True, "human": True, "universal": True}},
    {"lemma": "everything", "forms": {}, "freq": 4.8, "aoa": 3.0, "domains": ["FUNCTION_WORD"], "features": {"indefinite": True, "universal": True}},
    {"lemma": "everywhere", "forms": {}, "freq": 3.8, "aoa": 3.5, "domains": ["FUNCTION_WORD"], "features": {"indefinite": True, "locative": True, "universal": True}},
    {"lemma": "each", "forms": {}, "freq": 4.5, "aoa": 4.0, "domains": ["FUNCTION_WORD"], "features": {"indefinite": True, "distributive": True}},
    {"lemma": "both", "forms": {}, "freq": 4.5, "aoa": 3.5, "domains": ["FUNCTION_WORD"], "features": {"indefinite": True, "dual": True}},
    {"lemma": "all", "forms": {}, "freq": 5.5, "aoa": 2.5, "domains": ["FUNCTION_WORD"], "features": {"indefinite": True, "universal": True}},
    {"lemma": "none", "forms": {}, "freq": 4.2, "aoa": 4.0, "domains": ["FUNCTION_WORD"], "features": {"indefinite": True, "negative": True}},
    {"lemma": "some", "forms": {}, "freq": 5.2, "aoa": 2.5, "domains": ["FUNCTION_WORD"], "features": {"indefinite": True}},
    {"lemma": "any", "forms": {}, "freq": 4.8, "aoa": 3.0, "domains": ["FUNCTION_WORD"], "features": {"indefinite": True}},
    {"lemma": "one", "forms": {}, "freq": 5.5, "aoa": 2.0, "domains": ["FUNCTION_WORD"], "features": {"indefinite": True, "impersonal": True}},
    {"lemma": "other", "forms": {"plural": "others"}, "freq": 4.8, "aoa": 3.0, "domains": ["FUNCTION_WORD"], "features": {"indefinite": True}},
    {"lemma": "another", "forms": {}, "freq": 4.5, "aoa": 3.5, "domains": ["FUNCTION_WORD"], "features": {"indefinite": True}},
]

# Combine all pronouns
PRONOUNS = (
    PRONOUNS_PERSONAL + PRONOUNS_DEMONSTRATIVE + PRONOUNS_INTERROGATIVE +
    PRONOUNS_RELATIVE + PRONOUNS_INDEFINITE
)

