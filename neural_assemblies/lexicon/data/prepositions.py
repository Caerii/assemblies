"""Prepositions - Spatial, Temporal, and Abstract"""

PREPOSITIONS_SPATIAL = [
    {"lemma": "in", "forms": {}, "freq": 6.0, "aoa": 1.5, "domains": ["SPACE"], "features": {"spatial": True, "containment": True}},
    {"lemma": "on", "forms": {}, "freq": 5.8, "aoa": 1.5, "domains": ["SPACE"], "features": {"spatial": True, "surface": True}},
    {"lemma": "at", "forms": {}, "freq": 5.8, "aoa": 2.0, "domains": ["SPACE"], "features": {"spatial": True, "point": True}},
    {"lemma": "to", "forms": {}, "freq": 6.2, "aoa": 1.5, "domains": ["SPACE"], "features": {"spatial": True, "goal": True}},
    {"lemma": "from", "forms": {}, "freq": 5.5, "aoa": 2.0, "domains": ["SPACE"], "features": {"spatial": True, "source": True}},
    {"lemma": "into", "forms": {}, "freq": 4.8, "aoa": 2.5, "domains": ["SPACE"], "features": {"spatial": True, "containment": True, "motion": True}},
    {"lemma": "out", "forms": {}, "freq": 5.2, "aoa": 1.5, "domains": ["SPACE"], "features": {"spatial": True, "containment": True, "motion": True}},
    {"lemma": "up", "forms": {}, "freq": 5.5, "aoa": 1.5, "domains": ["SPACE"], "features": {"spatial": True, "vertical": True}},
    {"lemma": "down", "forms": {}, "freq": 5.2, "aoa": 1.5, "domains": ["SPACE"], "features": {"spatial": True, "vertical": True}},
    {"lemma": "over", "forms": {}, "freq": 5.0, "aoa": 2.5, "domains": ["SPACE"], "features": {"spatial": True, "vertical": True}},
    {"lemma": "under", "forms": {}, "freq": 4.5, "aoa": 2.5, "domains": ["SPACE"], "features": {"spatial": True, "vertical": True}},
    {"lemma": "above", "forms": {}, "freq": 4.0, "aoa": 3.5, "domains": ["SPACE"], "features": {"spatial": True, "vertical": True}},
    {"lemma": "below", "forms": {}, "freq": 3.8, "aoa": 4.0, "domains": ["SPACE"], "features": {"spatial": True, "vertical": True}},
    {"lemma": "between", "forms": {}, "freq": 4.2, "aoa": 3.5, "domains": ["SPACE"], "features": {"spatial": True}},
    {"lemma": "among", "forms": {}, "freq": 3.8, "aoa": 5.0, "domains": ["SPACE"], "features": {"spatial": True}},
    {"lemma": "through", "forms": {}, "freq": 4.5, "aoa": 3.0, "domains": ["SPACE"], "features": {"spatial": True, "path": True}},
    {"lemma": "across", "forms": {}, "freq": 4.0, "aoa": 3.5, "domains": ["SPACE"], "features": {"spatial": True, "path": True}},
    {"lemma": "along", "forms": {}, "freq": 4.0, "aoa": 4.0, "domains": ["SPACE"], "features": {"spatial": True, "path": True}},
    {"lemma": "around", "forms": {}, "freq": 4.5, "aoa": 2.5, "domains": ["SPACE"], "features": {"spatial": True}},
    {"lemma": "behind", "forms": {}, "freq": 4.0, "aoa": 3.0, "domains": ["SPACE"], "features": {"spatial": True}},
    {"lemma": "beside", "forms": {}, "freq": 3.5, "aoa": 3.5, "domains": ["SPACE"], "features": {"spatial": True}},
    {"lemma": "near", "forms": {}, "freq": 4.2, "aoa": 3.0, "domains": ["SPACE"], "features": {"spatial": True, "proximity": True}},
    {"lemma": "by", "forms": {}, "freq": 5.5, "aoa": 2.5, "domains": ["SPACE"], "features": {"spatial": True, "proximity": True}},
    {"lemma": "off", "forms": {}, "freq": 4.8, "aoa": 2.0, "domains": ["SPACE"], "features": {"spatial": True, "separation": True}},
    {"lemma": "away", "forms": {}, "freq": 4.8, "aoa": 2.0, "domains": ["SPACE"], "features": {"spatial": True, "separation": True}},
    {"lemma": "toward", "forms": {}, "freq": 4.0, "aoa": 4.0, "domains": ["SPACE"], "features": {"spatial": True, "direction": True}},
    {"lemma": "towards", "forms": {}, "freq": 4.0, "aoa": 4.0, "domains": ["SPACE"], "features": {"spatial": True, "direction": True}},
    {"lemma": "inside", "forms": {}, "freq": 4.0, "aoa": 2.5, "domains": ["SPACE"], "features": {"spatial": True, "containment": True}},
    {"lemma": "outside", "forms": {}, "freq": 4.0, "aoa": 2.5, "domains": ["SPACE"], "features": {"spatial": True, "containment": True}},
    {"lemma": "within", "forms": {}, "freq": 4.0, "aoa": 5.0, "domains": ["SPACE"], "features": {"spatial": True, "containment": True}},
    {"lemma": "beyond", "forms": {}, "freq": 3.8, "aoa": 5.0, "domains": ["SPACE"], "features": {"spatial": True}},
    {"lemma": "past", "forms": {}, "freq": 4.2, "aoa": 4.0, "domains": ["SPACE"], "features": {"spatial": True, "path": True}},
]

PREPOSITIONS_TEMPORAL = [
    {"lemma": "before", "forms": {}, "freq": 4.8, "aoa": 3.0, "domains": ["TIME"], "features": {"temporal": True, "sequence": True}},
    {"lemma": "after", "forms": {}, "freq": 4.8, "aoa": 3.0, "domains": ["TIME"], "features": {"temporal": True, "sequence": True}},
    {"lemma": "during", "forms": {}, "freq": 4.2, "aoa": 4.0, "domains": ["TIME"], "features": {"temporal": True, "duration": True}},
    {"lemma": "until", "forms": {}, "freq": 4.2, "aoa": 4.0, "domains": ["TIME"], "features": {"temporal": True, "endpoint": True}},
    {"lemma": "since", "forms": {}, "freq": 4.5, "aoa": 4.5, "domains": ["TIME"], "features": {"temporal": True, "startpoint": True}},
    {"lemma": "for", "forms": {}, "freq": 5.8, "aoa": 2.0, "domains": ["TIME"], "features": {"temporal": True, "duration": True}},
    {"lemma": "by", "forms": {}, "freq": 5.5, "aoa": 2.5, "domains": ["TIME"], "features": {"temporal": True, "deadline": True}},
]

PREPOSITIONS_ABSTRACT = [
    {"lemma": "of", "forms": {}, "freq": 6.3, "aoa": 2.0, "domains": ["FUNCTION_WORD"], "features": {"abstract": True, "possession": True}},
    {"lemma": "with", "forms": {}, "freq": 5.8, "aoa": 2.0, "domains": ["FUNCTION_WORD"], "features": {"abstract": True, "accompaniment": True}},
    {"lemma": "without", "forms": {}, "freq": 4.5, "aoa": 3.5, "domains": ["FUNCTION_WORD"], "features": {"abstract": True, "accompaniment": True, "negative": True}},
    {"lemma": "for", "forms": {}, "freq": 5.8, "aoa": 2.0, "domains": ["FUNCTION_WORD"], "features": {"abstract": True, "beneficiary": True}},
    {"lemma": "about", "forms": {}, "freq": 5.2, "aoa": 3.0, "domains": ["FUNCTION_WORD"], "features": {"abstract": True, "topic": True}},
    {"lemma": "like", "forms": {}, "freq": 5.5, "aoa": 2.5, "domains": ["FUNCTION_WORD"], "features": {"abstract": True, "similarity": True}},
    {"lemma": "as", "forms": {}, "freq": 5.5, "aoa": 3.0, "domains": ["FUNCTION_WORD"], "features": {"abstract": True, "role": True}},
    {"lemma": "against", "forms": {}, "freq": 4.2, "aoa": 4.0, "domains": ["FUNCTION_WORD"], "features": {"abstract": True, "opposition": True}},
    {"lemma": "except", "forms": {}, "freq": 3.8, "aoa": 5.0, "domains": ["FUNCTION_WORD"], "features": {"abstract": True, "exception": True}},
    {"lemma": "despite", "forms": {}, "freq": 3.5, "aoa": 6.0, "domains": ["FUNCTION_WORD"], "features": {"abstract": True, "concession": True}},
    {"lemma": "according", "forms": {}, "freq": 3.8, "aoa": 6.0, "domains": ["FUNCTION_WORD"], "features": {"abstract": True, "source": True}},
    {"lemma": "because", "forms": {}, "freq": 5.0, "aoa": 3.0, "domains": ["FUNCTION_WORD"], "features": {"abstract": True, "cause": True}},
    {"lemma": "per", "forms": {}, "freq": 3.5, "aoa": 6.0, "domains": ["FUNCTION_WORD"], "features": {"abstract": True, "rate": True}},
    {"lemma": "via", "forms": {}, "freq": 3.2, "aoa": 8.0, "domains": ["FUNCTION_WORD"], "features": {"abstract": True, "means": True}},
    {"lemma": "upon", "forms": {}, "freq": 4.0, "aoa": 5.0, "domains": ["FUNCTION_WORD"], "features": {"abstract": True}},
    {"lemma": "regarding", "forms": {}, "freq": 3.2, "aoa": 8.0, "domains": ["FUNCTION_WORD"], "features": {"abstract": True, "topic": True}},
    {"lemma": "concerning", "forms": {}, "freq": 3.0, "aoa": 8.0, "domains": ["FUNCTION_WORD"], "features": {"abstract": True, "topic": True}},
    {"lemma": "including", "forms": {}, "freq": 4.0, "aoa": 6.0, "domains": ["FUNCTION_WORD"], "features": {"abstract": True, "inclusion": True}},
]

# Combine all prepositions
PREPOSITIONS = (
    PREPOSITIONS_SPATIAL + PREPOSITIONS_TEMPORAL + PREPOSITIONS_ABSTRACT
)

