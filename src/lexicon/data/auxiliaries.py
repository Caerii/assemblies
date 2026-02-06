"""Auxiliaries and Modals"""

AUXILIARIES_PRIMARY = [
    {"lemma": "be", "forms": {"1sg": "am", "3sg": "is", "pl": "are", "past_1sg": "was", "past_3sg": "was", "past_pl": "were", "ppart": "been", "prog": "being"}, "freq": 6.5, "aoa": 1.0, "domains": ["FUNCTION_WORD"], "features": {"auxiliary": True, "primary": True}},
    {"lemma": "have", "forms": {"3sg": "has", "past": "had", "ppart": "had", "prog": "having"}, "freq": 6.0, "aoa": 1.5, "domains": ["FUNCTION_WORD"], "features": {"auxiliary": True, "primary": True, "perfect": True}},
    {"lemma": "do", "forms": {"3sg": "does", "past": "did", "ppart": "done", "prog": "doing"}, "freq": 6.0, "aoa": 1.5, "domains": ["FUNCTION_WORD"], "features": {"auxiliary": True, "primary": True, "emphasis": True}},
]

AUXILIARIES_MODAL = [
    {"lemma": "can", "forms": {"past": "could", "neg": "cannot"}, "freq": 5.5, "aoa": 2.0, "domains": ["FUNCTION_WORD"], "features": {"modal": True, "ability": True, "permission": True}},
    {"lemma": "could", "forms": {"neg": "couldn't"}, "freq": 5.2, "aoa": 2.5, "domains": ["FUNCTION_WORD"], "features": {"modal": True, "ability": True, "possibility": True, "past": True}},
    {"lemma": "will", "forms": {"past": "would", "neg": "won't"}, "freq": 5.8, "aoa": 2.0, "domains": ["FUNCTION_WORD"], "features": {"modal": True, "future": True, "volition": True}},
    {"lemma": "would", "forms": {"neg": "wouldn't"}, "freq": 5.5, "aoa": 3.0, "domains": ["FUNCTION_WORD"], "features": {"modal": True, "conditional": True, "past": True}},
    {"lemma": "shall", "forms": {"past": "should", "neg": "shan't"}, "freq": 4.0, "aoa": 4.0, "domains": ["FUNCTION_WORD"], "features": {"modal": True, "future": True, "obligation": True}},
    {"lemma": "should", "forms": {"neg": "shouldn't"}, "freq": 5.0, "aoa": 3.5, "domains": ["FUNCTION_WORD"], "features": {"modal": True, "obligation": True, "advice": True}},
    {"lemma": "may", "forms": {"past": "might", "neg": "may not"}, "freq": 4.8, "aoa": 3.0, "domains": ["FUNCTION_WORD"], "features": {"modal": True, "permission": True, "possibility": True}},
    {"lemma": "might", "forms": {"neg": "might not"}, "freq": 4.5, "aoa": 3.5, "domains": ["FUNCTION_WORD"], "features": {"modal": True, "possibility": True, "past": True}},
    {"lemma": "must", "forms": {"neg": "mustn't"}, "freq": 4.8, "aoa": 3.0, "domains": ["FUNCTION_WORD"], "features": {"modal": True, "obligation": True, "necessity": True}},
    {"lemma": "need", "forms": {"3sg": "needs", "past": "needed", "neg": "needn't"}, "freq": 5.2, "aoa": 2.5, "domains": ["FUNCTION_WORD"], "features": {"semi_modal": True, "necessity": True}},
    {"lemma": "dare", "forms": {"3sg": "dares", "past": "dared", "neg": "daren't"}, "freq": 3.5, "aoa": 4.0, "domains": ["FUNCTION_WORD"], "features": {"semi_modal": True, "courage": True}},
    {"lemma": "ought", "forms": {"neg": "ought not"}, "freq": 3.5, "aoa": 5.0, "domains": ["FUNCTION_WORD"], "features": {"modal": True, "obligation": True}},
]

AUXILIARIES_SEMI = [
    {"lemma": "going to", "forms": {"1sg": "am going to", "3sg": "is going to", "pl": "are going to"}, "freq": 5.0, "aoa": 2.5, "domains": ["FUNCTION_WORD"], "features": {"semi_modal": True, "future": True}},
    {"lemma": "have to", "forms": {"3sg": "has to", "past": "had to"}, "freq": 5.0, "aoa": 2.5, "domains": ["FUNCTION_WORD"], "features": {"semi_modal": True, "obligation": True}},
    {"lemma": "used to", "forms": {}, "freq": 4.5, "aoa": 4.0, "domains": ["FUNCTION_WORD"], "features": {"semi_modal": True, "habitual": True, "past": True}},
    {"lemma": "be able to", "forms": {"1sg": "am able to", "3sg": "is able to", "pl": "are able to", "past": "was able to"}, "freq": 4.2, "aoa": 4.0, "domains": ["FUNCTION_WORD"], "features": {"semi_modal": True, "ability": True}},
    {"lemma": "be about to", "forms": {"1sg": "am about to", "3sg": "is about to", "pl": "are about to"}, "freq": 3.8, "aoa": 5.0, "domains": ["FUNCTION_WORD"], "features": {"semi_modal": True, "imminent": True}},
    {"lemma": "be supposed to", "forms": {"1sg": "am supposed to", "3sg": "is supposed to", "pl": "are supposed to"}, "freq": 4.0, "aoa": 5.0, "domains": ["FUNCTION_WORD"], "features": {"semi_modal": True, "expectation": True}},
]

# Combine all auxiliaries
AUXILIARIES = (
    AUXILIARIES_PRIMARY + AUXILIARIES_MODAL + AUXILIARIES_SEMI
)

