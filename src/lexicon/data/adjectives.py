"""Adjectives - Organized by semantic type"""

# Format: {"lemma", "forms": {"comp": ..., "super": ...}, "freq": log_freq, "aoa": age, "domains": [...], "features": {...}}

ADJECTIVES_SIZE = [
    {"lemma": "big", "forms": {"comp": "bigger", "super": "biggest"}, "freq": 5.0, "aoa": 1.5, "domains": ["QUALITY"], "features": {"dimension": True, "positive": True}},
    {"lemma": "small", "forms": {"comp": "smaller", "super": "smallest"}, "freq": 4.5, "aoa": 1.5, "domains": ["QUALITY"], "features": {"dimension": True, "negative": True}},
    {"lemma": "large", "forms": {"comp": "larger", "super": "largest"}, "freq": 4.5, "aoa": 3.0, "domains": ["QUALITY"], "features": {"dimension": True, "positive": True}},
    {"lemma": "little", "forms": {"comp": "littler", "super": "littlest"}, "freq": 5.0, "aoa": 1.5, "domains": ["QUALITY"], "features": {"dimension": True, "negative": True}},
    {"lemma": "tiny", "forms": {"comp": "tinier", "super": "tiniest"}, "freq": 3.5, "aoa": 3.0, "domains": ["QUALITY"], "features": {"dimension": True, "negative": True}},
    {"lemma": "huge", "forms": {"comp": "huger", "super": "hugest"}, "freq": 4.0, "aoa": 3.5, "domains": ["QUALITY"], "features": {"dimension": True, "positive": True}},
    {"lemma": "tall", "forms": {"comp": "taller", "super": "tallest"}, "freq": 4.0, "aoa": 2.5, "domains": ["QUALITY"], "features": {"dimension": True, "vertical": True}},
    {"lemma": "short", "forms": {"comp": "shorter", "super": "shortest"}, "freq": 4.2, "aoa": 2.5, "domains": ["QUALITY"], "features": {"dimension": True, "vertical": True}},
    {"lemma": "long", "forms": {"comp": "longer", "super": "longest"}, "freq": 4.8, "aoa": 2.5, "domains": ["QUALITY"], "features": {"dimension": True}},
    {"lemma": "wide", "forms": {"comp": "wider", "super": "widest"}, "freq": 3.8, "aoa": 4.0, "domains": ["QUALITY"], "features": {"dimension": True}},
    {"lemma": "narrow", "forms": {"comp": "narrower", "super": "narrowest"}, "freq": 3.2, "aoa": 5.0, "domains": ["QUALITY"], "features": {"dimension": True}},
    {"lemma": "thick", "forms": {"comp": "thicker", "super": "thickest"}, "freq": 3.5, "aoa": 4.0, "domains": ["QUALITY"], "features": {"dimension": True}},
    {"lemma": "thin", "forms": {"comp": "thinner", "super": "thinnest"}, "freq": 3.5, "aoa": 4.0, "domains": ["QUALITY"], "features": {"dimension": True}},
    {"lemma": "deep", "forms": {"comp": "deeper", "super": "deepest"}, "freq": 4.0, "aoa": 4.0, "domains": ["QUALITY"], "features": {"dimension": True}},
    {"lemma": "shallow", "forms": {"comp": "shallower", "super": "shallowest"}, "freq": 2.8, "aoa": 5.0, "domains": ["QUALITY"], "features": {"dimension": True}},
]

ADJECTIVES_COLOR = [
    {"lemma": "red", "forms": {}, "freq": 4.0, "aoa": 2.0, "domains": ["QUALITY"], "features": {"color": True}},
    {"lemma": "blue", "forms": {}, "freq": 4.0, "aoa": 2.0, "domains": ["QUALITY"], "features": {"color": True}},
    {"lemma": "green", "forms": {}, "freq": 3.8, "aoa": 2.0, "domains": ["QUALITY"], "features": {"color": True}},
    {"lemma": "yellow", "forms": {}, "freq": 3.5, "aoa": 2.0, "domains": ["QUALITY"], "features": {"color": True}},
    {"lemma": "black", "forms": {}, "freq": 4.2, "aoa": 2.0, "domains": ["QUALITY"], "features": {"color": True}},
    {"lemma": "white", "forms": {}, "freq": 4.2, "aoa": 2.0, "domains": ["QUALITY"], "features": {"color": True}},
    {"lemma": "orange", "forms": {}, "freq": 3.2, "aoa": 2.5, "domains": ["QUALITY"], "features": {"color": True}},
    {"lemma": "purple", "forms": {}, "freq": 3.0, "aoa": 2.5, "domains": ["QUALITY"], "features": {"color": True}},
    {"lemma": "pink", "forms": {}, "freq": 3.2, "aoa": 2.5, "domains": ["QUALITY"], "features": {"color": True}},
    {"lemma": "brown", "forms": {}, "freq": 3.5, "aoa": 2.5, "domains": ["QUALITY"], "features": {"color": True}},
    {"lemma": "gray", "forms": {}, "freq": 3.2, "aoa": 3.0, "domains": ["QUALITY"], "features": {"color": True}},
    {"lemma": "golden", "forms": {}, "freq": 3.0, "aoa": 4.0, "domains": ["QUALITY"], "features": {"color": True}},
    {"lemma": "silver", "forms": {}, "freq": 3.0, "aoa": 4.0, "domains": ["QUALITY"], "features": {"color": True}},
    {"lemma": "dark", "forms": {"comp": "darker", "super": "darkest"}, "freq": 4.2, "aoa": 2.5, "domains": ["QUALITY"], "features": {"light": True}},
    {"lemma": "light", "forms": {"comp": "lighter", "super": "lightest"}, "freq": 4.5, "aoa": 2.5, "domains": ["QUALITY"], "features": {"light": True}},
    {"lemma": "bright", "forms": {"comp": "brighter", "super": "brightest"}, "freq": 3.8, "aoa": 3.0, "domains": ["QUALITY"], "features": {"light": True}},
]

ADJECTIVES_PHYSICAL = [
    {"lemma": "hot", "forms": {"comp": "hotter", "super": "hottest"}, "freq": 4.2, "aoa": 2.0, "domains": ["QUALITY"], "features": {"temperature": True}},
    {"lemma": "cold", "forms": {"comp": "colder", "super": "coldest"}, "freq": 4.2, "aoa": 2.0, "domains": ["QUALITY"], "features": {"temperature": True}},
    {"lemma": "warm", "forms": {"comp": "warmer", "super": "warmest"}, "freq": 4.0, "aoa": 2.5, "domains": ["QUALITY"], "features": {"temperature": True}},
    {"lemma": "cool", "forms": {"comp": "cooler", "super": "coolest"}, "freq": 4.0, "aoa": 3.0, "domains": ["QUALITY"], "features": {"temperature": True}},
    {"lemma": "hard", "forms": {"comp": "harder", "super": "hardest"}, "freq": 4.5, "aoa": 2.5, "domains": ["QUALITY"], "features": {"texture": True}},
    {"lemma": "soft", "forms": {"comp": "softer", "super": "softest"}, "freq": 3.8, "aoa": 2.5, "domains": ["QUALITY"], "features": {"texture": True}},
    {"lemma": "wet", "forms": {"comp": "wetter", "super": "wettest"}, "freq": 3.5, "aoa": 2.5, "domains": ["QUALITY"], "features": {"moisture": True}},
    {"lemma": "dry", "forms": {"comp": "drier", "super": "driest"}, "freq": 3.8, "aoa": 2.5, "domains": ["QUALITY"], "features": {"moisture": True}},
    {"lemma": "heavy", "forms": {"comp": "heavier", "super": "heaviest"}, "freq": 4.0, "aoa": 3.0, "domains": ["QUALITY"], "features": {"weight": True}},
    {"lemma": "light", "forms": {"comp": "lighter", "super": "lightest"}, "freq": 4.5, "aoa": 2.5, "domains": ["QUALITY"], "features": {"weight": True}},
    {"lemma": "fast", "forms": {"comp": "faster", "super": "fastest"}, "freq": 4.2, "aoa": 2.5, "domains": ["QUALITY"], "features": {"speed": True}},
    {"lemma": "slow", "forms": {"comp": "slower", "super": "slowest"}, "freq": 4.0, "aoa": 2.5, "domains": ["QUALITY"], "features": {"speed": True}},
    {"lemma": "quick", "forms": {"comp": "quicker", "super": "quickest"}, "freq": 4.0, "aoa": 3.0, "domains": ["QUALITY"], "features": {"speed": True}},
    {"lemma": "loud", "forms": {"comp": "louder", "super": "loudest"}, "freq": 3.5, "aoa": 3.0, "domains": ["QUALITY"], "features": {"sound": True}},
    {"lemma": "quiet", "forms": {"comp": "quieter", "super": "quietest"}, "freq": 3.8, "aoa": 3.0, "domains": ["QUALITY"], "features": {"sound": True}},
    {"lemma": "sharp", "forms": {"comp": "sharper", "super": "sharpest"}, "freq": 3.5, "aoa": 3.5, "domains": ["QUALITY"], "features": {"edge": True}},
    {"lemma": "smooth", "forms": {"comp": "smoother", "super": "smoothest"}, "freq": 3.5, "aoa": 4.0, "domains": ["QUALITY"], "features": {"texture": True}},
    {"lemma": "rough", "forms": {"comp": "rougher", "super": "roughest"}, "freq": 3.2, "aoa": 4.0, "domains": ["QUALITY"], "features": {"texture": True}},
    {"lemma": "clean", "forms": {"comp": "cleaner", "super": "cleanest"}, "freq": 4.0, "aoa": 2.5, "domains": ["QUALITY"], "features": {"cleanliness": True}},
    {"lemma": "dirty", "forms": {"comp": "dirtier", "super": "dirtiest"}, "freq": 3.5, "aoa": 2.5, "domains": ["QUALITY"], "features": {"cleanliness": True}},
    {"lemma": "full", "forms": {"comp": "fuller", "super": "fullest"}, "freq": 4.2, "aoa": 2.5, "domains": ["QUALITY"], "features": {"amount": True}},
    {"lemma": "empty", "forms": {"comp": "emptier", "super": "emptiest"}, "freq": 3.8, "aoa": 3.0, "domains": ["QUALITY"], "features": {"amount": True}},
]

ADJECTIVES_EVALUATIVE = [
    {"lemma": "good", "forms": {"comp": "better", "super": "best"}, "freq": 5.5, "aoa": 1.5, "domains": ["QUALITY"], "features": {"evaluative": True, "positive": True}},
    {"lemma": "bad", "forms": {"comp": "worse", "super": "worst"}, "freq": 5.0, "aoa": 2.0, "domains": ["QUALITY"], "features": {"evaluative": True, "negative": True}},
    {"lemma": "nice", "forms": {"comp": "nicer", "super": "nicest"}, "freq": 4.5, "aoa": 2.0, "domains": ["QUALITY"], "features": {"evaluative": True, "positive": True}},
    {"lemma": "great", "forms": {"comp": "greater", "super": "greatest"}, "freq": 5.0, "aoa": 2.5, "domains": ["QUALITY"], "features": {"evaluative": True, "positive": True}},
    {"lemma": "beautiful", "forms": {"comp": "more beautiful", "super": "most beautiful"}, "freq": 4.0, "aoa": 3.0, "domains": ["QUALITY"], "features": {"evaluative": True, "positive": True}},
    {"lemma": "ugly", "forms": {"comp": "uglier", "super": "ugliest"}, "freq": 3.2, "aoa": 3.5, "domains": ["QUALITY"], "features": {"evaluative": True, "negative": True}},
    {"lemma": "pretty", "forms": {"comp": "prettier", "super": "prettiest"}, "freq": 4.0, "aoa": 2.5, "domains": ["QUALITY"], "features": {"evaluative": True, "positive": True}},
    {"lemma": "wonderful", "forms": {"comp": "more wonderful", "super": "most wonderful"}, "freq": 3.8, "aoa": 4.0, "domains": ["QUALITY"], "features": {"evaluative": True, "positive": True}},
    {"lemma": "terrible", "forms": {"comp": "more terrible", "super": "most terrible"}, "freq": 3.5, "aoa": 4.0, "domains": ["QUALITY"], "features": {"evaluative": True, "negative": True}},
    {"lemma": "perfect", "forms": {"comp": "more perfect", "super": "most perfect"}, "freq": 4.0, "aoa": 4.0, "domains": ["QUALITY"], "features": {"evaluative": True, "positive": True}},
    {"lemma": "excellent", "forms": {"comp": "more excellent", "super": "most excellent"}, "freq": 3.8, "aoa": 5.0, "domains": ["QUALITY"], "features": {"evaluative": True, "positive": True}},
    {"lemma": "awful", "forms": {"comp": "more awful", "super": "most awful"}, "freq": 3.5, "aoa": 4.0, "domains": ["QUALITY"], "features": {"evaluative": True, "negative": True}},
    {"lemma": "fine", "forms": {"comp": "finer", "super": "finest"}, "freq": 4.5, "aoa": 3.0, "domains": ["QUALITY"], "features": {"evaluative": True, "positive": True}},
    {"lemma": "okay", "forms": {}, "freq": 4.5, "aoa": 3.0, "domains": ["QUALITY"], "features": {"evaluative": True, "neutral": True}},
]

ADJECTIVES_EMOTION = [
    {"lemma": "happy", "forms": {"comp": "happier", "super": "happiest"}, "freq": 4.5, "aoa": 2.0, "domains": ["EMOTION"], "features": {"emotion": True, "positive": True}},
    {"lemma": "sad", "forms": {"comp": "sadder", "super": "saddest"}, "freq": 4.0, "aoa": 2.0, "domains": ["EMOTION"], "features": {"emotion": True, "negative": True}},
    {"lemma": "angry", "forms": {"comp": "angrier", "super": "angriest"}, "freq": 4.0, "aoa": 2.5, "domains": ["EMOTION"], "features": {"emotion": True, "negative": True}},
    {"lemma": "scared", "forms": {"comp": "more scared", "super": "most scared"}, "freq": 3.8, "aoa": 2.5, "domains": ["EMOTION"], "features": {"emotion": True, "negative": True}},
    {"lemma": "afraid", "forms": {"comp": "more afraid", "super": "most afraid"}, "freq": 4.0, "aoa": 3.0, "domains": ["EMOTION"], "features": {"emotion": True, "negative": True}},
    {"lemma": "tired", "forms": {"comp": "more tired", "super": "most tired"}, "freq": 4.0, "aoa": 2.5, "domains": ["EMOTION"], "features": {"emotion": True, "negative": True}},
    {"lemma": "hungry", "forms": {"comp": "hungrier", "super": "hungriest"}, "freq": 3.5, "aoa": 2.0, "domains": ["EMOTION"], "features": {"emotion": True, "physical": True}},
    {"lemma": "thirsty", "forms": {"comp": "thirstier", "super": "thirstiest"}, "freq": 3.0, "aoa": 2.5, "domains": ["EMOTION"], "features": {"emotion": True, "physical": True}},
    {"lemma": "excited", "forms": {"comp": "more excited", "super": "most excited"}, "freq": 3.8, "aoa": 3.0, "domains": ["EMOTION"], "features": {"emotion": True, "positive": True}},
    {"lemma": "bored", "forms": {"comp": "more bored", "super": "most bored"}, "freq": 3.5, "aoa": 3.5, "domains": ["EMOTION"], "features": {"emotion": True, "negative": True}},
    {"lemma": "surprised", "forms": {"comp": "more surprised", "super": "most surprised"}, "freq": 3.8, "aoa": 3.5, "domains": ["EMOTION"], "features": {"emotion": True}},
    {"lemma": "worried", "forms": {"comp": "more worried", "super": "most worried"}, "freq": 3.8, "aoa": 4.0, "domains": ["EMOTION"], "features": {"emotion": True, "negative": True}},
    {"lemma": "proud", "forms": {"comp": "prouder", "super": "proudest"}, "freq": 3.8, "aoa": 4.0, "domains": ["EMOTION"], "features": {"emotion": True, "positive": True}},
    {"lemma": "lonely", "forms": {"comp": "lonelier", "super": "loneliest"}, "freq": 3.5, "aoa": 4.0, "domains": ["EMOTION"], "features": {"emotion": True, "negative": True}},
    {"lemma": "calm", "forms": {"comp": "calmer", "super": "calmest"}, "freq": 3.5, "aoa": 4.0, "domains": ["EMOTION"], "features": {"emotion": True, "positive": True}},
    {"lemma": "nervous", "forms": {"comp": "more nervous", "super": "most nervous"}, "freq": 3.5, "aoa": 4.5, "domains": ["EMOTION"], "features": {"emotion": True, "negative": True}},
]

ADJECTIVES_AGE = [
    {"lemma": "old", "forms": {"comp": "older", "super": "oldest"}, "freq": 5.0, "aoa": 2.0, "domains": ["QUALITY"], "features": {"age": True}},
    {"lemma": "young", "forms": {"comp": "younger", "super": "youngest"}, "freq": 4.5, "aoa": 2.5, "domains": ["QUALITY"], "features": {"age": True}},
    {"lemma": "new", "forms": {"comp": "newer", "super": "newest"}, "freq": 5.2, "aoa": 2.0, "domains": ["QUALITY"], "features": {"age": True}},
    {"lemma": "ancient", "forms": {"comp": "more ancient", "super": "most ancient"}, "freq": 3.2, "aoa": 6.0, "domains": ["QUALITY"], "features": {"age": True}},
    {"lemma": "modern", "forms": {"comp": "more modern", "super": "most modern"}, "freq": 3.8, "aoa": 6.0, "domains": ["QUALITY"], "features": {"age": True}},
    {"lemma": "fresh", "forms": {"comp": "fresher", "super": "freshest"}, "freq": 3.8, "aoa": 3.5, "domains": ["QUALITY"], "features": {"age": True}},
]

ADJECTIVES_QUANTITY = [
    {"lemma": "many", "forms": {"comp": "more", "super": "most"}, "freq": 5.0, "aoa": 2.5, "domains": ["QUANTITY"], "features": {"quantity": True, "count": True}},
    {"lemma": "few", "forms": {"comp": "fewer", "super": "fewest"}, "freq": 4.2, "aoa": 3.5, "domains": ["QUANTITY"], "features": {"quantity": True, "count": True}},
    {"lemma": "much", "forms": {"comp": "more", "super": "most"}, "freq": 4.8, "aoa": 2.5, "domains": ["QUANTITY"], "features": {"quantity": True, "mass": True}},
    {"lemma": "little", "forms": {"comp": "less", "super": "least"}, "freq": 5.0, "aoa": 1.5, "domains": ["QUANTITY"], "features": {"quantity": True, "mass": True}},
    {"lemma": "enough", "forms": {}, "freq": 4.5, "aoa": 3.5, "domains": ["QUANTITY"], "features": {"quantity": True}},
    {"lemma": "extra", "forms": {}, "freq": 3.8, "aoa": 4.0, "domains": ["QUANTITY"], "features": {"quantity": True}},
]

ADJECTIVES_PERSONALITY = [
    {"lemma": "kind", "forms": {"comp": "kinder", "super": "kindest"}, "freq": 4.0, "aoa": 3.0, "domains": ["SOCIAL"], "features": {"personality": True, "positive": True}},
    {"lemma": "mean", "forms": {"comp": "meaner", "super": "meanest"}, "freq": 4.0, "aoa": 3.0, "domains": ["SOCIAL"], "features": {"personality": True, "negative": True}},
    {"lemma": "smart", "forms": {"comp": "smarter", "super": "smartest"}, "freq": 4.0, "aoa": 3.0, "domains": ["COGNITION"], "features": {"personality": True, "positive": True}},
    {"lemma": "stupid", "forms": {"comp": "stupider", "super": "stupidest"}, "freq": 3.8, "aoa": 4.0, "domains": ["COGNITION"], "features": {"personality": True, "negative": True}},
    {"lemma": "clever", "forms": {"comp": "cleverer", "super": "cleverest"}, "freq": 3.5, "aoa": 4.0, "domains": ["COGNITION"], "features": {"personality": True, "positive": True}},
    {"lemma": "funny", "forms": {"comp": "funnier", "super": "funniest"}, "freq": 4.2, "aoa": 2.5, "domains": ["SOCIAL"], "features": {"personality": True, "positive": True}},
    {"lemma": "serious", "forms": {"comp": "more serious", "super": "most serious"}, "freq": 4.0, "aoa": 4.0, "domains": ["SOCIAL"], "features": {"personality": True}},
    {"lemma": "friendly", "forms": {"comp": "friendlier", "super": "friendliest"}, "freq": 3.8, "aoa": 3.5, "domains": ["SOCIAL"], "features": {"personality": True, "positive": True}},
    {"lemma": "shy", "forms": {"comp": "shyer", "super": "shyest"}, "freq": 3.2, "aoa": 4.0, "domains": ["SOCIAL"], "features": {"personality": True}},
    {"lemma": "brave", "forms": {"comp": "braver", "super": "bravest"}, "freq": 3.5, "aoa": 4.0, "domains": ["SOCIAL"], "features": {"personality": True, "positive": True}},
    {"lemma": "lazy", "forms": {"comp": "lazier", "super": "laziest"}, "freq": 3.2, "aoa": 4.0, "domains": ["SOCIAL"], "features": {"personality": True, "negative": True}},
    {"lemma": "crazy", "forms": {"comp": "crazier", "super": "craziest"}, "freq": 4.0, "aoa": 3.5, "domains": ["COGNITION"], "features": {"personality": True}},
    {"lemma": "busy", "forms": {"comp": "busier", "super": "busiest"}, "freq": 4.0, "aoa": 3.5, "domains": ["QUALITY"], "features": {"activity": True}},
    {"lemma": "careful", "forms": {"comp": "more careful", "super": "most careful"}, "freq": 3.8, "aoa": 4.0, "domains": ["QUALITY"], "features": {"personality": True, "positive": True}},
    {"lemma": "careless", "forms": {"comp": "more careless", "super": "most careless"}, "freq": 3.0, "aoa": 5.0, "domains": ["QUALITY"], "features": {"personality": True, "negative": True}},
]

ADJECTIVES_OTHER = [
    {"lemma": "same", "forms": {}, "freq": 5.0, "aoa": 3.0, "domains": ["QUALITY"], "features": {"comparison": True}},
    {"lemma": "different", "forms": {"comp": "more different", "super": "most different"}, "freq": 4.5, "aoa": 3.5, "domains": ["QUALITY"], "features": {"comparison": True}},
    {"lemma": "other", "forms": {}, "freq": 5.2, "aoa": 2.5, "domains": ["QUALITY"], "features": {"comparison": True}},
    {"lemma": "next", "forms": {}, "freq": 5.0, "aoa": 3.0, "domains": ["QUALITY"], "features": {"order": True}},
    {"lemma": "last", "forms": {}, "freq": 5.0, "aoa": 3.0, "domains": ["QUALITY"], "features": {"order": True}},
    {"lemma": "first", "forms": {}, "freq": 5.2, "aoa": 2.5, "domains": ["QUALITY"], "features": {"order": True}},
    {"lemma": "only", "forms": {}, "freq": 5.2, "aoa": 3.0, "domains": ["QUALITY"], "features": {"exclusivity": True}},
    {"lemma": "own", "forms": {}, "freq": 5.0, "aoa": 3.0, "domains": ["QUALITY"], "features": {"possession": True}},
    {"lemma": "real", "forms": {"comp": "more real", "super": "most real"}, "freq": 4.8, "aoa": 3.5, "domains": ["QUALITY"], "features": {"reality": True}},
    {"lemma": "true", "forms": {"comp": "truer", "super": "truest"}, "freq": 4.8, "aoa": 3.5, "domains": ["QUALITY"], "features": {"truth": True}},
    {"lemma": "false", "forms": {"comp": "more false", "super": "most false"}, "freq": 3.8, "aoa": 5.0, "domains": ["QUALITY"], "features": {"truth": True}},
    {"lemma": "wrong", "forms": {"comp": "more wrong", "super": "most wrong"}, "freq": 4.5, "aoa": 3.0, "domains": ["QUALITY"], "features": {"correctness": True}},
    {"lemma": "right", "forms": {"comp": "more right", "super": "most right"}, "freq": 5.2, "aoa": 2.5, "domains": ["QUALITY"], "features": {"correctness": True}},
    {"lemma": "important", "forms": {"comp": "more important", "super": "most important"}, "freq": 4.5, "aoa": 4.5, "domains": ["QUALITY"], "features": {"importance": True}},
    {"lemma": "special", "forms": {"comp": "more special", "super": "most special"}, "freq": 4.2, "aoa": 3.5, "domains": ["QUALITY"], "features": {"uniqueness": True}},
    {"lemma": "strange", "forms": {"comp": "stranger", "super": "strangest"}, "freq": 4.0, "aoa": 4.0, "domains": ["QUALITY"], "features": {"unusualness": True}},
    {"lemma": "normal", "forms": {"comp": "more normal", "super": "most normal"}, "freq": 4.0, "aoa": 4.5, "domains": ["QUALITY"], "features": {"normalcy": True}},
    {"lemma": "usual", "forms": {"comp": "more usual", "super": "most usual"}, "freq": 4.0, "aoa": 5.0, "domains": ["QUALITY"], "features": {"normalcy": True}},
    {"lemma": "possible", "forms": {"comp": "more possible", "super": "most possible"}, "freq": 4.5, "aoa": 5.0, "domains": ["QUALITY"], "features": {"possibility": True}},
    {"lemma": "impossible", "forms": {"comp": "more impossible", "super": "most impossible"}, "freq": 3.8, "aoa": 5.0, "domains": ["QUALITY"], "features": {"possibility": True}},
    {"lemma": "easy", "forms": {"comp": "easier", "super": "easiest"}, "freq": 4.5, "aoa": 3.0, "domains": ["QUALITY"], "features": {"difficulty": True}},
    {"lemma": "difficult", "forms": {"comp": "more difficult", "super": "most difficult"}, "freq": 4.0, "aoa": 4.5, "domains": ["QUALITY"], "features": {"difficulty": True}},
    {"lemma": "hard", "forms": {"comp": "harder", "super": "hardest"}, "freq": 4.5, "aoa": 2.5, "domains": ["QUALITY"], "features": {"difficulty": True}},
    {"lemma": "simple", "forms": {"comp": "simpler", "super": "simplest"}, "freq": 4.0, "aoa": 4.0, "domains": ["QUALITY"], "features": {"complexity": True}},
    {"lemma": "free", "forms": {"comp": "freer", "super": "freest"}, "freq": 4.8, "aoa": 3.5, "domains": ["QUALITY"], "features": {"freedom": True}},
    {"lemma": "safe", "forms": {"comp": "safer", "super": "safest"}, "freq": 4.2, "aoa": 3.5, "domains": ["QUALITY"], "features": {"safety": True}},
    {"lemma": "dangerous", "forms": {"comp": "more dangerous", "super": "most dangerous"}, "freq": 3.8, "aoa": 4.0, "domains": ["QUALITY"], "features": {"safety": True}},
    {"lemma": "ready", "forms": {"comp": "readier", "super": "readiest"}, "freq": 4.5, "aoa": 3.0, "domains": ["QUALITY"], "features": {"preparedness": True}},
    {"lemma": "sure", "forms": {"comp": "surer", "super": "surest"}, "freq": 5.0, "aoa": 3.0, "domains": ["QUALITY"], "features": {"certainty": True}},
    {"lemma": "certain", "forms": {"comp": "more certain", "super": "most certain"}, "freq": 4.2, "aoa": 5.0, "domains": ["QUALITY"], "features": {"certainty": True}},
    {"lemma": "alive", "forms": {}, "freq": 4.2, "aoa": 3.0, "domains": ["QUALITY"], "features": {"life": True}},
    {"lemma": "dead", "forms": {}, "freq": 4.2, "aoa": 3.0, "domains": ["QUALITY"], "features": {"life": True}},
    {"lemma": "open", "forms": {"comp": "more open", "super": "most open"}, "freq": 4.5, "aoa": 2.5, "domains": ["QUALITY"], "features": {"state": True}},
    {"lemma": "closed", "forms": {"comp": "more closed", "super": "most closed"}, "freq": 4.0, "aoa": 2.5, "domains": ["QUALITY"], "features": {"state": True}},
    {"lemma": "near", "forms": {"comp": "nearer", "super": "nearest"}, "freq": 4.2, "aoa": 3.0, "domains": ["SPACE"], "features": {"distance": True}},
    {"lemma": "far", "forms": {"comp": "farther", "super": "farthest"}, "freq": 4.5, "aoa": 2.5, "domains": ["SPACE"], "features": {"distance": True}},
    {"lemma": "close", "forms": {"comp": "closer", "super": "closest"}, "freq": 4.5, "aoa": 3.0, "domains": ["SPACE"], "features": {"distance": True}},
]

# Combine all adjectives
ADJECTIVES = (
    ADJECTIVES_SIZE + ADJECTIVES_COLOR + ADJECTIVES_PHYSICAL +
    ADJECTIVES_EVALUATIVE + ADJECTIVES_EMOTION + ADJECTIVES_AGE +
    ADJECTIVES_QUANTITY + ADJECTIVES_PERSONALITY + ADJECTIVES_OTHER
)

