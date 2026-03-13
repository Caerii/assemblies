"""Nouns - Organized by semantic domain"""

# Format: {"lemma", "forms": {"plural": ...}, "freq": log_freq, "aoa": age, "domains": [...], "features": {...}}

NOUNS_PEOPLE = [
    {"lemma": "man", "forms": {"plural": "men"}, "freq": 5.2, "aoa": 2.0, "domains": ["PERSON"], "features": {"animate": True, "human": True, "gender": "m"}},
    {"lemma": "woman", "forms": {"plural": "women"}, "freq": 5.0, "aoa": 2.0, "domains": ["PERSON"], "features": {"animate": True, "human": True, "gender": "f"}},
    {"lemma": "child", "forms": {"plural": "children"}, "freq": 4.8, "aoa": 2.0, "domains": ["PERSON"], "features": {"animate": True, "human": True, "young": True}},
    {"lemma": "boy", "forms": {"plural": "boys"}, "freq": 4.5, "aoa": 2.0, "domains": ["PERSON"], "features": {"animate": True, "human": True, "gender": "m", "young": True}},
    {"lemma": "girl", "forms": {"plural": "girls"}, "freq": 4.5, "aoa": 2.0, "domains": ["PERSON"], "features": {"animate": True, "human": True, "gender": "f", "young": True}},
    {"lemma": "baby", "forms": {"plural": "babies"}, "freq": 4.2, "aoa": 1.5, "domains": ["PERSON"], "features": {"animate": True, "human": True, "young": True}},
    {"lemma": "mother", "forms": {"plural": "mothers"}, "freq": 4.5, "aoa": 1.5, "domains": ["PERSON"], "features": {"animate": True, "human": True, "gender": "f", "family": True}},
    {"lemma": "father", "forms": {"plural": "fathers"}, "freq": 4.3, "aoa": 1.5, "domains": ["PERSON"], "features": {"animate": True, "human": True, "gender": "m", "family": True}},
    {"lemma": "mom", "forms": {"plural": "moms"}, "freq": 4.8, "aoa": 1.2, "domains": ["PERSON"], "features": {"animate": True, "human": True, "gender": "f", "family": True}},
    {"lemma": "dad", "forms": {"plural": "dads"}, "freq": 4.6, "aoa": 1.2, "domains": ["PERSON"], "features": {"animate": True, "human": True, "gender": "m", "family": True}},
    {"lemma": "sister", "forms": {"plural": "sisters"}, "freq": 4.0, "aoa": 2.5, "domains": ["PERSON"], "features": {"animate": True, "human": True, "gender": "f", "family": True}},
    {"lemma": "brother", "forms": {"plural": "brothers"}, "freq": 4.0, "aoa": 2.5, "domains": ["PERSON"], "features": {"animate": True, "human": True, "gender": "m", "family": True}},
    {"lemma": "friend", "forms": {"plural": "friends"}, "freq": 4.5, "aoa": 2.5, "domains": ["PERSON", "SOCIAL"], "features": {"animate": True, "human": True}},
    {"lemma": "person", "forms": {"plural": "people"}, "freq": 4.8, "aoa": 3.0, "domains": ["PERSON"], "features": {"animate": True, "human": True}},
    {"lemma": "teacher", "forms": {"plural": "teachers"}, "freq": 4.0, "aoa": 3.5, "domains": ["PERSON"], "features": {"animate": True, "human": True, "profession": True}},
    {"lemma": "doctor", "forms": {"plural": "doctors"}, "freq": 4.0, "aoa": 3.5, "domains": ["PERSON"], "features": {"animate": True, "human": True, "profession": True}},
    {"lemma": "king", "forms": {"plural": "kings"}, "freq": 3.8, "aoa": 4.0, "domains": ["PERSON"], "features": {"animate": True, "human": True, "gender": "m", "royalty": True}},
    {"lemma": "queen", "forms": {"plural": "queens"}, "freq": 3.5, "aoa": 4.0, "domains": ["PERSON"], "features": {"animate": True, "human": True, "gender": "f", "royalty": True}},
]

NOUNS_ANIMALS = [
    {"lemma": "dog", "forms": {"plural": "dogs"}, "freq": 4.5, "aoa": 1.5, "domains": ["ANIMAL"], "features": {"animate": True, "animal": True, "pet": True}},
    {"lemma": "cat", "forms": {"plural": "cats"}, "freq": 4.3, "aoa": 1.5, "domains": ["ANIMAL"], "features": {"animate": True, "animal": True, "pet": True}},
    {"lemma": "bird", "forms": {"plural": "birds"}, "freq": 4.0, "aoa": 2.0, "domains": ["ANIMAL"], "features": {"animate": True, "animal": True, "flies": True}},
    {"lemma": "fish", "forms": {"plural": "fish"}, "freq": 4.0, "aoa": 2.0, "domains": ["ANIMAL"], "features": {"animate": True, "animal": True, "swims": True}},
    {"lemma": "mouse", "forms": {"plural": "mice"}, "freq": 3.5, "aoa": 2.5, "domains": ["ANIMAL"], "features": {"animate": True, "animal": True, "small": True}},
    {"lemma": "horse", "forms": {"plural": "horses"}, "freq": 3.8, "aoa": 2.5, "domains": ["ANIMAL"], "features": {"animate": True, "animal": True, "large": True}},
    {"lemma": "cow", "forms": {"plural": "cows"}, "freq": 3.5, "aoa": 2.5, "domains": ["ANIMAL"], "features": {"animate": True, "animal": True, "farm": True}},
    {"lemma": "pig", "forms": {"plural": "pigs"}, "freq": 3.3, "aoa": 2.5, "domains": ["ANIMAL"], "features": {"animate": True, "animal": True, "farm": True}},
    {"lemma": "chicken", "forms": {"plural": "chickens"}, "freq": 3.5, "aoa": 2.5, "domains": ["ANIMAL", "FOOD"], "features": {"animate": True, "animal": True, "farm": True}},
    {"lemma": "duck", "forms": {"plural": "ducks"}, "freq": 3.2, "aoa": 2.5, "domains": ["ANIMAL"], "features": {"animate": True, "animal": True, "swims": True}},
    {"lemma": "rabbit", "forms": {"plural": "rabbits"}, "freq": 3.2, "aoa": 2.5, "domains": ["ANIMAL"], "features": {"animate": True, "animal": True, "pet": True}},
    {"lemma": "bear", "forms": {"plural": "bears"}, "freq": 3.5, "aoa": 2.5, "domains": ["ANIMAL"], "features": {"animate": True, "animal": True, "wild": True}},
    {"lemma": "lion", "forms": {"plural": "lions"}, "freq": 3.2, "aoa": 3.0, "domains": ["ANIMAL"], "features": {"animate": True, "animal": True, "wild": True}},
    {"lemma": "tiger", "forms": {"plural": "tigers"}, "freq": 3.0, "aoa": 3.0, "domains": ["ANIMAL"], "features": {"animate": True, "animal": True, "wild": True}},
    {"lemma": "elephant", "forms": {"plural": "elephants"}, "freq": 3.0, "aoa": 3.0, "domains": ["ANIMAL"], "features": {"animate": True, "animal": True, "large": True}},
    {"lemma": "monkey", "forms": {"plural": "monkeys"}, "freq": 3.0, "aoa": 3.0, "domains": ["ANIMAL"], "features": {"animate": True, "animal": True}},
    {"lemma": "snake", "forms": {"plural": "snakes"}, "freq": 3.0, "aoa": 3.0, "domains": ["ANIMAL"], "features": {"animate": True, "animal": True, "reptile": True}},
    {"lemma": "frog", "forms": {"plural": "frogs"}, "freq": 2.8, "aoa": 3.0, "domains": ["ANIMAL"], "features": {"animate": True, "animal": True, "amphibian": True}},
    {"lemma": "butterfly", "forms": {"plural": "butterflies"}, "freq": 2.8, "aoa": 3.0, "domains": ["ANIMAL"], "features": {"animate": True, "animal": True, "insect": True}},
    {"lemma": "bee", "forms": {"plural": "bees"}, "freq": 2.8, "aoa": 3.0, "domains": ["ANIMAL"], "features": {"animate": True, "animal": True, "insect": True}},
]

NOUNS_BODY = [
    {"lemma": "head", "forms": {"plural": "heads"}, "freq": 4.5, "aoa": 2.0, "domains": ["BODY_PART"], "features": {"body": True}},
    {"lemma": "eye", "forms": {"plural": "eyes"}, "freq": 4.5, "aoa": 1.5, "domains": ["BODY_PART"], "features": {"body": True, "sense": True}},
    {"lemma": "ear", "forms": {"plural": "ears"}, "freq": 3.8, "aoa": 2.0, "domains": ["BODY_PART"], "features": {"body": True, "sense": True}},
    {"lemma": "nose", "forms": {"plural": "noses"}, "freq": 3.5, "aoa": 2.0, "domains": ["BODY_PART"], "features": {"body": True, "sense": True}},
    {"lemma": "mouth", "forms": {"plural": "mouths"}, "freq": 4.0, "aoa": 2.0, "domains": ["BODY_PART"], "features": {"body": True}},
    {"lemma": "hand", "forms": {"plural": "hands"}, "freq": 4.8, "aoa": 1.5, "domains": ["BODY_PART"], "features": {"body": True}},
    {"lemma": "foot", "forms": {"plural": "feet"}, "freq": 4.0, "aoa": 2.0, "domains": ["BODY_PART"], "features": {"body": True}},
    {"lemma": "arm", "forms": {"plural": "arms"}, "freq": 4.0, "aoa": 2.5, "domains": ["BODY_PART"], "features": {"body": True}},
    {"lemma": "leg", "forms": {"plural": "legs"}, "freq": 4.0, "aoa": 2.5, "domains": ["BODY_PART"], "features": {"body": True}},
    {"lemma": "finger", "forms": {"plural": "fingers"}, "freq": 3.5, "aoa": 2.5, "domains": ["BODY_PART"], "features": {"body": True}},
    {"lemma": "toe", "forms": {"plural": "toes"}, "freq": 3.0, "aoa": 2.5, "domains": ["BODY_PART"], "features": {"body": True}},
    {"lemma": "hair", "forms": {"plural": "hairs"}, "freq": 4.0, "aoa": 2.0, "domains": ["BODY_PART"], "features": {"body": True}},
    {"lemma": "face", "forms": {"plural": "faces"}, "freq": 4.5, "aoa": 2.0, "domains": ["BODY_PART"], "features": {"body": True}},
    {"lemma": "heart", "forms": {"plural": "hearts"}, "freq": 4.2, "aoa": 3.0, "domains": ["BODY_PART"], "features": {"body": True, "organ": True}},
    {"lemma": "brain", "forms": {"plural": "brains"}, "freq": 3.5, "aoa": 4.0, "domains": ["BODY_PART"], "features": {"body": True, "organ": True}},
]

NOUNS_FOOD = [
    {"lemma": "food", "forms": {"plural": "foods"}, "freq": 4.5, "aoa": 2.0, "domains": ["FOOD"], "features": {"edible": True}},
    {"lemma": "water", "forms": {}, "freq": 5.0, "aoa": 1.5, "domains": ["FOOD"], "features": {"edible": True, "liquid": True, "mass": True}},
    {"lemma": "milk", "forms": {}, "freq": 4.0, "aoa": 1.5, "domains": ["FOOD"], "features": {"edible": True, "liquid": True, "mass": True}},
    {"lemma": "juice", "forms": {"plural": "juices"}, "freq": 3.5, "aoa": 2.0, "domains": ["FOOD"], "features": {"edible": True, "liquid": True}},
    {"lemma": "bread", "forms": {"plural": "breads"}, "freq": 3.8, "aoa": 2.0, "domains": ["FOOD"], "features": {"edible": True}},
    {"lemma": "apple", "forms": {"plural": "apples"}, "freq": 3.5, "aoa": 2.0, "domains": ["FOOD", "PLANT"], "features": {"edible": True, "fruit": True}},
    {"lemma": "banana", "forms": {"plural": "bananas"}, "freq": 3.2, "aoa": 2.0, "domains": ["FOOD", "PLANT"], "features": {"edible": True, "fruit": True}},
    {"lemma": "orange", "forms": {"plural": "oranges"}, "freq": 3.2, "aoa": 2.5, "domains": ["FOOD", "PLANT"], "features": {"edible": True, "fruit": True}},
    {"lemma": "cookie", "forms": {"plural": "cookies"}, "freq": 3.2, "aoa": 2.0, "domains": ["FOOD"], "features": {"edible": True, "sweet": True}},
    {"lemma": "cake", "forms": {"plural": "cakes"}, "freq": 3.5, "aoa": 2.5, "domains": ["FOOD"], "features": {"edible": True, "sweet": True}},
    {"lemma": "egg", "forms": {"plural": "eggs"}, "freq": 3.8, "aoa": 2.0, "domains": ["FOOD"], "features": {"edible": True}},
    {"lemma": "cheese", "forms": {"plural": "cheeses"}, "freq": 3.5, "aoa": 2.5, "domains": ["FOOD"], "features": {"edible": True}},
    {"lemma": "meat", "forms": {"plural": "meats"}, "freq": 3.8, "aoa": 3.0, "domains": ["FOOD"], "features": {"edible": True}},
    {"lemma": "rice", "forms": {}, "freq": 3.5, "aoa": 3.0, "domains": ["FOOD"], "features": {"edible": True, "mass": True}},
    {"lemma": "soup", "forms": {"plural": "soups"}, "freq": 3.2, "aoa": 2.5, "domains": ["FOOD"], "features": {"edible": True, "liquid": True}},
]

NOUNS_OBJECTS = [
    {"lemma": "ball", "forms": {"plural": "balls"}, "freq": 4.0, "aoa": 1.5, "domains": ["OBJECT"], "features": {"toy": True, "round": True}},
    {"lemma": "book", "forms": {"plural": "books"}, "freq": 4.5, "aoa": 2.0, "domains": ["OBJECT"], "features": {"readable": True}},
    {"lemma": "toy", "forms": {"plural": "toys"}, "freq": 3.8, "aoa": 2.0, "domains": ["OBJECT"], "features": {"toy": True}},
    {"lemma": "box", "forms": {"plural": "boxes"}, "freq": 4.0, "aoa": 2.0, "domains": ["OBJECT"], "features": {"container": True}},
    {"lemma": "cup", "forms": {"plural": "cups"}, "freq": 3.8, "aoa": 2.0, "domains": ["OBJECT"], "features": {"container": True}},
    {"lemma": "bottle", "forms": {"plural": "bottles"}, "freq": 3.5, "aoa": 2.0, "domains": ["OBJECT"], "features": {"container": True}},
    {"lemma": "key", "forms": {"plural": "keys"}, "freq": 4.0, "aoa": 2.5, "domains": ["OBJECT", "TOOL"], "features": {"tool": True}},
    {"lemma": "phone", "forms": {"plural": "phones"}, "freq": 4.5, "aoa": 3.0, "domains": ["OBJECT"], "features": {"electronic": True}},
    {"lemma": "clock", "forms": {"plural": "clocks"}, "freq": 3.5, "aoa": 3.0, "domains": ["OBJECT"], "features": {"time": True}},
    {"lemma": "picture", "forms": {"plural": "pictures"}, "freq": 4.0, "aoa": 2.5, "domains": ["OBJECT"], "features": {"visual": True}},
    {"lemma": "paper", "forms": {"plural": "papers"}, "freq": 4.2, "aoa": 3.0, "domains": ["OBJECT"], "features": {"material": True}},
    {"lemma": "money", "forms": {}, "freq": 4.5, "aoa": 3.5, "domains": ["OBJECT"], "features": {"valuable": True, "mass": True}},
    {"lemma": "bag", "forms": {"plural": "bags"}, "freq": 4.0, "aoa": 2.5, "domains": ["OBJECT"], "features": {"container": True}},
    {"lemma": "letter", "forms": {"plural": "letters"}, "freq": 4.0, "aoa": 4.0, "domains": ["OBJECT"], "features": {"communication": True}},
    {"lemma": "gift", "forms": {"plural": "gifts"}, "freq": 3.5, "aoa": 3.0, "domains": ["OBJECT"], "features": {"social": True}},
]

NOUNS_FURNITURE = [
    {"lemma": "table", "forms": {"plural": "tables"}, "freq": 4.2, "aoa": 2.0, "domains": ["OBJECT"], "features": {"furniture": True}},
    {"lemma": "chair", "forms": {"plural": "chairs"}, "freq": 4.0, "aoa": 2.0, "domains": ["OBJECT"], "features": {"furniture": True, "seating": True}},
    {"lemma": "bed", "forms": {"plural": "beds"}, "freq": 4.2, "aoa": 1.5, "domains": ["OBJECT"], "features": {"furniture": True, "sleeping": True}},
    {"lemma": "door", "forms": {"plural": "doors"}, "freq": 4.5, "aoa": 2.0, "domains": ["OBJECT", "BUILDING"], "features": {"opening": True}},
    {"lemma": "window", "forms": {"plural": "windows"}, "freq": 4.0, "aoa": 2.5, "domains": ["OBJECT", "BUILDING"], "features": {"opening": True}},
    {"lemma": "floor", "forms": {"plural": "floors"}, "freq": 4.2, "aoa": 2.5, "domains": ["BUILDING"], "features": {"surface": True}},
    {"lemma": "wall", "forms": {"plural": "walls"}, "freq": 4.0, "aoa": 3.0, "domains": ["BUILDING"], "features": {"structure": True}},
    {"lemma": "couch", "forms": {"plural": "couches"}, "freq": 3.2, "aoa": 3.0, "domains": ["OBJECT"], "features": {"furniture": True, "seating": True}},
    {"lemma": "desk", "forms": {"plural": "desks"}, "freq": 3.5, "aoa": 4.0, "domains": ["OBJECT"], "features": {"furniture": True, "work": True}},
    {"lemma": "lamp", "forms": {"plural": "lamps"}, "freq": 3.0, "aoa": 3.0, "domains": ["OBJECT"], "features": {"light": True}},
]

NOUNS_PLACES = [
    {"lemma": "house", "forms": {"plural": "houses"}, "freq": 4.8, "aoa": 2.0, "domains": ["BUILDING", "SPACE"], "features": {"building": True, "dwelling": True}},
    {"lemma": "home", "forms": {"plural": "homes"}, "freq": 5.0, "aoa": 1.5, "domains": ["BUILDING", "SPACE"], "features": {"building": True, "dwelling": True}},
    {"lemma": "room", "forms": {"plural": "rooms"}, "freq": 4.5, "aoa": 2.5, "domains": ["SPACE"], "features": {"indoor": True}},
    {"lemma": "school", "forms": {"plural": "schools"}, "freq": 4.5, "aoa": 3.0, "domains": ["BUILDING"], "features": {"building": True, "education": True}},
    {"lemma": "store", "forms": {"plural": "stores"}, "freq": 4.0, "aoa": 3.0, "domains": ["BUILDING"], "features": {"building": True, "commerce": True}},
    {"lemma": "park", "forms": {"plural": "parks"}, "freq": 4.0, "aoa": 2.5, "domains": ["SPACE"], "features": {"outdoor": True, "recreation": True}},
    {"lemma": "street", "forms": {"plural": "streets"}, "freq": 4.2, "aoa": 3.0, "domains": ["SPACE"], "features": {"outdoor": True, "path": True}},
    {"lemma": "city", "forms": {"plural": "cities"}, "freq": 4.5, "aoa": 4.0, "domains": ["SPACE"], "features": {"large": True, "urban": True}},
    {"lemma": "country", "forms": {"plural": "countries"}, "freq": 4.5, "aoa": 4.0, "domains": ["SPACE"], "features": {"large": True, "political": True}},
    {"lemma": "world", "forms": {"plural": "worlds"}, "freq": 5.0, "aoa": 3.5, "domains": ["SPACE"], "features": {"large": True}},
    {"lemma": "place", "forms": {"plural": "places"}, "freq": 5.0, "aoa": 3.0, "domains": ["SPACE"], "features": {"abstract": True}},
    {"lemma": "hospital", "forms": {"plural": "hospitals"}, "freq": 3.8, "aoa": 4.0, "domains": ["BUILDING"], "features": {"building": True, "medical": True}},
    {"lemma": "church", "forms": {"plural": "churches"}, "freq": 3.8, "aoa": 4.0, "domains": ["BUILDING"], "features": {"building": True, "religious": True}},
    {"lemma": "library", "forms": {"plural": "libraries"}, "freq": 3.5, "aoa": 5.0, "domains": ["BUILDING"], "features": {"building": True, "education": True}},
    {"lemma": "restaurant", "forms": {"plural": "restaurants"}, "freq": 3.5, "aoa": 5.0, "domains": ["BUILDING"], "features": {"building": True, "food": True}},
]

NOUNS_NATURE = [
    {"lemma": "tree", "forms": {"plural": "trees"}, "freq": 4.0, "aoa": 2.0, "domains": ["PLANT"], "features": {"plant": True, "large": True}},
    {"lemma": "flower", "forms": {"plural": "flowers"}, "freq": 3.8, "aoa": 2.0, "domains": ["PLANT"], "features": {"plant": True, "beautiful": True}},
    {"lemma": "grass", "forms": {}, "freq": 3.5, "aoa": 2.5, "domains": ["PLANT"], "features": {"plant": True, "mass": True}},
    {"lemma": "sun", "forms": {"plural": "suns"}, "freq": 4.2, "aoa": 2.0, "domains": ["SPACE"], "features": {"celestial": True, "light": True}},
    {"lemma": "moon", "forms": {"plural": "moons"}, "freq": 3.8, "aoa": 2.5, "domains": ["SPACE"], "features": {"celestial": True}},
    {"lemma": "star", "forms": {"plural": "stars"}, "freq": 4.0, "aoa": 2.5, "domains": ["SPACE"], "features": {"celestial": True, "light": True}},
    {"lemma": "sky", "forms": {"plural": "skies"}, "freq": 4.0, "aoa": 2.5, "domains": ["SPACE"], "features": {"above": True}},
    {"lemma": "cloud", "forms": {"plural": "clouds"}, "freq": 3.5, "aoa": 3.0, "domains": ["SPACE"], "features": {"weather": True}},
    {"lemma": "rain", "forms": {}, "freq": 4.0, "aoa": 2.5, "domains": ["SPACE"], "features": {"weather": True, "water": True}},
    {"lemma": "snow", "forms": {}, "freq": 3.5, "aoa": 2.5, "domains": ["SPACE"], "features": {"weather": True, "cold": True}},
    {"lemma": "wind", "forms": {"plural": "winds"}, "freq": 3.8, "aoa": 3.0, "domains": ["SPACE"], "features": {"weather": True}},
    {"lemma": "fire", "forms": {"plural": "fires"}, "freq": 4.2, "aoa": 2.5, "domains": ["OBJECT"], "features": {"hot": True, "dangerous": True}},
    {"lemma": "rock", "forms": {"plural": "rocks"}, "freq": 3.5, "aoa": 2.5, "domains": ["OBJECT"], "features": {"natural": True, "hard": True}},
    {"lemma": "mountain", "forms": {"plural": "mountains"}, "freq": 3.5, "aoa": 3.5, "domains": ["SPACE"], "features": {"large": True, "natural": True}},
    {"lemma": "river", "forms": {"plural": "rivers"}, "freq": 3.5, "aoa": 3.5, "domains": ["SPACE"], "features": {"water": True, "natural": True}},
    {"lemma": "ocean", "forms": {"plural": "oceans"}, "freq": 3.5, "aoa": 4.0, "domains": ["SPACE"], "features": {"water": True, "large": True}},
    {"lemma": "beach", "forms": {"plural": "beaches"}, "freq": 3.5, "aoa": 3.5, "domains": ["SPACE"], "features": {"water": True, "sand": True}},
]

NOUNS_ABSTRACT = [
    {"lemma": "time", "forms": {"plural": "times"}, "freq": 5.5, "aoa": 3.0, "domains": ["TIME"], "features": {"abstract": True}},
    {"lemma": "day", "forms": {"plural": "days"}, "freq": 5.2, "aoa": 2.0, "domains": ["TIME"], "features": {"abstract": True}},
    {"lemma": "night", "forms": {"plural": "nights"}, "freq": 4.8, "aoa": 2.0, "domains": ["TIME"], "features": {"abstract": True}},
    {"lemma": "year", "forms": {"plural": "years"}, "freq": 5.0, "aoa": 3.5, "domains": ["TIME"], "features": {"abstract": True}},
    {"lemma": "week", "forms": {"plural": "weeks"}, "freq": 4.5, "aoa": 4.0, "domains": ["TIME"], "features": {"abstract": True}},
    {"lemma": "month", "forms": {"plural": "months"}, "freq": 4.2, "aoa": 4.0, "domains": ["TIME"], "features": {"abstract": True}},
    {"lemma": "hour", "forms": {"plural": "hours"}, "freq": 4.5, "aoa": 4.0, "domains": ["TIME"], "features": {"abstract": True}},
    {"lemma": "minute", "forms": {"plural": "minutes"}, "freq": 4.5, "aoa": 4.5, "domains": ["TIME"], "features": {"abstract": True}},
    {"lemma": "thing", "forms": {"plural": "things"}, "freq": 5.5, "aoa": 2.5, "domains": ["QUALITY"], "features": {"abstract": True}},
    {"lemma": "way", "forms": {"plural": "ways"}, "freq": 5.5, "aoa": 3.0, "domains": ["QUALITY"], "features": {"abstract": True}},
    {"lemma": "name", "forms": {"plural": "names"}, "freq": 5.0, "aoa": 2.0, "domains": ["COGNITION"], "features": {"abstract": True}},
    {"lemma": "word", "forms": {"plural": "words"}, "freq": 4.8, "aoa": 3.0, "domains": ["COGNITION"], "features": {"abstract": True}},
    {"lemma": "idea", "forms": {"plural": "ideas"}, "freq": 4.5, "aoa": 5.0, "domains": ["COGNITION"], "features": {"abstract": True}},
    {"lemma": "thought", "forms": {"plural": "thoughts"}, "freq": 4.2, "aoa": 5.0, "domains": ["COGNITION"], "features": {"abstract": True}},
    {"lemma": "question", "forms": {"plural": "questions"}, "freq": 4.5, "aoa": 4.0, "domains": ["COGNITION"], "features": {"abstract": True}},
    {"lemma": "answer", "forms": {"plural": "answers"}, "freq": 4.2, "aoa": 4.0, "domains": ["COGNITION"], "features": {"abstract": True}},
    {"lemma": "problem", "forms": {"plural": "problems"}, "freq": 4.5, "aoa": 5.0, "domains": ["COGNITION"], "features": {"abstract": True}},
    {"lemma": "reason", "forms": {"plural": "reasons"}, "freq": 4.5, "aoa": 5.0, "domains": ["COGNITION"], "features": {"abstract": True}},
    {"lemma": "truth", "forms": {"plural": "truths"}, "freq": 4.0, "aoa": 6.0, "domains": ["COGNITION"], "features": {"abstract": True}},
    {"lemma": "life", "forms": {"plural": "lives"}, "freq": 5.2, "aoa": 3.0, "domains": ["QUALITY"], "features": {"abstract": True}},
    {"lemma": "death", "forms": {"plural": "deaths"}, "freq": 4.5, "aoa": 4.0, "domains": ["QUALITY"], "features": {"abstract": True}},
]

NOUNS_EMOTIONS = [
    {"lemma": "love", "forms": {"plural": "loves"}, "freq": 5.0, "aoa": 2.5, "domains": ["EMOTION"], "features": {"abstract": True, "positive": True}},
    {"lemma": "hate", "forms": {"plural": "hates"}, "freq": 4.0, "aoa": 3.5, "domains": ["EMOTION"], "features": {"abstract": True, "negative": True}},
    {"lemma": "fear", "forms": {"plural": "fears"}, "freq": 4.2, "aoa": 3.5, "domains": ["EMOTION"], "features": {"abstract": True, "negative": True}},
    {"lemma": "joy", "forms": {"plural": "joys"}, "freq": 3.5, "aoa": 4.0, "domains": ["EMOTION"], "features": {"abstract": True, "positive": True}},
    {"lemma": "anger", "forms": {}, "freq": 3.8, "aoa": 4.0, "domains": ["EMOTION"], "features": {"abstract": True, "negative": True}},
    {"lemma": "hope", "forms": {"plural": "hopes"}, "freq": 4.2, "aoa": 4.0, "domains": ["EMOTION"], "features": {"abstract": True, "positive": True}},
    {"lemma": "pain", "forms": {"plural": "pains"}, "freq": 4.2, "aoa": 3.0, "domains": ["EMOTION"], "features": {"abstract": True, "negative": True}},
    {"lemma": "surprise", "forms": {"plural": "surprises"}, "freq": 3.8, "aoa": 4.0, "domains": ["EMOTION"], "features": {"abstract": True}},
    {"lemma": "happiness", "forms": {}, "freq": 3.5, "aoa": 4.0, "domains": ["EMOTION"], "features": {"abstract": True, "positive": True}},
    {"lemma": "sadness", "forms": {}, "freq": 3.0, "aoa": 4.0, "domains": ["EMOTION"], "features": {"abstract": True, "negative": True}},
]

# Combine all nouns
NOUNS = (
    NOUNS_PEOPLE + NOUNS_ANIMALS + NOUNS_BODY + NOUNS_FOOD + 
    NOUNS_OBJECTS + NOUNS_FURNITURE + NOUNS_PLACES + NOUNS_NATURE + 
    NOUNS_ABSTRACT + NOUNS_EMOTIONS
)

