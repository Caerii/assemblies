"""Verbs - Organized by semantic type"""

# Format: {"lemma", "forms": {"3sg": ..., "past": ..., "ppart": ..., "prog": ...}, 
#          "freq": log_freq, "aoa": age, "domains": [...], "features": {...}, "args": [...]}

VERBS_MOTION = [
    {"lemma": "go", "forms": {"3sg": "goes", "past": "went", "ppart": "gone", "prog": "going"}, "freq": 5.8, "aoa": 1.5, "domains": ["MOTION"], "features": {"intransitive": True}, "args": ["agent"]},
    {"lemma": "come", "forms": {"3sg": "comes", "past": "came", "ppart": "come", "prog": "coming"}, "freq": 5.5, "aoa": 1.5, "domains": ["MOTION"], "features": {"intransitive": True}, "args": ["agent"]},
    {"lemma": "run", "forms": {"3sg": "runs", "past": "ran", "ppart": "run", "prog": "running"}, "freq": 4.5, "aoa": 2.0, "domains": ["MOTION"], "features": {"intransitive": True, "manner": True}, "args": ["agent"]},
    {"lemma": "walk", "forms": {"3sg": "walks", "past": "walked", "ppart": "walked", "prog": "walking"}, "freq": 4.2, "aoa": 2.0, "domains": ["MOTION"], "features": {"intransitive": True, "manner": True}, "args": ["agent"]},
    {"lemma": "jump", "forms": {"3sg": "jumps", "past": "jumped", "ppart": "jumped", "prog": "jumping"}, "freq": 3.5, "aoa": 2.5, "domains": ["MOTION"], "features": {"intransitive": True}, "args": ["agent"]},
    {"lemma": "fall", "forms": {"3sg": "falls", "past": "fell", "ppart": "fallen", "prog": "falling"}, "freq": 4.2, "aoa": 2.5, "domains": ["MOTION"], "features": {"intransitive": True, "unaccusative": True}, "args": ["theme"]},
    {"lemma": "fly", "forms": {"3sg": "flies", "past": "flew", "ppart": "flown", "prog": "flying"}, "freq": 4.0, "aoa": 2.5, "domains": ["MOTION"], "features": {"intransitive": True}, "args": ["agent"]},
    {"lemma": "swim", "forms": {"3sg": "swims", "past": "swam", "ppart": "swum", "prog": "swimming"}, "freq": 3.2, "aoa": 3.0, "domains": ["MOTION"], "features": {"intransitive": True}, "args": ["agent"]},
    {"lemma": "climb", "forms": {"3sg": "climbs", "past": "climbed", "ppart": "climbed", "prog": "climbing"}, "freq": 3.2, "aoa": 3.0, "domains": ["MOTION"], "features": {"ambitransitive": True}, "args": ["agent", "path"]},
    {"lemma": "move", "forms": {"3sg": "moves", "past": "moved", "ppart": "moved", "prog": "moving"}, "freq": 4.5, "aoa": 3.0, "domains": ["MOTION"], "features": {"ambitransitive": True}, "args": ["agent", "theme"]},
    {"lemma": "turn", "forms": {"3sg": "turns", "past": "turned", "ppart": "turned", "prog": "turning"}, "freq": 4.5, "aoa": 3.0, "domains": ["MOTION"], "features": {"ambitransitive": True}, "args": ["agent", "theme"]},
    {"lemma": "stop", "forms": {"3sg": "stops", "past": "stopped", "ppart": "stopped", "prog": "stopping"}, "freq": 4.5, "aoa": 2.5, "domains": ["MOTION"], "features": {"ambitransitive": True}, "args": ["agent", "theme"]},
    {"lemma": "start", "forms": {"3sg": "starts", "past": "started", "ppart": "started", "prog": "starting"}, "freq": 4.8, "aoa": 3.0, "domains": ["MOTION"], "features": {"ambitransitive": True}, "args": ["agent", "theme"]},
    {"lemma": "leave", "forms": {"3sg": "leaves", "past": "left", "ppart": "left", "prog": "leaving"}, "freq": 4.8, "aoa": 3.0, "domains": ["MOTION"], "features": {"transitive": True}, "args": ["agent", "source"]},
    {"lemma": "return", "forms": {"3sg": "returns", "past": "returned", "ppart": "returned", "prog": "returning"}, "freq": 4.2, "aoa": 4.0, "domains": ["MOTION"], "features": {"intransitive": True}, "args": ["agent"]},
    {"lemma": "arrive", "forms": {"3sg": "arrives", "past": "arrived", "ppart": "arrived", "prog": "arriving"}, "freq": 4.0, "aoa": 4.0, "domains": ["MOTION"], "features": {"intransitive": True}, "args": ["agent"]},
    {"lemma": "enter", "forms": {"3sg": "enters", "past": "entered", "ppart": "entered", "prog": "entering"}, "freq": 4.0, "aoa": 4.0, "domains": ["MOTION"], "features": {"transitive": True}, "args": ["agent", "goal"]},
    {"lemma": "follow", "forms": {"3sg": "follows", "past": "followed", "ppart": "followed", "prog": "following"}, "freq": 4.2, "aoa": 3.5, "domains": ["MOTION"], "features": {"transitive": True}, "args": ["agent", "theme"]},
    {"lemma": "chase", "forms": {"3sg": "chases", "past": "chased", "ppart": "chased", "prog": "chasing"}, "freq": 3.5, "aoa": 3.0, "domains": ["MOTION"], "features": {"transitive": True}, "args": ["agent", "theme"]},
    {"lemma": "escape", "forms": {"3sg": "escapes", "past": "escaped", "ppart": "escaped", "prog": "escaping"}, "freq": 3.5, "aoa": 5.0, "domains": ["MOTION"], "features": {"intransitive": True}, "args": ["agent"]},
]

VERBS_PERCEPTION = [
    {"lemma": "see", "forms": {"3sg": "sees", "past": "saw", "ppart": "seen", "prog": "seeing"}, "freq": 5.5, "aoa": 1.5, "domains": ["PERCEPTION"], "features": {"transitive": True, "stative": True}, "args": ["experiencer", "stimulus"]},
    {"lemma": "look", "forms": {"3sg": "looks", "past": "looked", "ppart": "looked", "prog": "looking"}, "freq": 5.2, "aoa": 1.5, "domains": ["PERCEPTION"], "features": {"intransitive": True}, "args": ["agent"]},
    {"lemma": "watch", "forms": {"3sg": "watches", "past": "watched", "ppart": "watched", "prog": "watching"}, "freq": 4.5, "aoa": 2.0, "domains": ["PERCEPTION"], "features": {"transitive": True}, "args": ["agent", "theme"]},
    {"lemma": "hear", "forms": {"3sg": "hears", "past": "heard", "ppart": "heard", "prog": "hearing"}, "freq": 4.8, "aoa": 2.0, "domains": ["PERCEPTION"], "features": {"transitive": True, "stative": True}, "args": ["experiencer", "stimulus"]},
    {"lemma": "listen", "forms": {"3sg": "listens", "past": "listened", "ppart": "listened", "prog": "listening"}, "freq": 4.0, "aoa": 2.5, "domains": ["PERCEPTION"], "features": {"intransitive": True}, "args": ["agent"]},
    {"lemma": "feel", "forms": {"3sg": "feels", "past": "felt", "ppart": "felt", "prog": "feeling"}, "freq": 4.8, "aoa": 2.5, "domains": ["PERCEPTION"], "features": {"ambitransitive": True}, "args": ["experiencer", "stimulus"]},
    {"lemma": "touch", "forms": {"3sg": "touches", "past": "touched", "ppart": "touched", "prog": "touching"}, "freq": 4.0, "aoa": 2.5, "domains": ["PERCEPTION"], "features": {"transitive": True}, "args": ["agent", "patient"]},
    {"lemma": "smell", "forms": {"3sg": "smells", "past": "smelled", "ppart": "smelled", "prog": "smelling"}, "freq": 3.5, "aoa": 3.0, "domains": ["PERCEPTION"], "features": {"ambitransitive": True}, "args": ["experiencer", "stimulus"]},
    {"lemma": "taste", "forms": {"3sg": "tastes", "past": "tasted", "ppart": "tasted", "prog": "tasting"}, "freq": 3.2, "aoa": 3.0, "domains": ["PERCEPTION"], "features": {"ambitransitive": True}, "args": ["experiencer", "stimulus"]},
    {"lemma": "notice", "forms": {"3sg": "notices", "past": "noticed", "ppart": "noticed", "prog": "noticing"}, "freq": 4.0, "aoa": 4.0, "domains": ["PERCEPTION"], "features": {"transitive": True}, "args": ["experiencer", "stimulus"]},
    {"lemma": "find", "forms": {"3sg": "finds", "past": "found", "ppart": "found", "prog": "finding"}, "freq": 5.0, "aoa": 2.5, "domains": ["PERCEPTION"], "features": {"transitive": True}, "args": ["agent", "theme"]},
]

VERBS_COMMUNICATION = [
    {"lemma": "say", "forms": {"3sg": "says", "past": "said", "ppart": "said", "prog": "saying"}, "freq": 5.8, "aoa": 1.5, "domains": ["COMMUNICATION"], "features": {"transitive": True}, "args": ["agent", "theme"]},
    {"lemma": "tell", "forms": {"3sg": "tells", "past": "told", "ppart": "told", "prog": "telling"}, "freq": 5.2, "aoa": 2.0, "domains": ["COMMUNICATION"], "features": {"ditransitive": True}, "args": ["agent", "recipient", "theme"]},
    {"lemma": "talk", "forms": {"3sg": "talks", "past": "talked", "ppart": "talked", "prog": "talking"}, "freq": 4.5, "aoa": 2.0, "domains": ["COMMUNICATION"], "features": {"intransitive": True}, "args": ["agent"]},
    {"lemma": "speak", "forms": {"3sg": "speaks", "past": "spoke", "ppart": "spoken", "prog": "speaking"}, "freq": 4.2, "aoa": 3.0, "domains": ["COMMUNICATION"], "features": {"intransitive": True}, "args": ["agent"]},
    {"lemma": "ask", "forms": {"3sg": "asks", "past": "asked", "ppart": "asked", "prog": "asking"}, "freq": 5.0, "aoa": 2.5, "domains": ["COMMUNICATION"], "features": {"ditransitive": True}, "args": ["agent", "recipient", "theme"]},
    {"lemma": "answer", "forms": {"3sg": "answers", "past": "answered", "ppart": "answered", "prog": "answering"}, "freq": 4.2, "aoa": 3.0, "domains": ["COMMUNICATION"], "features": {"transitive": True}, "args": ["agent", "theme"]},
    {"lemma": "call", "forms": {"3sg": "calls", "past": "called", "ppart": "called", "prog": "calling"}, "freq": 5.0, "aoa": 2.0, "domains": ["COMMUNICATION"], "features": {"transitive": True}, "args": ["agent", "theme"]},
    {"lemma": "read", "forms": {"3sg": "reads", "past": "read", "ppart": "read", "prog": "reading"}, "freq": 4.5, "aoa": 3.0, "domains": ["COMMUNICATION"], "features": {"transitive": True}, "args": ["agent", "theme"]},
    {"lemma": "write", "forms": {"3sg": "writes", "past": "wrote", "ppart": "written", "prog": "writing"}, "freq": 4.5, "aoa": 3.5, "domains": ["COMMUNICATION"], "features": {"transitive": True}, "args": ["agent", "theme"]},
    {"lemma": "sing", "forms": {"3sg": "sings", "past": "sang", "ppart": "sung", "prog": "singing"}, "freq": 3.8, "aoa": 2.5, "domains": ["COMMUNICATION"], "features": {"ambitransitive": True}, "args": ["agent", "theme"]},
    {"lemma": "shout", "forms": {"3sg": "shouts", "past": "shouted", "ppart": "shouted", "prog": "shouting"}, "freq": 3.5, "aoa": 3.0, "domains": ["COMMUNICATION"], "features": {"intransitive": True}, "args": ["agent"]},
    {"lemma": "whisper", "forms": {"3sg": "whispers", "past": "whispered", "ppart": "whispered", "prog": "whispering"}, "freq": 3.2, "aoa": 4.0, "domains": ["COMMUNICATION"], "features": {"intransitive": True}, "args": ["agent"]},
    {"lemma": "explain", "forms": {"3sg": "explains", "past": "explained", "ppart": "explained", "prog": "explaining"}, "freq": 4.0, "aoa": 5.0, "domains": ["COMMUNICATION"], "features": {"transitive": True}, "args": ["agent", "theme"]},
    {"lemma": "describe", "forms": {"3sg": "describes", "past": "described", "ppart": "described", "prog": "describing"}, "freq": 3.8, "aoa": 5.0, "domains": ["COMMUNICATION"], "features": {"transitive": True}, "args": ["agent", "theme"]},
]

VERBS_POSSESSION = [
    {"lemma": "have", "forms": {"3sg": "has", "past": "had", "ppart": "had", "prog": "having"}, "freq": 6.0, "aoa": 1.5, "domains": ["POSSESSION"], "features": {"transitive": True, "stative": True}, "args": ["possessor", "theme"]},
    {"lemma": "get", "forms": {"3sg": "gets", "past": "got", "ppart": "gotten", "prog": "getting"}, "freq": 5.8, "aoa": 1.5, "domains": ["POSSESSION"], "features": {"transitive": True}, "args": ["agent", "theme"]},
    {"lemma": "give", "forms": {"3sg": "gives", "past": "gave", "ppart": "given", "prog": "giving"}, "freq": 5.2, "aoa": 2.0, "domains": ["POSSESSION"], "features": {"ditransitive": True}, "args": ["agent", "recipient", "theme"]},
    {"lemma": "take", "forms": {"3sg": "takes", "past": "took", "ppart": "taken", "prog": "taking"}, "freq": 5.2, "aoa": 2.0, "domains": ["POSSESSION"], "features": {"transitive": True}, "args": ["agent", "theme"]},
    {"lemma": "bring", "forms": {"3sg": "brings", "past": "brought", "ppart": "brought", "prog": "bringing"}, "freq": 4.5, "aoa": 2.5, "domains": ["POSSESSION"], "features": {"ditransitive": True}, "args": ["agent", "recipient", "theme"]},
    {"lemma": "keep", "forms": {"3sg": "keeps", "past": "kept", "ppart": "kept", "prog": "keeping"}, "freq": 4.8, "aoa": 2.5, "domains": ["POSSESSION"], "features": {"transitive": True}, "args": ["agent", "theme"]},
    {"lemma": "put", "forms": {"3sg": "puts", "past": "put", "ppart": "put", "prog": "putting"}, "freq": 5.0, "aoa": 2.0, "domains": ["POSSESSION"], "features": {"transitive": True}, "args": ["agent", "theme", "location"]},
    {"lemma": "hold", "forms": {"3sg": "holds", "past": "held", "ppart": "held", "prog": "holding"}, "freq": 4.5, "aoa": 2.5, "domains": ["POSSESSION"], "features": {"transitive": True}, "args": ["agent", "theme"]},
    {"lemma": "carry", "forms": {"3sg": "carries", "past": "carried", "ppart": "carried", "prog": "carrying"}, "freq": 4.0, "aoa": 3.0, "domains": ["POSSESSION"], "features": {"transitive": True}, "args": ["agent", "theme"]},
    {"lemma": "send", "forms": {"3sg": "sends", "past": "sent", "ppart": "sent", "prog": "sending"}, "freq": 4.5, "aoa": 3.5, "domains": ["POSSESSION"], "features": {"ditransitive": True}, "args": ["agent", "recipient", "theme"]},
    {"lemma": "receive", "forms": {"3sg": "receives", "past": "received", "ppart": "received", "prog": "receiving"}, "freq": 4.0, "aoa": 5.0, "domains": ["POSSESSION"], "features": {"transitive": True}, "args": ["agent", "theme"]},
    {"lemma": "buy", "forms": {"3sg": "buys", "past": "bought", "ppart": "bought", "prog": "buying"}, "freq": 4.5, "aoa": 3.0, "domains": ["POSSESSION"], "features": {"transitive": True}, "args": ["agent", "theme"]},
    {"lemma": "sell", "forms": {"3sg": "sells", "past": "sold", "ppart": "sold", "prog": "selling"}, "freq": 4.2, "aoa": 4.0, "domains": ["POSSESSION"], "features": {"ditransitive": True}, "args": ["agent", "recipient", "theme"]},
    {"lemma": "pay", "forms": {"3sg": "pays", "past": "paid", "ppart": "paid", "prog": "paying"}, "freq": 4.5, "aoa": 4.0, "domains": ["POSSESSION"], "features": {"transitive": True}, "args": ["agent", "theme"]},
    {"lemma": "steal", "forms": {"3sg": "steals", "past": "stole", "ppart": "stolen", "prog": "stealing"}, "freq": 3.5, "aoa": 4.0, "domains": ["POSSESSION"], "features": {"transitive": True}, "args": ["agent", "theme"]},
    {"lemma": "share", "forms": {"3sg": "shares", "past": "shared", "ppart": "shared", "prog": "sharing"}, "freq": 4.0, "aoa": 3.0, "domains": ["POSSESSION"], "features": {"transitive": True}, "args": ["agent", "theme"]},
    {"lemma": "lose", "forms": {"3sg": "loses", "past": "lost", "ppart": "lost", "prog": "losing"}, "freq": 4.5, "aoa": 3.0, "domains": ["POSSESSION"], "features": {"transitive": True}, "args": ["agent", "theme"]},
]

VERBS_COGNITION = [
    {"lemma": "know", "forms": {"3sg": "knows", "past": "knew", "ppart": "known", "prog": "knowing"}, "freq": 5.5, "aoa": 2.0, "domains": ["COGNITION"], "features": {"transitive": True, "stative": True}, "args": ["experiencer", "theme"]},
    {"lemma": "think", "forms": {"3sg": "thinks", "past": "thought", "ppart": "thought", "prog": "thinking"}, "freq": 5.5, "aoa": 2.5, "domains": ["COGNITION"], "features": {"transitive": True}, "args": ["experiencer", "theme"]},
    {"lemma": "understand", "forms": {"3sg": "understands", "past": "understood", "ppart": "understood", "prog": "understanding"}, "freq": 4.5, "aoa": 3.5, "domains": ["COGNITION"], "features": {"transitive": True}, "args": ["experiencer", "theme"]},
    {"lemma": "believe", "forms": {"3sg": "believes", "past": "believed", "ppart": "believed", "prog": "believing"}, "freq": 4.5, "aoa": 4.0, "domains": ["COGNITION"], "features": {"transitive": True}, "args": ["experiencer", "theme"]},
    {"lemma": "remember", "forms": {"3sg": "remembers", "past": "remembered", "ppart": "remembered", "prog": "remembering"}, "freq": 4.5, "aoa": 3.0, "domains": ["COGNITION"], "features": {"transitive": True}, "args": ["experiencer", "theme"]},
    {"lemma": "forget", "forms": {"3sg": "forgets", "past": "forgot", "ppart": "forgotten", "prog": "forgetting"}, "freq": 4.2, "aoa": 3.0, "domains": ["COGNITION"], "features": {"transitive": True}, "args": ["experiencer", "theme"]},
    {"lemma": "learn", "forms": {"3sg": "learns", "past": "learned", "ppart": "learned", "prog": "learning"}, "freq": 4.5, "aoa": 3.0, "domains": ["COGNITION"], "features": {"transitive": True}, "args": ["experiencer", "theme"]},
    {"lemma": "teach", "forms": {"3sg": "teaches", "past": "taught", "ppart": "taught", "prog": "teaching"}, "freq": 4.2, "aoa": 3.5, "domains": ["COGNITION"], "features": {"ditransitive": True}, "args": ["agent", "recipient", "theme"]},
    {"lemma": "decide", "forms": {"3sg": "decides", "past": "decided", "ppart": "decided", "prog": "deciding"}, "freq": 4.2, "aoa": 4.0, "domains": ["COGNITION"], "features": {"transitive": True}, "args": ["agent", "theme"]},
    {"lemma": "choose", "forms": {"3sg": "chooses", "past": "chose", "ppart": "chosen", "prog": "choosing"}, "freq": 4.2, "aoa": 3.5, "domains": ["COGNITION"], "features": {"transitive": True}, "args": ["agent", "theme"]},
    {"lemma": "guess", "forms": {"3sg": "guesses", "past": "guessed", "ppart": "guessed", "prog": "guessing"}, "freq": 3.8, "aoa": 3.5, "domains": ["COGNITION"], "features": {"transitive": True}, "args": ["agent", "theme"]},
    {"lemma": "wonder", "forms": {"3sg": "wonders", "past": "wondered", "ppart": "wondered", "prog": "wondering"}, "freq": 4.0, "aoa": 4.0, "domains": ["COGNITION"], "features": {"transitive": True}, "args": ["experiencer", "theme"]},
    {"lemma": "imagine", "forms": {"3sg": "imagines", "past": "imagined", "ppart": "imagined", "prog": "imagining"}, "freq": 4.0, "aoa": 4.0, "domains": ["COGNITION"], "features": {"transitive": True}, "args": ["experiencer", "theme"]},
    {"lemma": "dream", "forms": {"3sg": "dreams", "past": "dreamed", "ppart": "dreamed", "prog": "dreaming"}, "freq": 3.8, "aoa": 3.0, "domains": ["COGNITION"], "features": {"intransitive": True}, "args": ["experiencer"]},
]

VERBS_EMOTION = [
    {"lemma": "love", "forms": {"3sg": "loves", "past": "loved", "ppart": "loved", "prog": "loving"}, "freq": 5.0, "aoa": 2.0, "domains": ["EMOTION"], "features": {"transitive": True, "stative": True}, "args": ["experiencer", "stimulus"]},
    {"lemma": "like", "forms": {"3sg": "likes", "past": "liked", "ppart": "liked", "prog": "liking"}, "freq": 5.2, "aoa": 2.0, "domains": ["EMOTION"], "features": {"transitive": True, "stative": True}, "args": ["experiencer", "stimulus"]},
    {"lemma": "hate", "forms": {"3sg": "hates", "past": "hated", "ppart": "hated", "prog": "hating"}, "freq": 4.2, "aoa": 3.0, "domains": ["EMOTION"], "features": {"transitive": True, "stative": True}, "args": ["experiencer", "stimulus"]},
    {"lemma": "want", "forms": {"3sg": "wants", "past": "wanted", "ppart": "wanted", "prog": "wanting"}, "freq": 5.5, "aoa": 1.5, "domains": ["EMOTION"], "features": {"transitive": True}, "args": ["experiencer", "theme"]},
    {"lemma": "need", "forms": {"3sg": "needs", "past": "needed", "ppart": "needed", "prog": "needing"}, "freq": 5.2, "aoa": 2.5, "domains": ["EMOTION"], "features": {"transitive": True}, "args": ["experiencer", "theme"]},
    {"lemma": "hope", "forms": {"3sg": "hopes", "past": "hoped", "ppart": "hoped", "prog": "hoping"}, "freq": 4.5, "aoa": 3.5, "domains": ["EMOTION"], "features": {"transitive": True}, "args": ["experiencer", "theme"]},
    {"lemma": "wish", "forms": {"3sg": "wishes", "past": "wished", "ppart": "wished", "prog": "wishing"}, "freq": 4.2, "aoa": 3.0, "domains": ["EMOTION"], "features": {"transitive": True}, "args": ["experiencer", "theme"]},
    {"lemma": "fear", "forms": {"3sg": "fears", "past": "feared", "ppart": "feared", "prog": "fearing"}, "freq": 3.8, "aoa": 4.0, "domains": ["EMOTION"], "features": {"transitive": True}, "args": ["experiencer", "stimulus"]},
    {"lemma": "enjoy", "forms": {"3sg": "enjoys", "past": "enjoyed", "ppart": "enjoyed", "prog": "enjoying"}, "freq": 4.2, "aoa": 4.0, "domains": ["EMOTION"], "features": {"transitive": True}, "args": ["experiencer", "stimulus"]},
    {"lemma": "miss", "forms": {"3sg": "misses", "past": "missed", "ppart": "missed", "prog": "missing"}, "freq": 4.5, "aoa": 3.0, "domains": ["EMOTION"], "features": {"transitive": True}, "args": ["experiencer", "stimulus"]},
    {"lemma": "care", "forms": {"3sg": "cares", "past": "cared", "ppart": "cared", "prog": "caring"}, "freq": 4.5, "aoa": 3.0, "domains": ["EMOTION"], "features": {"intransitive": True}, "args": ["experiencer"]},
    {"lemma": "worry", "forms": {"3sg": "worries", "past": "worried", "ppart": "worried", "prog": "worrying"}, "freq": 4.0, "aoa": 4.0, "domains": ["EMOTION"], "features": {"intransitive": True}, "args": ["experiencer"]},
    {"lemma": "surprise", "forms": {"3sg": "surprises", "past": "surprised", "ppart": "surprised", "prog": "surprising"}, "freq": 4.0, "aoa": 4.0, "domains": ["EMOTION"], "features": {"transitive": True}, "args": ["stimulus", "experiencer"]},
]

VERBS_CREATION = [
    {"lemma": "make", "forms": {"3sg": "makes", "past": "made", "ppart": "made", "prog": "making"}, "freq": 5.5, "aoa": 2.0, "domains": ["CREATION"], "features": {"transitive": True}, "args": ["agent", "theme"]},
    {"lemma": "build", "forms": {"3sg": "builds", "past": "built", "ppart": "built", "prog": "building"}, "freq": 4.2, "aoa": 3.0, "domains": ["CREATION"], "features": {"transitive": True}, "args": ["agent", "theme"]},
    {"lemma": "create", "forms": {"3sg": "creates", "past": "created", "ppart": "created", "prog": "creating"}, "freq": 4.2, "aoa": 5.0, "domains": ["CREATION"], "features": {"transitive": True}, "args": ["agent", "theme"]},
    {"lemma": "draw", "forms": {"3sg": "draws", "past": "drew", "ppart": "drawn", "prog": "drawing"}, "freq": 4.0, "aoa": 2.5, "domains": ["CREATION"], "features": {"transitive": True}, "args": ["agent", "theme"]},
    {"lemma": "paint", "forms": {"3sg": "paints", "past": "painted", "ppart": "painted", "prog": "painting"}, "freq": 3.5, "aoa": 3.0, "domains": ["CREATION"], "features": {"transitive": True}, "args": ["agent", "theme"]},
    {"lemma": "cook", "forms": {"3sg": "cooks", "past": "cooked", "ppart": "cooked", "prog": "cooking"}, "freq": 3.8, "aoa": 3.0, "domains": ["CREATION"], "features": {"transitive": True}, "args": ["agent", "theme"]},
    {"lemma": "grow", "forms": {"3sg": "grows", "past": "grew", "ppart": "grown", "prog": "growing"}, "freq": 4.2, "aoa": 3.0, "domains": ["CREATION"], "features": {"ambitransitive": True}, "args": ["agent", "theme"]},
    {"lemma": "fix", "forms": {"3sg": "fixes", "past": "fixed", "ppart": "fixed", "prog": "fixing"}, "freq": 4.0, "aoa": 3.5, "domains": ["CREATION"], "features": {"transitive": True}, "args": ["agent", "theme"]},
    {"lemma": "break", "forms": {"3sg": "breaks", "past": "broke", "ppart": "broken", "prog": "breaking"}, "freq": 4.5, "aoa": 2.5, "domains": ["DESTRUCTION"], "features": {"ambitransitive": True}, "args": ["agent", "patient"]},
    {"lemma": "cut", "forms": {"3sg": "cuts", "past": "cut", "ppart": "cut", "prog": "cutting"}, "freq": 4.2, "aoa": 2.5, "domains": ["DESTRUCTION"], "features": {"transitive": True}, "args": ["agent", "patient"]},
    {"lemma": "destroy", "forms": {"3sg": "destroys", "past": "destroyed", "ppart": "destroyed", "prog": "destroying"}, "freq": 3.8, "aoa": 5.0, "domains": ["DESTRUCTION"], "features": {"transitive": True}, "args": ["agent", "patient"]},
    {"lemma": "kill", "forms": {"3sg": "kills", "past": "killed", "ppart": "killed", "prog": "killing"}, "freq": 4.5, "aoa": 4.0, "domains": ["DESTRUCTION"], "features": {"transitive": True}, "args": ["agent", "patient"]},
]

VERBS_CONSUMPTION = [
    {"lemma": "eat", "forms": {"3sg": "eats", "past": "ate", "ppart": "eaten", "prog": "eating"}, "freq": 4.5, "aoa": 1.5, "domains": ["CONSUMPTION"], "features": {"transitive": True}, "args": ["agent", "theme"]},
    {"lemma": "drink", "forms": {"3sg": "drinks", "past": "drank", "ppart": "drunk", "prog": "drinking"}, "freq": 4.2, "aoa": 1.5, "domains": ["CONSUMPTION"], "features": {"transitive": True}, "args": ["agent", "theme"]},
    {"lemma": "sleep", "forms": {"3sg": "sleeps", "past": "slept", "ppart": "slept", "prog": "sleeping"}, "freq": 4.2, "aoa": 1.5, "domains": ["CONSUMPTION"], "features": {"intransitive": True}, "args": ["agent"]},
    {"lemma": "wake", "forms": {"3sg": "wakes", "past": "woke", "ppart": "woken", "prog": "waking"}, "freq": 4.0, "aoa": 2.0, "domains": ["CONSUMPTION"], "features": {"ambitransitive": True}, "args": ["agent", "patient"]},
    {"lemma": "rest", "forms": {"3sg": "rests", "past": "rested", "ppart": "rested", "prog": "resting"}, "freq": 3.5, "aoa": 3.5, "domains": ["CONSUMPTION"], "features": {"intransitive": True}, "args": ["agent"]},
    {"lemma": "breathe", "forms": {"3sg": "breathes", "past": "breathed", "ppart": "breathed", "prog": "breathing"}, "freq": 3.5, "aoa": 4.0, "domains": ["CONSUMPTION"], "features": {"intransitive": True}, "args": ["agent"]},
]

VERBS_STATES = [
    {"lemma": "be", "forms": {"3sg": "is", "past": "was", "ppart": "been", "prog": "being", "1sg": "am", "2sg": "are", "pl": "are", "past_pl": "were"}, "freq": 6.5, "aoa": 1.0, "domains": ["QUALITY"], "features": {"stative": True, "copula": True}, "args": ["theme"]},
    {"lemma": "become", "forms": {"3sg": "becomes", "past": "became", "ppart": "become", "prog": "becoming"}, "freq": 4.8, "aoa": 3.5, "domains": ["QUALITY"], "features": {"inchoative": True}, "args": ["theme"]},
    {"lemma": "seem", "forms": {"3sg": "seems", "past": "seemed", "ppart": "seemed", "prog": "seeming"}, "freq": 4.5, "aoa": 4.0, "domains": ["QUALITY"], "features": {"stative": True}, "args": ["theme"]},
    {"lemma": "appear", "forms": {"3sg": "appears", "past": "appeared", "ppart": "appeared", "prog": "appearing"}, "freq": 4.2, "aoa": 4.0, "domains": ["QUALITY"], "features": {"stative": True}, "args": ["theme"]},
    {"lemma": "remain", "forms": {"3sg": "remains", "past": "remained", "ppart": "remained", "prog": "remaining"}, "freq": 4.0, "aoa": 5.0, "domains": ["QUALITY"], "features": {"stative": True}, "args": ["theme"]},
    {"lemma": "stay", "forms": {"3sg": "stays", "past": "stayed", "ppart": "stayed", "prog": "staying"}, "freq": 4.5, "aoa": 2.5, "domains": ["QUALITY"], "features": {"stative": True}, "args": ["agent"]},
    {"lemma": "live", "forms": {"3sg": "lives", "past": "lived", "ppart": "lived", "prog": "living"}, "freq": 5.0, "aoa": 2.5, "domains": ["QUALITY"], "features": {"stative": True}, "args": ["agent"]},
    {"lemma": "die", "forms": {"3sg": "dies", "past": "died", "ppart": "died", "prog": "dying"}, "freq": 4.5, "aoa": 3.5, "domains": ["QUALITY"], "features": {"inchoative": True}, "args": ["theme"]},
    {"lemma": "exist", "forms": {"3sg": "exists", "past": "existed", "ppart": "existed", "prog": "existing"}, "freq": 4.0, "aoa": 6.0, "domains": ["QUALITY"], "features": {"stative": True}, "args": ["theme"]},
]

VERBS_ACTION = [
    {"lemma": "do", "forms": {"3sg": "does", "past": "did", "ppart": "done", "prog": "doing"}, "freq": 6.0, "aoa": 1.5, "domains": ["QUALITY"], "features": {"transitive": True}, "args": ["agent", "theme"]},
    {"lemma": "try", "forms": {"3sg": "tries", "past": "tried", "ppart": "tried", "prog": "trying"}, "freq": 5.0, "aoa": 2.5, "domains": ["QUALITY"], "features": {"transitive": True}, "args": ["agent", "theme"]},
    {"lemma": "help", "forms": {"3sg": "helps", "past": "helped", "ppart": "helped", "prog": "helping"}, "freq": 4.8, "aoa": 2.5, "domains": ["SOCIAL"], "features": {"transitive": True}, "args": ["agent", "beneficiary"]},
    {"lemma": "work", "forms": {"3sg": "works", "past": "worked", "ppart": "worked", "prog": "working"}, "freq": 5.0, "aoa": 3.0, "domains": ["QUALITY"], "features": {"intransitive": True}, "args": ["agent"]},
    {"lemma": "play", "forms": {"3sg": "plays", "past": "played", "ppart": "played", "prog": "playing"}, "freq": 4.8, "aoa": 2.0, "domains": ["QUALITY"], "features": {"ambitransitive": True}, "args": ["agent", "theme"]},
    {"lemma": "use", "forms": {"3sg": "uses", "past": "used", "ppart": "used", "prog": "using"}, "freq": 5.0, "aoa": 3.0, "domains": ["QUALITY"], "features": {"transitive": True}, "args": ["agent", "instrument"]},
    {"lemma": "open", "forms": {"3sg": "opens", "past": "opened", "ppart": "opened", "prog": "opening"}, "freq": 4.5, "aoa": 2.0, "domains": ["QUALITY"], "features": {"ambitransitive": True}, "args": ["agent", "patient"]},
    {"lemma": "close", "forms": {"3sg": "closes", "past": "closed", "ppart": "closed", "prog": "closing"}, "freq": 4.5, "aoa": 2.0, "domains": ["QUALITY"], "features": {"ambitransitive": True}, "args": ["agent", "patient"]},
    {"lemma": "sit", "forms": {"3sg": "sits", "past": "sat", "ppart": "sat", "prog": "sitting"}, "freq": 4.5, "aoa": 2.0, "domains": ["QUALITY"], "features": {"intransitive": True}, "args": ["agent"]},
    {"lemma": "stand", "forms": {"3sg": "stands", "past": "stood", "ppart": "stood", "prog": "standing"}, "freq": 4.5, "aoa": 2.0, "domains": ["QUALITY"], "features": {"intransitive": True}, "args": ["agent"]},
    {"lemma": "wait", "forms": {"3sg": "waits", "past": "waited", "ppart": "waited", "prog": "waiting"}, "freq": 4.5, "aoa": 2.5, "domains": ["QUALITY"], "features": {"intransitive": True}, "args": ["agent"]},
    {"lemma": "change", "forms": {"3sg": "changes", "past": "changed", "ppart": "changed", "prog": "changing"}, "freq": 4.8, "aoa": 3.5, "domains": ["QUALITY"], "features": {"ambitransitive": True}, "args": ["agent", "patient"]},
    {"lemma": "begin", "forms": {"3sg": "begins", "past": "began", "ppart": "begun", "prog": "beginning"}, "freq": 4.5, "aoa": 3.5, "domains": ["QUALITY"], "features": {"ambitransitive": True}, "args": ["agent", "theme"]},
    {"lemma": "end", "forms": {"3sg": "ends", "past": "ended", "ppart": "ended", "prog": "ending"}, "freq": 4.5, "aoa": 3.5, "domains": ["QUALITY"], "features": {"ambitransitive": True}, "args": ["agent", "theme"]},
    {"lemma": "finish", "forms": {"3sg": "finishes", "past": "finished", "ppart": "finished", "prog": "finishing"}, "freq": 4.2, "aoa": 3.0, "domains": ["QUALITY"], "features": {"transitive": True}, "args": ["agent", "theme"]},
    {"lemma": "continue", "forms": {"3sg": "continues", "past": "continued", "ppart": "continued", "prog": "continuing"}, "freq": 4.2, "aoa": 4.0, "domains": ["QUALITY"], "features": {"ambitransitive": True}, "args": ["agent", "theme"]},
    {"lemma": "happen", "forms": {"3sg": "happens", "past": "happened", "ppart": "happened", "prog": "happening"}, "freq": 4.8, "aoa": 3.5, "domains": ["QUALITY"], "features": {"intransitive": True}, "args": ["theme"]},
    {"lemma": "let", "forms": {"3sg": "lets", "past": "let", "ppart": "let", "prog": "letting"}, "freq": 5.0, "aoa": 2.5, "domains": ["QUALITY"], "features": {"transitive": True}, "args": ["agent", "theme"]},
    {"lemma": "show", "forms": {"3sg": "shows", "past": "showed", "ppart": "shown", "prog": "showing"}, "freq": 5.0, "aoa": 2.5, "domains": ["QUALITY"], "features": {"ditransitive": True}, "args": ["agent", "recipient", "theme"]},
    {"lemma": "meet", "forms": {"3sg": "meets", "past": "met", "ppart": "met", "prog": "meeting"}, "freq": 4.5, "aoa": 3.0, "domains": ["SOCIAL"], "features": {"transitive": True}, "args": ["agent", "theme"]},
]

# Combine all verbs
VERBS = (
    VERBS_MOTION + VERBS_PERCEPTION + VERBS_COMMUNICATION + 
    VERBS_POSSESSION + VERBS_COGNITION + VERBS_EMOTION + 
    VERBS_CREATION + VERBS_CONSUMPTION + VERBS_STATES + VERBS_ACTION
)

