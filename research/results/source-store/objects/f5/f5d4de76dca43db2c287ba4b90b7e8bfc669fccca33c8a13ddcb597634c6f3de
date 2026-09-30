"""What role updates does the ACTUAL role-training corpus emit for `dog`?

`train_unsupervised` is called with a PRECOMPILED `corpus_index`, so wrapping
`compile_corpus` captured nothing. The corpus that trains roles is therefore not
necessarily stage3+stage4, and assuming it is would repeat the wrong-corpus
error that put `chases` in the ERP frames.
"""
import collections
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"F:\Github\assemblies")))
os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")
os.environ.pop("EMERGENT_DEV_CURRICULUM", None)

from neural_assemblies.assembly_calculus.emergent import EmergentParser
from neural_assemblies.assembly_calculus.emergent.curriculum import CurriculumTrainer
from neural_assemblies.assembly_calculus.emergent.parser_mixins import (
    unsupervised as unsup_mod,
)

captured = []
for cls_name in dir(unsup_mod):
    cls = getattr(unsup_mod, cls_name)
    if isinstance(cls, type) and hasattr(cls, "train_unsupervised"):
        target, orig = cls, cls.train_unsupervised
        break


def wrapped(self_, sentences, repetitions=3, *, corpus_index=None, **kw):
    captured.append((sentences, corpus_index))
    return orig(self_, sentences, repetitions,
                corpus_index=corpus_index, **kw)


target.train_unsupervised = wrapped

g = json.loads(Path(r"F:\Github\assemblies\research\literature\parity\golden"
                    r"\nemo2025_curriculum.json").read_text(encoding="utf-8"))
p = g["parameters"]
parser = EmergentParser(n=p["n"], k=p["k"], p=p["p"], beta=p["beta"],
                        seed=42, rounds=p["rounds"])
tr = CurriculumTrainer(parser)
for stage in p["stages"]:
    tr.train_stage(stage)
target.train_unsupervised = orig

print(f"train_unsupervised captured {len(captured)} times")
roles = collections.Counter()
dog_sents = []
for sents, ci in captured:
    print(f"  call: {len(sents) if sents else 0} sentences, "
          f"corpus_index={'given' if ci is not None else 'None'}")
    idx = ci
    if idx is None:
        continue
    for u in idx.role_updates:
        if u.word == "dog":
            roles[u.role_area] += 1
    for s in getattr(idx, "raw", []):
        if "dog" in s:
            dog_sents.append(s)

print("role_updates for 'dog':", dict(roles) or "NONE")
print(f"sentences containing dog in the ROLE corpus: {len(dog_sents)}")
for s in dog_sents[:15]:
    print("   ", " ".join(s) if isinstance(s, list) else s)
