"""#28: is the N400 saturated, or was it read on the wrong channel?

CONTEXT. #28 records the N400 as "saturated / bit-identical across parse
arms". The shipped N400 (`measure_lexical_surprise`) reads mean pre-k-WTA
ENERGY landing on the word's stored PREDICTION assembly -- a HOW-MUCH
quantity. #121 measured the landing channel (competition OUTCOME under an
isolated probe) at AUC 0.916 on role pathways where the drive family read
near chance. `prediction_landing_surprise` (registered with this
experiment) reads the SAME settling -- `_settle_context_into_prediction`,
the one shared implementation -- and scores 1 - overlap(settled assembly,
stored entry).

DESIGN. Contexts are REAL training-corpus sentence prefixes; attestation is
computed from the corpus itself (bigram on the prefix's last word), so the
arms cannot differ by fiat labels:

    attested         the sentence's actual next word (trained noun)
    unattested_noun  trained noun that NEVER follows the prefix's last word
    verb_violation   trained verb, same zero-bigram requirement
    novel            declared-holdout word (registered stimulus, never
                     trained; arm VOID and reported if none exist)

Both channels are measured per item on the same parser (within-substrate,
paired). Every escape is `Measured.undefined` and REPORTED, never 1.0.

REGISTERED PREDICTIONS (before any cell; each can fail):
  P-SAT   The ENERGY channel's semantic contrast (unattested_noun above
          attested) reads AUC <= 0.65 -- the #28 insensitivity claim made
          precise. If it EXCEEDS 0.75, #28's premise is STALE on the
          post-ratchet engine (the #121 P-DRIVE precedent) and is reported
          as such, not hidden.
  P-LIVE  The landing channel is not a constant: per-arm defined values
          have range > 0.05 on every seed. A dead readout fails here.
  P-SEM   The open question, all three readings pre-stated:
            AUC(unattested_noun above attested) >= 0.75  => LEXICAL
              next-word expectation exists in the substrate; the N400 arm
              of the 2x2 is LIVE.
            ~0.5 while P-SYN separates => prediction is CATEGORY-level
              only -- the finding is #87's representational limit measured
              from the ERP side, and the N400 is testable only at the
              category grain.
            < 0.5 (inversion) => the pre-norm_init post-k-WTA sign hazard
              (module docstring of erp/adapters.py) resurfaced; diagnose
              via per-item overlap distributions before ANY use.
  P-SYN   AUC(verb_violation above attested) >= 0.75 on the landing
          channel -- category-level expectation is the minimum the
          substrate should carry (TWO_WORD+ trains transitions).
  VOID RULE  an arm with < half its items defined voids every contrast it
          enters, with per-arm undefined counts printed.
  DECISION RULE (pre-stated): #28 closes as MEASURED whichever way P-SEM
          resolves. `prediction_landing_surprise` becomes the sanctioned
          N400 quantity ONLY if P-LIVE holds and P-SEM is not inverted;
          the full crossed 2x2 (lexical x syntactic, against
          role_binding_deficit) is the registered follow-up, not part of
          this unit. No composite or threshold changes here.

SEEDS 7,11,12,19,42-47 (10). Reseeded per cell. 12 contexts/seed, 1 item
per arm per context.
"""
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

os.environ.setdefault("EMERGENT_FAST_TRAINING", "1")
os.environ.setdefault("TRAIN_PROGRESS", "0")
os.environ.setdefault("EMERGENT_ERP_FAST", "1")

import json                                                             # noqa: E402

import numpy as np                                                      # noqa: E402

from neural_assemblies.assembly_calculus.emergent.evaluation.erp.adapters import (  # noqa: E402
    measure_lexical_surprise, prediction_landing_surprise,
)
from neural_assemblies.assembly_calculus.emergent.evaluation.generalization import (  # noqa: E402
    default_holdout_set,
)
from neural_assemblies.assembly_calculus.emergent.evaluation.sweep import (  # noqa: E402
    get_parser_cache,
)
from neural_assemblies.diagnostics import separation                    # noqa: E402

SEEDS = [7, 11, 12, 19, 42, 43, 44, 45, 46, 47]
N_CTX = 12
ARMS = ("attested", "unattested_noun", "verb_violation", "novel")
OUT_PATH = os.path.join(os.path.dirname(__file__),
                        "n400_landing_2x2_results.json")


def _corpus():
    from neural_assemblies.lexicon.curriculum.stage4_sentences import STAGE4_CORPUS
    from neural_assemblies.lexicon.curriculum.stage3_two_word import STAGE3_CORPUS
    return [line.split() for line in list(STAGE4_CORPUS) + list(STAGE3_CORPUS)]


def _build_items(parser, rng):
    """Contexts + one word per arm, all requirements checked on THIS parser."""
    sentences = _corpus()
    bigram = Counter()
    for words in sentences:
        for a, b in zip(words, words[1:]):
            bigram[(a, b)] += 1

    nouns = sorted(w for w in parser.core_lexicons.get("NOUN_CORE", {})
                   if parser.classify_word(w)[0] == "NOUN")
    verbs = sorted(w for w in parser.core_lexicons.get("VERB_CORE", {})
                   if parser.classify_word(w)[0] == "VERB")
    trained = {w for lex in parser.core_lexicons.values() for w in lex}
    novel_pool = sorted(w for w in default_holdout_set()
                        if w in parser.stim_map and w not in trained)

    items = []
    seen_prefix = set()
    for words in rng.sample(sentences, len(sentences)):
        if len(words) < 3:
            continue
        prefix, target = tuple(words[:-1]), words[-1]
        if prefix in seen_prefix or target not in nouns:
            continue
        last = prefix[-1]
        un_nouns = [w for w in nouns
                    if w != target and bigram[(last, w)] == 0]
        un_verbs = [w for w in verbs if bigram[(last, w)] == 0]
        if not un_nouns or not un_verbs:
            continue
        seen_prefix.add(prefix)
        items.append({
            "prefix": prefix,
            "attested": target,
            "unattested_noun": rng.choice(un_nouns),
            "verb_violation": rng.choice(un_verbs),
            "novel": rng.choice(novel_pool) if novel_pool else None,
        })
        if len(items) >= N_CTX:
            break
    return items


def main():
    out = {"seeds": {}, "analysis": {}}
    for seed in SEEDS:
        random.seed(seed)
        np.random.seed(seed)
        rng = random.Random(seed)
        parser = get_parser_cache().fork("SENTENCES", seed=seed)
        items = _build_items(parser, rng)
        if len(items) < N_CTX / 2:
            print(f"seed {seed}: only {len(items)} usable contexts; skipping")
            continue

        vals = {ch: defaultdict(list) for ch in ("energy", "landing")}
        undef = {ch: Counter() for ch in ("energy", "landing")}
        rows = []
        for it in items:
            for arm in ARMS:
                w = it[arm]
                if w is None:
                    continue
                e = measure_lexical_surprise(parser, it["prefix"], w)
                l = prediction_landing_surprise(parser, it["prefix"], w)
                row = {"prefix": " ".join(it["prefix"]), "arm": arm, "word": w}
                for ch, m in (("energy", e), ("landing", l)):
                    if m.defined:
                        vals[ch][arm].append(float(m))
                        row[ch] = float(m)
                        if ch == "landing":
                            row["overlap"] = (m.detail or {}).get("overlap")
                    else:
                        undef[ch][arm] += 1
                        row[f"{ch}_undefined"] = m.why[:80]
                rows.append(row)

        rec = {"n_contexts": len(items), "rows": rows,
               "undefined": {ch: dict(c) for ch, c in undef.items()},
               "auc": {}}
        n_items = len(items)
        print(f"=== seed {seed} ({n_items} contexts) ===")
        for ch in ("energy", "landing"):
            for name, hi in (("sem", "unattested_noun"),
                             ("syn", "verb_violation"),
                             ("nov", "novel")):
                hi_v, lo_v = vals[ch][hi], vals[ch]["attested"]
                if len(hi_v) < n_items / 2 or len(lo_v) < n_items / 2:
                    print(f"  {ch:8s} {name} VOID (n={len(hi_v)},{len(lo_v)})")
                    continue
                s = separation(hi_v, lo_v, label=f"{ch}:{name}")
                rec["auc"][f"{name}_{ch}"] = s.auc
            means = {a: (round(sum(v) / len(v), 4) if (v := vals[ch][a])
                         else None) for a in ARMS}
            clip = {a: sum(1 for x in vals[ch][a] if x >= 0.95) for a in ARMS}
            rec[f"{ch}_arm_means"] = means
            rec[f"{ch}_clip_ge_095"] = clip
            print(f"  {ch:8s} AUC " + " ".join(
                f"{k.split('_')[0]}={v:.3f}" for k, v in rec["auc"].items()
                if k.endswith(ch)) + f"  means={means}  undef={dict(undef[ch])}")
        out["seeds"][str(seed)] = rec

    print("=" * 66)
    for key in sorted({k for rec in out["seeds"].values()
                       for k in rec["auc"]}):
        aucs = [rec["auc"][key] for rec in out["seeds"].values()
                if key in rec["auc"]]
        m = sum(aucs) / len(aucs)
        sd = (sum((a - m) ** 2 for a in aucs) / max(len(aucs) - 1, 1)) ** 0.5
        out["analysis"][key] = {"mean": m, "sd": sd, "n": len(aucs),
                                "per_seed": aucs}
        print(f"  {key:16s} AUC={m:.4f} sd={sd:.4f} n={len(aucs)}")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"-> {OUT_PATH}")


if __name__ == "__main__":
    main()
