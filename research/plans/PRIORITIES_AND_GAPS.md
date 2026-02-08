# What else would be most valuable?

Beyond the theoretical throughlines (SM, ML/linear algebra, complex systems) and the existing plans (control, robotics, curriculum), these additions would add the **most value** to the research program — by strengthening theory, unblocking validation, or turning results into publishable claims.

---

## 1. Unblock validation: autonomous recurrence and noise robustness

**Q21** (autonomous recurrence not supported) and **Q10** (noise robustness needs rework) are **blockers** for several claims:

- You cannot properly test "attractor" or "autonomous recovery" if the implementation never runs pure recurrence (or only in limited scenarios). Phase diagram and attractor experiments use `project({}, {"A": ["A"]})`; if that path is broken or stimulus is still required elsewhere, key theory predictions are untestable.
- **Noise robustness** was deemed invalid because the test kept providing stimulus, so "recovery" was trivial. A **redesigned** test: train assembly → remove stimulus → inject noise (or perturb winners) → run **only** self-projection for several steps → measure whether the same assembly is recovered. That requires **true autonomous recurrence** to be supported and used.

**Most valuable action:** Resolve **Q21** (extend or clarify when brain runs pure recurrent steps) and **redesign Q10** so the protocol is falsifiable (no stimulus during recovery, clear success/failure metric). Until then, attractor and noise-robustness claims are weak.

---

## 2. One derived theory result (not just empirical)

You already have **empirical** phase diagram (Q02) and scaling (Q03). The throughlines doc adds **frameworks** (mean-field, contraction, etc.). The highest leverage next step is **one concrete derivation** that makes a **testable prediction**:

- **Option A — Phase boundary:** Mean-field or cavity argument that predicts a **critical** (e.g. p_c or k/n_c) where the system goes from "no stable assembly" to "stable attractor." Then compare to your phase diagram data.
- **Option B — Convergence rate:** Contraction argument that predicts **convergence in O(log n) steps** (or a specific exponent). You already have O(log N) empirically (Q03); a derivation would explain *why* and whether it’s universal.
- **Option C — Capacity:** Upper or lower bound on number of stable attractors as a function of n, k, p (e.g. from random-matrix or stability condition). Then check in simulation.

**Most valuable action:** Pick **one** of A/B/C and do it (even heuristics or bounds). That turns "we have a phase diagram" into "we have a phase diagram *and* a derived critical point," or "we have scaling *and* a theoretical explanation." One such result is worth more than many more empirical plots.

---

## 3. Claims pipeline: turn validated results into defensible claims

You have **validated** results (Q01, Q03, Q11, Q20, Q07, Q09*) but the research README still says "Validated Claims: To be populated." Without a **claims** layer, papers stay vague ("we ran experiments") instead of crisp ("we claim X; here is the evidence; here are the limitations").

**Most valuable action:** For each validated question, add a **claim** in `claims/`: one short **claim.md** (one sentence), **evidence.md** (which experiments, which metrics), **limitations.md** (what we do *not* claim), and optionally **suitable_venues.md**. Start with Q01, Q03, Q11, Q20. That gives you a clear "what we can defend" list and a direct path to paper sections.

---

## 4. Biological validation: one concrete comparison to data

**Q06** (assembly detection in real data) is Tier 2. Even a **small** step would elevate the work: e.g. compare **assembly-like statistics** (sparsity, overlap, persistence) in your simulations to published statistics from one neural dataset (e.g. calcium imaging or spike data), with clear assumptions. You don’t need to "detect" assemblies in data yet — showing "our simulated statistics are in the same ballpark as real data" is already a claim.

**Most valuable action:** Choose **one** dataset and **one** statistic (e.g. sparsity, or distribution of pairwise correlations in active sets) and do a side-by-side comparison with your model. Document assumptions and limitations. That makes the "biological relevance" claim concrete instead of speculative.

---

## 5. Learning rules (Q12) and catastrophic forgetting

**Q12** (Hebbian plasticity, stable modification without catastrophic forgetting) is Tier 2 and **central** to the calculus: association and merge rely on Hebbian updates. You don’t yet have a clear experiment that tests "we can add a new assembly without destroying old ones" under controlled conditions (e.g. fixed number of assemblies, measure overlap before/after learning a new one).

**Most valuable action:** Design one **minimal** experiment: e.g. form assemblies A and B, then form C (association or merge); measure overlap A–B, A–C, B–C and stability of A and B after learning C. That would support (or falsify) "Hebbian assembly learning is stable" and feed into the learning chapter of the monograph.

---

## 6. Falsifiability and experimental design (Q19)

Several experiments have been found **invalid** or **unfalsifiable** (noise robustness, some distinctiveness setups). Making **each** key claim tied to an experiment that **could** disprove it would strengthen the whole program.

**Most valuable action:** For each Tier 1–2 claim, write one paragraph: "We claim X. An experiment that would **falsify** X is: … . We ran it and got … ." If you can’t state a falsifying experiment, the claim is not yet testable. Add this to **Q19** (experimental design) and to the validation framework (Q18).

---

## Summary: order of impact

| Priority | Action | Why |
|----------|--------|-----|
| **1** | Resolve Q21 + redesign Q10 (autonomous recurrence, noise robustness) | Unblocks attractor and recovery claims; fixes invalid test. |
| **2** | One derived theory result (phase boundary, convergence rate, or capacity) | Turns empirical results into theory-backed predictions. |
| **3** | Populate claims/ for validated Qs (Q01, Q03, Q11, Q20) | Turns "we have results" into "we claim X with evidence and limitations." |
| **4** | One biological comparison (one dataset, one statistic) | Makes biological relevance concrete. |
| **5** | Q12: minimal learning-stability experiment | Central to calculus; currently untested. |
| **6** | Falsifiability pass (Q19): one falsifying experiment per key claim | Strengthens scientific rigor. |

These are **complementary** to the theoretical throughlines (ML/LA, complex systems, SM): the throughlines tell you *what* to derive and *how* to frame; the items above tell you *what* to fix, *what* to derive first, and *how* to turn results into defensible, publishable claims.
