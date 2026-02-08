# Curriculum task analysis and evidence

**Goal:** Analyze every task in the embodied + social curriculum; question whether each makes sense given the assembly calculus; and assess what evidence we have (or lack) to support it.

---

## Summary table

| Stage | Task (short) | Makes sense? | Evidence | Verdict |
|-------|--------------|--------------|----------|---------|
| 0 | Object + action assemblies in one area (no words) | Only if stim-only or multi-area | Stim-only → distinct (low overlap); stim+self → collapse | **Conditional** |
| 1 | Object–action association (affordance) in shared area | Yes | Association A→B recovery 0.85–0.95 | **Supported** |
| 2 | First words (nouns/verbs) grounded in visual/motor | Only if distinct assemblies per word | Same as Stage 0: stim-only or multi-area | **Conditional** |
| 3 | Two-word phrases (red block, grasp cup) | Binding assumes distinct concepts | Depends on 0, 2 | **Conditional** |
| 4 | Commands (pick up the red block) | Coordination of concepts + action | No direct evidence | **No evidence** |
| 5 | Dialogue (Q&A, clarification) | Multi-turn, production + comprehension | NEMO has Q&A; no embodied evidence | **Partial** |
| 6 | Two robots, shared task | Perspective, turn-taking | No evidence | **No evidence** |
| 7 | Robot–robot dialogue | Coordination, conventions | No evidence | **No evidence** |
| 8 | Open-ended, novelty, curiosity | Generalization, drive | No evidence | **No evidence** |

**Overall:** We have **solid evidence** for projection stability, cross-area association, and (under stim-only) distinct assemblies in one area. We have **no evidence** for embodied control, production readout, multi-turn dialogue, multi-agent, or Isaac Lab. The curriculum is **partially evidence-based** for Stages 0–2 if we **restrict to stim-only or multi-area**; Stages 3–8 are **aspirational** and need validation or redesign.

---

## Stage 0: Sensorimotor bootstrapping

**Tasks:** (1) Form object-specific assemblies (e.g. “cup”) from visual/proprioceptive input in a sensory area. (2) Form action assemblies (reach, grasp, place) as motor primitives. (3) No words; reward from RL or scripted success.

**Does it make sense?** Yes, if “concept = assembly” and we can have **multiple distinct** assemblies (one per object, one per action). The assembly calculus supports projection, Hebbian plasticity, and winner-take-all; it does **not** by itself specify how many assemblies can coexist in one area.

**Evidence:**

- **Projection (test_projection, RESULTS_projection):** Single assembly converges in ~5 rounds, persistence 0.86–0.98 depending on n and stim-only vs stim+self. So **one** stable assembly per area is supported.
- **Distinctiveness (test_assembly_distinctiveness, RESULTS_assembly_distinctiveness):**
  - **Stim+self** in one area, sequential training → pairwise overlap **0.98–0.99** (collapse to one attractor). So **multiple distinct assemblies in one area under stim+self is contradicted**.
  - **Stim-only** (current test_assembly_distinctiveness.py): no self-projection → “no global attractor”; design supports distinct assemblies. **Competition_mechanisms** JSON: sequential/interleaved overlaps **~2–3%** (distinct). So **stim-only in one area can yield distinct assemblies**.
- **Q20 (open_questions):** “Same brain” → 3% overlap (distinct); “separate brains” → 94% (not distinct). So **competition in same brain** is the regime where we get distinctiveness.
- **Projection H2:** Stim-only persistence **0.86** (vs 0.94 stim+self). So if we use stim-only for multiple objects, assemblies are **less persistent** (0.86) but distinct.

**Questions:**

1. **Stim-only vs stim+self for Stage 0?** Curriculum does not specify. If we use stim+self for “stable” object/action assemblies, we get **one** global attractor (distinctiveness failure). If we use stim-only, we get distinct assemblies but **lower persistence** (0.86). So we must either: (a) **restrict Stage 0 to stim-only** for multiple object/action assemblies and accept 0.86 persistence, or (b) use **one area per object/action** (multi-area) and allow stim+self per area.
2. **Encoder: continuous obs → assembly.** Curriculum assumes “encode visual + proprioceptive state → sensory area” so that “cup” input drives “cup” assembly. We have **no experiment** that implements this encoder (continuous → discrete assembly). Brain currently takes **discrete stimuli** (stimulus index) or fixed winner sets. So “object-specific assembly from vision” is **not validated**; it’s an assumption.
3. **Motor readout.** “Reach assembly → reach trajectory” assumes a readout (assembly → continuous action). We have **no experiment** that trains or evaluates this readout. So motor primitive as assembly + readout is **assumed**, not evidenced.

**Verdict:** **Conditional.** Stage 0 is consistent with the calculus **only if** we (a) use **stim-only** for multiple object/action assemblies in one area (and accept lower persistence), or (b) use **multi-area** (one area per concept). We do **not** have evidence for continuous→assembly encoding or for assembly→action readout; both need to be added and tested.

---

## Stage 1: Affordances (object–action binding)

**Tasks:** Co-activate object assembly + action assembly in a shared (association) area; Hebbian strengthens link; “see cup → grasp” without words.

**Does it make sense?** Yes. This is exactly **association**: two source areas (object, action) project into a third (association); the association primitive in the codebase does A→C and B→C (and optionally A↔B).

**Evidence:**

- **Association (test_association, RESULTS_association):** A→B recovery **0.85–0.95** (n=200–2000); 1 round ≈ 50 rounds; identity of A/B preserved. So **cross-area association is well supported**.
- **Merge** (A+B→C) is implemented; curriculum Stage 1 could use association (A,B→C) or merge. Association experiment is A→B only; merge has separate tests. So **association** has direct evidence; **merge** (both at once) is in code but not summarized here.

**Questions:**

1. **Object and action as separate areas?** If Stage 0 uses one area for “all objects” (stim-only), then “object assembly” is which winners are active in that area; “action assembly” could be another area (motor). Then “object area + motor area → association area” is exactly A,B→C. So Stage 1 **makes sense** given Stage 0 with at least two areas (object/sensory, motor).
2. **Generalization (“novel object similar to cup”).** Curriculum asks whether “grasp” generalizes to mug. We have **no experiment** on generalization of association to novel stimuli (e.g. new object assembly similar to “cup”). So generalization is **assumed**, not evidenced.

**Verdict:** **Supported.** Association (cross-area binding) is well validated. Stage 1 is evidence-based **provided** Stage 0 yields distinct object and action assemblies (stim-only or multi-area). Generalization to novel objects is not yet evidenced.

---

## Stage 2: First words (embodied)

**Tasks:** Name objects (“cup” while showing cup) and actions (“grasp” while grasping); language area receives word form; associate language + visual and language + motor; comprehension (word → select object or action) and optionally production (show object → say “cup”).

**Does it make sense?** Yes, if we have (a) distinct assemblies per object/action (Stage 0) and (b) a “language” area that can hold distinct assemblies per word. Same distinctiveness issue as Stage 0: **multiple word assemblies in one language area** require stim-only or multi-area, or we get collapse.

**Evidence:**

- **NEMO (lexicon, emergent learner):** Words grounded in visual/motor; First Words stage; vocabulary and grounding exist in code. So **language grounding in principle** is implemented, but in a **non-embodied** (no Isaac, no robot) setting.
- **Distinctiveness:** Same as Stage 0. If “language area” is one area and we train multiple words with **stim+self**, we expect collapse. If we use **stim-only** or **one area per word** (or per word type), we can have distinct word assemblies. So Stage 2 is **conditional** on the same design choice as Stage 0.
- **Production (“read out word from language area”).** We have **no experiment** that decodes assembly → word (e.g. winner indices → “cup”). NEMO has production in language models; the assembly→symbol readout for embodied production is **not** validated.

**Questions:**

1. **How does “language area receives ‘cup’” work?** Either (a) discrete input “cup” as a stimulus that drives a dedicated assembly, or (b) continuous (e.g. audio/embedding) mapped to assembly. We have evidence for (a) only (discrete stimulus → assembly). So curriculum should be explicit: instruction is **discrete word tokens** driving language area, unless we add and validate a continuous→assembly encoder for language.
2. **Comprehension check (“say ‘cup’; present two objects”).** Requires: word “cup” → language assembly → projection to visual/object area or association area → selection of cup. That’s multi-area projection + possibly competition. We have **association** evidence (A→B recovery) but not “word → correct object selection in a 2AFC” in an embodied setup. So comprehension as described is **plausible** but **not evidenced** in this form.

**Verdict:** **Conditional.** First words are consistent with the calculus and NEMO-style grounding **if** we use stim-only or multi-area for both object and word assemblies. We lack evidence for (1) continuous language input → assembly, (2) assembly → production (word readout), and (3) embodied comprehension (word → object/action selection in Isaac).

---

## Stage 3: Two-word & phrases

**Tasks:** Adjective–noun (“red block”), verb–object (“grasp cup”), verb–particle (“pick up”); robot selects correct object or executes action on correct object.

**Does it make sense?** Yes, as **binding** two concepts (property + kind, or action + object). That requires (a) distinct assemblies for “red”, “block”, “grasp”, “cup”, and (b) a mechanism to combine them (e.g. both project to same area, or sequence of projections). Association/merge (A,B→C) in principle does that.

**Evidence:**

- **Association:** We have A→B and A,B→C (merge) in code; association recovery is strong. We do **not** have an experiment that does “red” + “block” → select red block from a set, or “grasp” + “cup” → grasp cup. So **two-concept binding in a task** is **not** directly tested.
- **NEMO:** Two-word stage, adjective–noun, verb–object generators exist. So **linguistic structure** is in the curriculum/generators; **embodied execution** (phrase → action on correct object) is not.

**Questions:**

1. **“Red” and “block” in one area or two?** If one area holds both “red” and “block” assemblies (stim-only), we need a **composition** step: activate “red” and “block” and get “red block” (subset of objects). That could be intersection of two assemblies (if we had a set interpretation) or a third area that receives both. The assembly calculus doesn’t have an explicit “intersection” primitive; it has projection and merge. So “red block” might be implemented as: “red” assembly + “block” assembly → association area → readout that selects object satisfying both. That’s **not** a primitive we have validated; it’s a design choice to test.
2. **“Grasp cup”: verb–object.** Same as Stage 1 affordance but cued by **language** (word “grasp” + word “cup”) instead of visual (object in view). So we need language→object and language→action mappings plus binding. That’s Stage 2 + Stage 1 + binding. No direct evidence for this chain.

**Verdict:** **Conditional.** Two-word binding is **conceptually** supported by association/merge and NEMO two-word stage. We have **no evidence** for (1) modifier–noun binding (e.g. “red block” → selection) or (2) verb–object from **language** (e.g. “grasp cup” → grasp the cup) in an embodied task. Depends on Stages 0 and 2 being valid and on implementing composition/binding in a way we can test.

---

## Stage 4: Commands & simple sentences

**Tasks:** Full command comprehension (“Pick up the red block”, “Give me the cup”) → robot executes; optionally production of intent (“I will pick up the red block”) then act; reference resolution (“which block?”); spatial (“put the block to the left of the cup”).

**Does it make sense?** Yes, as **coordination** of multiple concepts (action + object + property/location) and possibly **sequence** (say then do). MHC order 6–7.

**Evidence:**

- **None** for embodied command execution or intent production. We have projection, association, stability; we do not have “natural language command → robot action” or “assembly readout → utterance” in any experiment.
- **NEMO:** Telegraphic/simple sentences exist in the curriculum; no embodied counterpart.

**Questions:**

1. **Reference resolution.** “The red block” in a scene with multiple blocks requires binding “red” + “block” + selection. We have not tested this.
2. **Spatial (“left of the cup”).** Requires spatial relation assemblies or spatial area. NEMO has spatial grounding in vocabulary; we have **no experiment** that tests spatial language → action (e.g. “put X left of Y”).
3. **Production of intent.** Readout from “plan” or “motor” area to language (assembly → word sequence). No experiment implements or evaluates this.

**Verdict:** **No evidence.** Stage 4 is a **design target**. Every component (command comprehension, reference, spatial, production) is **unvalidated** in the assembly + embodied setting.

---

## Stage 5: Dialogue (single robot + human/script)

**Tasks:** Q&A (“what is this?” → “cup”; “where is the block?” → “on the table”); clarification (“which block?” → “the red one”); follow-up (“put it on the table” → “it” = cup); turn-taking.

**Does it make sense?** Yes, as **multi-turn** coordination: comprehend question → produce answer; comprehend command with pronoun → resolve reference from prior turn.

**Evidence:**

- **NEMO:** Dialogue, interrogative generators, comprehension. So **dialogue structure** exists in code; not embodied.
- **No experiment** on (1) multi-turn state (e.g. “it” = last mentioned object), (2) question assembly → answer assembly → production, or (3) turn-taking policy (when to speak vs act).

**Questions:**

1. **Reference across turns (“it”).** Requires **memory** of last referent (e.g. “cup”). We have no validated “discourse state” or “focus” assembly that persists across turns. So cross-turn reference is **assumed**, not evidenced.
2. **When to ask vs act.** Pragmatics (ask for clarification when ambiguous). No experiment on when the system chooses to produce a question vs an action.

**Verdict:** **Partial.** Dialogue is supported **in NEMO** (structure, generators). **No evidence** for embodied multi-turn dialogue, reference resolution across turns, or pragmatic choice (ask vs act).

---

## Stage 6: Two robots, shared task

**Tasks:** Human instructs both robots; division of labor (“you take red, I take blue”); turn-taking (A says what it will do, B acts); reporting (“what did you do?” → “I picked up the cup”).

**Does it make sense?** Yes, as **perspective-taking** (you vs I) and **coordination** (who does what). MHC order 8–9.

**Evidence:**

- **None.** No multi-agent assembly experiment, no Isaac Lab, no “you”/“I” resolution, no division-of-labor task.

**Questions:**

1. **“You” and “I”.** Requires mapping “you” to other robot and “I” to self. That’s social grounding (pronoun → agent id). Not implemented or tested.
2. **Reporting past action.** “What did you do?” → “I picked up the cup” requires **memory** of own past action (or current state) and production. We have no experiment on episodic or action memory in assemblies.

**Verdict:** **No evidence.** Stage 6 is **fully aspirational**.

---

## Stage 7: Social dialogue (robot–robot)

**Tasks:** Robots coordinate by talking (“I’ll take the cup”; “can you pass the cup?”); describe (“the block is on the table”); answer each other’s questions; minimal human; conventions emerge.

**Does it make sense?** Yes, as **pragmatics** and **theory of mind** (what does the other need to know?). MHC order 9.

**Evidence:**

- **None.** No robot–robot dialogue, no coordination experiment, no emergence of conventions.

**Questions:**

1. **Emergence of conventions.** Curriculum claims “I’ll take X → other takes Y” can emerge from repeated interaction. That’s a **hypothesis**, not a result. We’d need many runs and a measure of “convention” (e.g. consistency of division of labor).
2. **Theory of mind.** “What does the other need to know?” (e.g. describe location) requires modeling the other’s knowledge. No experiment or implementation.

**Verdict:** **No evidence.** Stage 7 is **fully aspirational**.

---

## Stage 8: Open-ended

**Tasks:** Novel objects/phrases; few-shot (“this is a widget”); curiosity (“what’s that?”); information-seeking (ask partner, use answer); minimal script; “culture” emerges.

**Does it make sense?** Yes, as **generalization** and **meta** (ask when uncertain). MHC order 9+.

**Evidence:**

- **None.** No generalization experiment (novel object/word), no curiosity drive, no information-seeking dialogue.

**Questions:**

1. **“Curiosity” drive.** Curriculum assumes “unexpected perception → more processing → ask.” We have no implementation of “unexpected” or “drive to ask.”
2. **Few-shot new word.** “This is a widget” → robot uses “widget” in dialogue. That’s **fast mapping** (one or few trials). We have not tested whether one association trial (Stage 2 style) suffices for a new word in a multi-word system.

**Verdict:** **No evidence.** Stage 8 is **fully aspirational**.

---

## Cross-cutting gaps

1. **Encoder: continuous → assembly.** Vision, proprioception, language (audio/embedding) → assembly. Only **discrete** stimulus → assembly is in use in experiments. Curriculum assumes continuous→assembly for perception and possibly language; **not validated**.
2. **Readout: assembly → continuous.** Motor readout (assembly → joint torques/trajectory) and **production** readout (assembly → word/phrase) are **not** implemented or tested in the repo. Curriculum assumes both.
3. **Distinctiveness vs stability.** Multiple assemblies in one area require **stim-only** (distinct but persistence ~0.86) or **multi-area** (one area per concept). Curriculum does not specify; **must** be chosen and documented.
4. **Memory across time/turns.** “It” = cup, “what did you do?” = past action. No validated mechanism for **discourse state** or **episodic action memory** in assemblies.
5. **Isaac Lab / embodiment.** No experiment runs in Isaac Lab; no robot control loop; no embodied grounding. All of Stages 0–8 in Isaac are **future work**.
6. **MHC order vs task design.** We mapped stages to MHC orders, but we have **not** run tasks that are explicitly scored by MHC order (e.g. “order-5 task” vs “order-6 task”). So the MHC alignment is **conceptual**, not empirically validated.

---

## Recommendations

1. **Specify protocol for multiple assemblies.** In the curriculum doc, state explicitly: for multiple object/word assemblies in one area use **stim-only** (and cite persistence ~0.86), or use **multi-area** (one area per concept). Do not assume stim+self for multiple concepts in one area (evidence says it collapses).
2. **Add minimal validated Stage 0–1.** (a) **Stim-only** multi-assembly: e.g. 3–5 stimuli → one area, measure pairwise overlap (expect low) and persistence (expect ~0.86). (b) **Association** object→action: two areas (object, action) → association area; measure “see object → correct action” (or A→C recovery). Document as “evidence for Stage 0–1.”
3. **Add encoder and readout experiments.** (a) Simple encoder: continuous vector (e.g. object pose) → binned or hashed → stimulus index → assembly. (b) Simple readout: motor assembly → fixed primitive (e.g. one assembly = one trajectory). Run in simulation (no Isaac yet) to show feasibility.
4. **Treat Stages 3–8 as hypotheses.** Keep them in the curriculum as **targets**, but label clearly: “No evidence yet; requires validation.” Prioritize Stage 3 (two-word binding) and Stage 4 (one command → action) as first validation steps.
5. **Reconcile RESULTS_assembly_distinctiveness with current code.** RESULTS doc describes stim+self collapse; current test_assembly_distinctiveness.py uses stim-only. Either (a) update RESULTS to reflect stim-only (and distinctiveness success), or (b) add a separate “stim+self distinctiveness” result and state that curriculum uses **stim-only** for multi-concept areas.
6. **MHC:** Keep task analysis and MHC mapping for **design** (order of introduction, non-arbitrary coordination). Add later: tasks explicitly labeled by MHC order and a small battery that scores “highest order completed” to test MHC predictions.

---

## Conclusion

The curriculum **makes sense** as a sequence of tasks that fit the assembly calculus and MHC. **Evidence** strongly supports:

- Single-assembly stability and convergence (projection).
- Cross-area association (association recovery, identity preservation).
- Distinct assemblies in one area **under stim-only** (current distinctiveness test + competition_mechanisms).

Evidence is **missing** or **contradicted** for:

- Multiple distinct assemblies in one area **under stim+self** (contradicted: collapse).
- Continuous→assembly encoding and assembly→action/word readout (not implemented).
- Embodied execution of any stage (no Isaac/robot).
- Multi-turn dialogue, reference, perspective, social coordination, generalization, curiosity (no experiments).

So: the curriculum is **partially evidence-based** for Stages 0–2 **if** we restrict to **stim-only** (or multi-area) and treat encoder/readout as **to be built**. Stages 3–8 are **aspirational** and should be explicitly marked as such until we have at least one validated task per stage.
