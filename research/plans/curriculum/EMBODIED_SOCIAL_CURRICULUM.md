# Rich curriculum for embodied + social assembly cognition

**Goal:** Imagine a **rich curriculum** for the Isaac Lab + assembly-brain setup: embodied cognition (concepts grounded in doing), then language (with instruction), then multi-robot social cognition (dialogue, coordination). Stages build on each other; each stage has concrete activities, what gets taught, and how instruction is delivered.

This aligns with your existing NEMO curriculum (First Words → Two-Word → Telegraphic → Complex → Fluent) but **starts with embodied, pre-linguistic** stages and **ends with social dialogue** between robots.

---

## Overview: curriculum stages

| Stage | Name | Focus | Instruction source | Outcome |
|-------|------|--------|---------------------|---------|
| 0 | Sensorimotor bootstrapping | Perception + motor primitives, no words | None (exploration + RL) | Stable object & action assemblies |
| 1 | Affordances | Object–action binding (see cup → grasp) | None or minimal | Association area links object ↔ action |
| 2 | First words (embodied) | Nouns + verbs in context | Human/script + doing | Words grounded in visual + motor |
| 3 | Two-word & phrases | "Red block", "grasp cup", "pick up" | Human/script + doing | Phrases grounded in perception + action |
| 4 | Commands & simple sentences | "Pick up the red block", "Give me the cup" | Human/script + doing | Comprehension → action; production of intent |
| 5 | Dialogue (single robot) | Q&A, follow-up, clarification | Human or script | Robot answers questions, asks for clarification |
| 6 | Two robots: shared task | Turn-taking, division of labor | Human instructs both | "You take red, I take blue"; coordination |
| 7 | Social dialogue | Robots talk to each other, coordinate, ask for help | Minimal (emergent) | Shared lexicon, conventions, simple theory of mind |
| 8 | Open-ended | Novel objects, novel phrases, minimal script | Human + emergence | Generalization, curiosity, information-seeking |

---

## Model of Hierarchical Complexity (MHC) and curriculum structure

The **Model of Hierarchical Complexity** (MHC; Commons & Richards, 1980s; [Wikipedia](https://en.wikipedia.org/wiki/Model_of_hierarchical_complexity)) is a formal, task-based framework for scoring how complex a **behavior** or **task** is. It is not about mental structures (like Piaget’s stages) but about the **hierarchical complexity of the task** and the **order of actions** required to complete it. Key points:

### MHC in brief

- **Higher-order actions:** (a) are **defined in terms of** the next lower ones (hierarchy); (b) **organize** the next lower actions; (c) organize them in a **non-arbitrary** way (so it’s coordination, not just a chain).
- **Vertical complexity:** Number of **recursions/coordinations**: an order-N task operates on one or more order-(N−1) tasks and coordinates them. Example: “3 × (4 + 1)” requires coordinating addition and multiplication (distributive act) → one order more complex than addition or multiplication alone.
- **Tasks are quantal:** A task is either completed correctly or not; stage = **highest order of task successfully completed**. So curriculum must present tasks in **strict order**: you cannot succeed at order N without having completed and **practiced** order N−1.
- **Task vs performance:** MHC separates **task complexity** (what is demanded) from **performance** (what the participant does). Stage is defined as the most hierarchically complex **task** the participant has solved successfully.
- **Cross-domain:** Scoring depends on **how** information is organized (hierarchical recursion), not on **content**. So the same ordering principle applies to sensorimotor, language, and social tasks.

### MHC stages (orders 0–16, simplified)

| Order | MHC stage | What they do (summary) |
|-------|-----------|-------------------------|
| 0 | Calculatory | Exact computation only, no generalization |
| 1 | Automatic | Single hard-wired response to single stimulus |
| 2 | Sensory & motor | Discriminate stimuli, move; rote generalization |
| 3 | Circular sensory-motor | Form open-ended classes; reach, touch, babble |
| 4 | Sensory-motor | Form **concepts**; respond to class non-stochastically |
| 5 | **Nominal** | Find **relations among concepts**; **single words** (nouns, verbs, names) |
| 6 | **Sentential** | **Imitate sequences**; **chain words**; pronouns |
| 7 | Preoperational | Simple deductions; lists, stories; connectives (as, when, then) |
| 8 | Primary | Simple logical deduction; arithmetic; time sequence |
| 9 | Concrete | Full arithmetic; social rules; **perspective-taking** |
| 10+ | Abstract, formal, systematic, … | Variables, quantification, systems, metasystems, paradigms |

So: **0–4** = pre-linguistic (sensory-motor, concepts); **5** = first words (nominal); **6** = word chains / sentential; **7–9** = reasoning, sequences, social perspective; **10+** = abstract and postformal.

### Mapping our curriculum to MHC orders

| Our stage | Our focus | MHC order (approx.) | Rationale |
|-----------|------------|----------------------|-----------|
| 0 | Sensorimotor bootstrapping | 2–4 | Discriminate stimuli, form concepts (object/action assemblies); no words |
| 1 | Affordances | 4→5 | **Coordinate** two concepts (object + action) in a non-arbitrary way; still no names |
| 2 | First words (embodied) | 5 | Nominal: relations among concepts, **single words** (nouns, verbs) |
| 3 | Two-word & phrases | 6 | Sentential: **sequences**, chain words (red block, grasp cup) |
| 4 | Commands & simple sentences | 6–7 | Sequences of acts; simple deductions (“pick up the red block” = select + act) |
| 5 | Dialogue (single robot) | 7–8 | Lists of acts, turn-taking, time sequence (question → answer) |
| 6 | Two robots, shared task | 8–9 | Coordination of sequences; **perspective-taking** (you vs me) |
| 7 | Social dialogue | 9 | Social rules, coordination, what “others” do and know |
| 8 | Open-ended | 9+ | Generalization, variables, novel contexts |

So our curriculum **already follows MHC** in spirit: we start with sensory-motor and concepts (0–4), then nominal (first words), then sentential (phrases, chains), then preoperational/primary (commands, dialogue), then concrete (two agents, perspective, social). The main addition from MHC is to make the **task analysis** explicit and to enforce **strict order** and **practice**.

### Implications for how our curriculum should be structured

1. **Strict order of stages**  
   Do not introduce a task of MHC order N until the learner has **successfully completed** at least one task of order N−1. So: no first words (5) until concepts (4) and ideally affordances (4→5) are in place; no two-word (6) until single words (5) are stable; no dialogue (7–8) until commands/sentences (6–7) are in place; no two-robot coordination (9) until single-robot dialogue and perspective (8–9) are in place.

2. **Task analysis for each stage**  
   For each curriculum stage, **define the task** in MHC terms: what lower-order actions does it coordinate? Example: “Pick up the red block” = coordinate (a) discrimination of “red” and “block” (concepts), (b) selection of the referent (relation), (c) action “pick up” (concept + motor). So the **task** has order 6–7; the curriculum should only give this task after order-5 and order-6 tasks are mastered.

3. **Practice before advance**  
   MHC stresses that “less complex tasks must be **completed and practiced** before more complex tasks can be acquired.” So each stage needs enough **successful completions** (and variety of instances) before moving on. Our “exit conditions” should be operationalized as: “successfully completed at least X tasks of this order” (and optionally “with Y% accuracy”).

4. **Non-arbitrary coordination**  
   Higher-order actions must **organize** lower-order actions in a **non-arbitrary** way (e.g. “pick up the red block” specifies which object and which action in a fixed relation). So curriculum design should avoid arbitrary chains; each new stage should add a **proper coordination** (e.g. object + action → affordance; word + referent → naming; utterance + response → turn).

5. **Score task, not just response**  
   Stage = highest **task** complexity successfully completed. So we should **define tasks** (e.g. “comprehend ‘give me the cup’ and execute”) and score **task completion** (correct/incorrect), not just “did the robot move.” That keeps our curriculum aligned with MHC’s separation of task and performance.

6. **Horizontal vs vertical**  
   MHC distinguishes **horizontal** complexity (more bits, more yes/no questions) from **vertical** (more recursions/coordinations). Our curriculum should increase **vertical** complexity stepwise (new stage = new order of coordination). Within a stage, we can increase **horizontal** complexity (more objects, more words, more variation) without jumping order.

7. **Domain and décalage**  
   MHC allows that “performance stage is different task area to task area” (horizontal décalage). So a robot might be “sentential” in object–action binding but still “nominal” in spatial language. Our curriculum can have **parallel tracks** (e.g. object names vs action names vs spatial terms) as long as within each track we respect **order** (no order-N task before order-N−1 in that track).

**Summary:** Structure the curriculum so that (a) **stages = task orders** (we map our stages to MHC orders as above), (b) **strict order** (no skipping; practice until success at order N before order N+1), (c) **task analysis** for each stage (what lower-order actions are coordinated?), (d) **exit = successful task completion** at that order (with practice and variety), and (e) **non-arbitrary** coordination at each step. That is what MHC implies for how our curriculum should be structured.

---

## Stage 0: Sensorimotor bootstrapping (pre-linguistic)

**Goal:** Form stable **object assemblies** (from vision/touch) and **action assemblies** (motor primitives) without any language. The brain learns to perceive and act; no words yet.

**Activities:**
- **Object exploration:** Robot sees and touches objects (cup, block, ball, etc.) in Isaac. Encode visual + proprioceptive/contact state → sensory area; Hebbian + projection form object-specific assemblies (e.g. “cup” assembly = pattern that fires when cup is in view).
- **Motor primitives:** Robot executes reach, grasp, place, push, etc. Each primitive is an assembly (or a readout from a motor area). RL or scripted demos train the readout so that “reach” assembly → reach trajectory.
- **No instruction:** Curriculum is just **scenes** (which objects, where) and **action scripts** (do reach, do grasp). Reward: successful grasp, successful place, etc.

**What gets taught:** Nothing linguistic. **Concepts** = object assemblies + action assemblies. Rich variety: many objects (shapes, colors, sizes), many actions (reach, grasp, place, push, pull).

**Exit condition:** Robot can reliably “see” objects (correct assembly wins for correct object) and execute primitives (correct action from assembly). Association area can be trained next (Stage 1).

---

## Stage 1: Affordances (object–action binding, no words)

**Goal:** **Association** primitive: co-activate object assembly + action assembly in shared area → affordance (e.g. cup + grasp → “grasp the cup”). Still no language.

**Activities:**
- **Object–action pairing:** Present object (e.g. cup); robot executes grasp. Co-activate object assembly + grasp assembly in association area; Hebbian strengthens link. Repeat for (cup, grasp), (cup, place), (block, push), (ball, roll), etc.
- **Generalization:** Novel object similar to cup (e.g. mug) → does “grasp” or “place” generalize? Measure overlap of novel object assembly with cup assembly; test if same action wins.
- **No instruction:** Curriculum is **(object, action)** pairs. Reward: success (e.g. grasp cup → reward).

**What gets taught:** Affordances (which action goes with which object). **Not** words yet.

**Exit condition:** Given object in view, association area + motor readout produce correct action (e.g. see cup → grasp) without words. Sets up Stage 2 (word = label for object/action assembly).

---

## Stage 2: First words (embodied)

**Goal:** **Words** for objects and actions, grounded in perception and action. Instruction: human (or script) says “cup” while robot sees/touches cup; “grasp” while robot grasps. Nouns → visual grounding; verbs → motor grounding (your existing NEMO mapping).

**Activities:**
- **Naming objects:** Script: present cup; say “cup”. Robot’s language area receives “cup”; visual area has cup assembly. Association (language + visual) strengthens. Repeat for many objects (block, ball, plate, etc.).
- **Naming actions:** Robot performs grasp; say “grasp”. Motor assembly + language “grasp” co-activated. Repeat for reach, place, push, etc.
- **Comprehension check:** Say “cup”; present two objects (cup, block). Reward if robot selects/touches cup. Say “grasp”; reward if robot performs grasp.
- **Production (optional):** After many trials, prompt robot (e.g. show cup, ask “what is this?”) and read out word from language area. Train production with imitation or RL.

**What gets taught:** Vocabulary (nouns, verbs) with **grounding** (visual for nouns, motor for verbs). Rich set: many nouns, many verbs, varied contexts.

**Exit condition:** Robot comprehends single words (word → correct object or action). Optionally produces single words when prompted. Maps to your **First Words** stage.

---

## Stage 3: Two-word & phrases

**Goal:** **Phrases** that combine modifier + noun (“red block”) or verb + object (“grasp cup”) or verb + particle (“pick up”). Grounded in perception (which object is red? which is block?) and action (grasp which? pick up what?).

**Activities:**
- **Adjective–noun:** “Red block”, “blue cup”, “big ball”. Present multiple objects; say phrase; reward if robot selects correct object. Robot must bind property (red) + kind (block) in same assembly or in sequence.
- **Verb–object:** “Grasp cup”, “place block”, “push ball”. Robot hears phrase; executes action on correct object. Association: (verb assembly, object assembly) → shared area → motor readout.
- **Verb–particle / compound:** “Pick up”, “put down”. Ground in motor (pick up = reach + grasp + lift). Instruction: say “pick up” while robot does it.
- **Spatial (optional):** “Block on table”, “cup to the left”. Ground in visual/spatial area. Reward if robot interprets correctly (e.g. points to or moves to correct relation).

**What gets taught:** Two-word combinations; **binding** (which red? which block? grasp which object?). Rich: many adjectives, many nouns, many verbs, many objects.

**Exit condition:** Robot comprehends phrases (“red block” → selects red block; “grasp cup” → grasps cup). Optionally produces phrases. Maps to your **Two-Word** stage.

---

## Stage 4: Commands & simple sentences

**Goal:** **Full commands / simple SVO:** “Pick up the red block”, “Give me the cup”, “Put the ball on the table”. Comprehension → action; optionally production (“I will pick up the red block”).

**Activities:**
- **Comprehension:** Human/script gives command; robot executes. Varied objects, colors, locations, actions. Reward for correct execution.
- **Production of intent:** Before acting, robot “says” what it will do (read out from language area): “I will grasp the cup.” Then acts. Train with imitation (match expert utterance) or RL (reward if utterance matches action).
- **Spatial and relational:** “Put the block to the left of the cup”, “Give the red block to me” (if second agent or human avatar). Ground in spatial + social areas.
- **Negation (optional):** “Don’t grasp the cup”, “Pick up the block that is not red.” Requires binding negation; can be later.

**What gets taught:** Sentence-level comprehension and production; **reference resolution** (which block? which cup?); **intent** (say then do). Maps to your **Telegraphic / Simple sentences**.

**Exit condition:** Robot follows varied commands; optionally announces intent; resolves reference correctly.

---

## Stage 5: Dialogue (single robot + human/script)

**Goal:** **Turn-taking** with a partner (human or script): questions, answers, follow-up, clarification. Robot both comprehends and produces multi-turn dialogue.

**Activities:**
- **Questions:** “What is this?” (show object) → robot says “cup”. “What are you doing?” (robot grasping) → “grasp”. “Where is the block?” → “on the table”.
- **Answers:** Script asks; robot answers from assembly readout. Reward for correct answer (or match to script).
- **Clarification:** “Pick up the block.” Two blocks. Robot asks “Which block?” or “The red one?” Script answers; robot acts. Train clarification as production + comprehension.
- **Follow-up:** “I need the cup.” Robot brings cup. “Put it on the table.” “It” = cup (reference resolution across turns).

**What gets taught:** Q&A, reference across turns, clarification. **Pragmatics** (when to ask, when to act). Single robot; partner is human or script.

**Exit condition:** Robot holds short dialogues (3–5 turns); answers questions; asks for clarification when needed. Maps to your **Complex / Questions**.

---

## Stage 6: Two robots, shared task

**Goal:** **Two robots** in same Isaac world; **human instructs both**. Division of labor, turn-taking, shared goal. Language is used to coordinate (“you take red, I take blue”) and to report (“I have the cup”).

**Activities:**
- **Instruction to both:** “Robot A, pick up the red block. Robot B, pick up the blue block.” Each robot hears (or gets) its own instruction; executes. Reward for both succeeding.
- **Turn-taking:** “Robot A, say what you will do.” A says “I will grasp the cup.” “Robot B, you take the block.” B acts. Then swap. Train turn-taking (who speaks when) by script or reward.
- **Division of labor:** “One of you get the cup, one get the plate.” Robots must coordinate (e.g. A says “I’ll get the cup”, B gets plate) or script assigns. Reward for task success.
- **Report:** “Robot A, what did you do?” A says “I picked up the cup.” Ground in motor/memory (what A did). Train production of past action.

**What gets taught:** **Social roles** (A vs B), **coordination** (who does what), **reporting** (what I did). Shared lexicon (both know “cup”, “block”, “grasp”) from shared instruction.

**Exit condition:** Two robots complete shared tasks from single or multi-turn instructions; take turns; report actions. Sets up Stage 7 (robots talk to each other without human in the loop).

---

## Stage 7: Social dialogue (robot–robot)

**Goal:** **Robots talk to each other** with minimal human in the loop. Coordinate (“I’ll take the red one”), ask for help (“can you pass the cup?”), describe (“the block is on the table”), answer each other’s questions.

**Activities:**
- **Coordination:** Same scene; no central instruction. Robot A says “I’ll pick up the cup.” Robot B says “I’ll take the block.” They act. Reward for task success (e.g. both objects collected) and for utterances that match actions. Optionally reward for “helpful” utterances (B says “the cup is on your left”).
- **Request:** Robot A: “Can you pass the cup?” Robot B comprehends, passes cup. B can say “Here” or “Done.” Train request–response with reward for correct action.
- **Describe:** Robot A sees something B might not see clearly. A says “The red block is under the table.” B uses that to find block. Reward for B’s correct use of description.
- **Question–answer between robots:** A: “Where is the cup?” B: “On the table.” A goes to table. Reward for correct answer and correct use.
- **Minimal script:** Human only intervenes to start (“you two, clear the table”) or to correct. Most dialogue is robot–robot. **Conventions** emerge from repeated interaction (e.g. “I’ll take X” → other takes Y).

**What gets taught:** **Pragmatics** (request, offer, describe, answer); **theory of mind** (what does the other need to know?); **conventions** (how we coordinate). Shared lexicon and shared world; meaning emerges from use.

**Exit condition:** Two robots complete tasks by talking to each other; ask for and give help; describe and resolve reference; minimal human input. **Emergent** coordination and dialogue.

---

## Stage 8: Open-ended (generalization, curiosity)

**Goal:** **Novel objects**, **novel phrases**, **minimal script**. Test generalization; allow **curiosity** (robot asks “what’s that?” for novel object); allow **information-seeking** (robot asks partner for information and uses it).

**Activities:**
- **Novel objects:** Introduce objects not in training (e.g. new shape, new color). Does “the blue thing” or “the round one” generalize? Train with few-shot instruction (“this is a widget”) and see if robot uses “widget” in dialogue.
- **Novel phrases:** “Put the thing that’s next to the cup” (never seen “thing that’s next to”). Reward if robot resolves and acts correctly. Measure generalization of binding.
- **Curiosity:** Robot sees novel object; has drive to “resolve” (e.g. unexpected perception → more processing). Robot asks partner “what’s that?” Partner (robot or human) answers. Robot updates. Reward for asking when novel, for correct use of answer.
- **Information-seeking:** Robot doesn’t know where object is. Asks “where is the cup?” Other robot answers. Robot uses answer to act. Reward for correct use of information.
- **Minimal script:** No fixed curriculum; human or script provides occasional new words or corrections. Most interaction is robot–robot; **culture** (shared lexicon, conventions) evolves from interaction.

**What gets taught:** **Generalization** (new objects, new phrases); **curiosity** (ask about novel); **information-seeking** (ask and use answer). **Emergence** of robust, flexible dialogue and coordination.

**Exit condition:** Robots handle novel objects and phrases; ask questions when useful; use each other’s answers; sustain dialogue and coordination with minimal supervision. **Rich curriculum** is “done” in the sense that the system is now open-ended.

---

## How instruction is delivered (practical)

- **Stages 0–1:** No language. **Scripts** in Isaac: which objects, which actions, reward function. RL or self-supervised Hebbian.
- **Stages 2–4:** **Human-in-the-loop** (e.g. human says “cup” while robot sees cup) or **script** that plays utterances + scene/action. Script can sample from your NEMO curriculum (GroundedSentence, grounding contexts) and map “scene” to Isaac state (which object, where) and “action” to robot primitive.
- **Stages 5–6:** **Human or script** as partner. Script can use your dialogue generators (declarative, interrogative) and map to Isaac (e.g. “what is this?” → show object; robot’s answer = readout from brain).
- **Stages 7–8:** **Minimal human.** Robots are each other’s partners. Human only for new vocabulary, corrections, or rare prompts. Instruction = “what you can say/hear” plus reward for task success and (optionally) for helpful dialogue.

---

## Mapping to your existing NEMO curriculum

| This curriculum (embodied) | Your NEMO stages | Notes |
|---------------------------|------------------|--------|
| Stage 0–1 | (pre-linguistic) | No NEMO equivalent; embodied only |
| Stage 2 | First Words | Nouns + verbs; grounding from Isaac (visual, motor) |
| Stage 3 | Two-Word | Adjective–noun, verb–object; grounding from Isaac |
| Stage 4 | Telegraphic / Simple | SVO, commands; comprehension → action |
| Stage 5 | Complex (Q&A) | Questions, answers, clarification; single robot |
| Stage 6–7 | (social) | Two robots; coordination, dialogue; extends Fluent |
| Stage 8 | Open-ended | Generalization, curiosity; beyond fixed curriculum |

Your **vocabulary** (nouns, verbs, adjectives, etc.) and **grounding** (visual, motor, spatial, social) map directly: e.g. `GroundedSentence` with `GroundingContext(visual=..., motor=...)` becomes “show this object / do this action in Isaac while saying this sentence.” Your **generators** (declarative, interrogative) supply sentence types; the **rich** part is the **embodied schedule**: when do we do object exploration, when do we add words, when do we add two robots, when do we let dialogue emerge.

---

## Summary

A **rich curriculum** for embodied + social assembly cognition:

1. **Starts with body and world** (sensorimotor, affordances), then **adds language** (first words, phrases, commands), then **adds social** (two robots, dialogue, coordination).
2. **Instruction** moves from none (Stage 0–1) to heavy (Stages 2–4: naming, phrases, commands) to dialogue with human/script (Stage 5–6) to **robot–robot with minimal human** (Stage 7–8).
3. **What gets taught:** object and action assemblies → affordances → words (grounded) → phrases → sentences → dialogue → social coordination → generalization and curiosity.
4. **Outcome:** Embodied, language-capable, socially coordinated agents whose concepts and conventions **emerge** from the same assembly calculus, at **realistic rates** of control and speech.

This is the curriculum you could implement in Isaac Lab + your assembly brain to get from “blank robot” to “embodied, speaking, coordinating” agents.
