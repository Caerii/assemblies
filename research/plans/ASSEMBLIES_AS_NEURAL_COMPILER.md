# Assemblies as neural compiler, neural programming language, and computational abstraction

**Goal:** Dive deeper into the idea that the assembly calculus is a **neural programming language**, the **target** of a **neural compiler**, and a **higher-level computational abstraction** above raw neurons. This doc expands the conceptual exploration: formal structure, execution semantics, program vs data, learning as recompilation, control flow, abstraction hierarchy, and implications.

---

## 1. Assembly calculus as a formal language

### 1.1 Primitives and composition

**Primitives:**
- **Project** (A → B): Activity in area A drives activity in area B via the A→B connectome; then top-k in B. So: "send A's assembly to B; B selects k winners."
- **Associate** (A, B → C): Co-activate A and B; Hebbian strengthens (A,B)→C; then project (A,B) to C so C comes to represent the *binding* of A and B.
- **Merge** (A, B → C): Co-activate A and B; form C as an assembly that *overlaps* both (e.g. by projecting A and B into C and taking top-k of the *summed* input). So C "contains" both A and B.
- **Select** (implicit): Within each area, top-k is the selection rule — "choose the k most active." So every area is a "k-winner" module.

**Composition:**
- **Chaining:** A → B → C (project A to B, then B to C). So we can build **pipelines**.
- **Branching:** A → B and A → C (A drives both B and C). So we can **fan out**.
- **Convergence:** A → C and B → C (both A and B drive C). So we can **fan in** (e.g. merge or association).
- **Recurrence:** A → A (area projects to itself). So we get **loops** — activity circulates until (under conditions) it reaches a **fixed point**.

So the "program" is a **graph**: nodes = areas (or assembly "registers"); edges = connectomes (who projects to whom); and we have two kinds of "multi-input" edges: **association** (A,B → C, with Hebbian) and **merge** (A,B → C, sum then top-k). **Execution** = run the dynamics: at each step, each area receives input from its predecessors (according to the graph), sums (and optionally applies Hebbian), then applies top-k. Repeat until fixed point or for a fixed number of steps.

### 1.2 Syntax and semantics

**Syntax (well-formed "programs"):**
- A **program** is a directed (multi)graph: vertices = areas; edges = projections (possibly with labels: "project," "associate," "merge"). Optionally we allow **stimulus** nodes (external input) and **readout** nodes (where we read the result).
- **Well-formed** means: every area has a well-defined update rule (top-k of incoming input; for association/merge, incoming from two or more sources with the right Hebbian or sum semantics).

**Semantics (execution):**
- **State** = assignment of a **set of k winners** (or an activation vector) to each area.
- **Step** = for each area, compute input (from predecessors according to the graph), apply Hebbian if the edge is associative, then set state = top-k(input).
- **Execution** = sequence of steps. We can define **termination**: e.g. "run until fixed point" (state unchanged for one step) or "run for T steps."
- **Result** = state at termination (or at a designated readout area).

So we have a **denotational** semantics: "program" + "initial state" (and optionally "stimulus over time") ⇒ "final state" (or trajectory). The **operational** semantics is the dynamics (repeated application of project/associate/merge + top-k).

### 1.3 Fixed-point semantics

A key fact: under suitable conditions (connectivity, sparsity, Hebbian structure), the dynamics **converge** to a **fixed point** — a state where one more step leaves the state unchanged. So we can give a **fixed-point semantics**: the "meaning" of running the program on input X is the **unique** (or one of the) fixed points reachable from the initial state that encodes X.

That is analogous to **recursive** or **iterative** programs in classical CS: "keep applying the update until nothing changes" = **while loop** or **recursion** until base case. So the assembly "program" has a clear **denotation**: the fixed point (the "answer"). The **runtime** is the number of steps to reach the fixed point — which you've observed is O(log n) in many regimes.

---

## 2. Program vs data: what is "the program"?

In classical computing we distinguish **program** (the code) and **data** (what the code operates on). In the assembly calculus, the boundary is blurry — and that's instructive.

**Option A: Connectome = program, activity = data**
- The **program** is the **wiring**: which areas exist, which connectomes exist (and their weights W). That's the "code" — it defines *what* projections and associations are possible.
- The **data** is the **current activity**: which assemblies are active (the k winners in each area). So at each moment, "data" = the current state; "program" = the fixed structure (W, topology).
- **Execution** = the program (connectome) acts on the data (activity) to produce new data (updated activity). So the program is **fixed** (or changes slowly via Hebbian); the data **flows** through the program.

**Option B: Sequence of operations = program, assemblies = data**
- The **program** is the **sequence** (or graph) of operations we *choose* to run: "first project LEX to LEX, then project LEX to SEM, then associate SEM and MOTOR to get ACTION." That's the "high-level" program — a recipe of operations.
- The **data** is the assemblies (the concepts): LEX, SEM, MOTOR, ACTION are "variables" holding assembly states.
- The **compiler** then maps this high-level program to the **connectome** (which areas, which W) that *implements* those operations. So the connectome is the **compiled object code**; the sequence of operations is the **source program**.

**Option C: Unified (program = data)**
- In Lisp, code is data (S-expressions). In the assembly calculus, we could say: the **same** substrate (neurons, areas, connectomes) holds both "which operations are available" (the connectome) and "what is currently being computed" (the activity). So **program and data are the same kind of thing** — patterns of connectivity and activity. Hebbian learning **changes the program** (the connectome) based on **data** (co-activation). So we have **self-modifying code**: execution (activity) modifies the program (connectome) via Hebbian. That's a **runtime recompilation** or **JIT** view: the "program" evolves with experience.

All three views are useful. **A** is the "machine code" view (connectome = the thing that runs). **B** is the "compiler" view (high-level program → compile to connectome). **C** is the "self-modifying / learning" view (program and data are the same substrate; learning = recompilation).

---

## 3. Learning as recompilation

When we learn (Hebbian), we **change** the connectome W. So we're **changing the program** that will run on future inputs. That is exactly **recompilation**: the "program" (which associations exist, how strong the projections are) is **rewritten** by execution (co-activation). So:

- **Compile time** (design time) = initial connectome (e.g. random Bernoulli).
- **Runtime** = activity flows; Hebbian updates W.
- **Recompilation** = each time we run, we *optionally* update W (Hebbian), so the "program" for the *next* run is slightly different.

So the **neural compiler** isn't just "map high-level to connectome once." It's **continuous**: experience (data) **recompiles** the program (connectome). That's a **learning compiler** or **adaptive compiler** — the target "code" (the connectome) is updated by the execution traces (which assemblies co-occurred). That's a deep analogy: **learning = recompilation**; the brain is both the **compiler** (it rewrites its own connectome) and the **runtime** (it runs the dynamics on that connectome).

---

## 4. Areas as modules / functions

Each **area** takes **inputs** (from other areas, or from stimulus) and **produces** an **output** (the k winners). So each area is like a **function** or **module**:

- **Signature:** inputs = activity from predecessor areas (and optionally stimulus); output = k winners (the "return value").
- **Body:** the "body" of every area is the same **generic** rule: sum incoming weighted input, then top-k. So the "code" of each area is **identical** (top-k); what differs is the **connectome** (who sends to whom, and with what weights). So the "program" is entirely in the **graph** and the **weights**, not in the "instruction set" of an area. That's like a **dataflow** or **connectionist** program: the "functions" are trivial (sum + top-k); the **graph** is the program.

**Projection A → B** is then like **"call B with argument A"**: B receives A's output as input and returns its own k winners. **Association A,B → C** is like **"call C with arguments A and B"**: C receives both and (with Hebbian) forms a binding. So the "program" is a **graph of function calls** (projections); the "functions" (areas) are all the same "instruction" (top-k of input), and the **data** (assemblies) flows along the edges.

**Recursion:** When A → A (self-projection), we have **recursive call**: A's output is fed back as A's input. So A is a **recursive function** — and the **fixed point** is the "return value" of that recursion (the base case, in a sense: when the output equals the input, we're done).

---

## 5. Control flow: conditionals, loops, sequencing

Classical programs have **control flow**: conditionals (if/else), loops (while, for), sequencing (do A then B). How do these show up in the assembly calculus?

**Sequencing:** Natural. We **order** operations in time: first project A → B, then project B → C. So "do A then B" = run one step (A→B), then the next (B→C). The **temporal order** of projections is the **sequence**. In a multi-area graph, we could run all areas in parallel (synchronous update) or in a prescribed order (asynchronous). So **sequencing** is either **explicit** (we choose the order of which projection to apply when) or **implicit** (the graph has a topological order; we run in that order).

**Loops:** Recurrence. When an area projects to itself (A → A), we're in a **loop**. We "exit" the loop when we hit a **fixed point** (state doesn't change). So **"repeat until stable"** is the **only** loop we need — and it's exactly what the dynamics do. So the assembly calculus has **built-in loops** (recurrence) with **built-in termination** (fixed point). We don't need an explicit "while" — the dynamics *are* the loop.

**Conditionals:** Harder. "If assembly X is active, then project to Y; else project to Z." That requires **gating** or **branching** based on state. Options:
- **Competition:** Have two "paths" Y and Z; both receive input from X (or from a common predecessor). Whichever assembly (Y or Z) is "closer" to the current state (or wins a competition) effectively gets selected. So **conditionals** could be implemented by **competition** between alternative projections — the "winner" is the branch taken.
- **Gating area:** An area G that "decides" which path to activate (e.g. G receives X; if G's assembly is "true," activate path to Y; if "false," activate path to Z). That requires G to have learned two assemblies (true/false) and the rest of the graph to use G's activity to gate. So **conditionals** = gating by a "decision" assembly.
- **Multiplicative gating:** If we had a mechanism where activity in X *modulates* the strength of the Y vs Z projection (e.g. X active → strengthen A→Y, weaken A→Z), we'd have soft gating. That's not in the basic calculus but could be an extension.

So **conditionals** are the one construct that isn't primitive — they emerge from **competition** or **gating** if we add a little structure. **Loops** and **sequencing** are native.

---

## 6. Higher-level language above assemblies

If the assembly calculus is the **target** (the "assembly language" or "bytecode"), what would a **higher-level** neural programming language look like?

**Possible constructs:**
- **Variables** = assembly names (e.g. `dog`, `run`, `subject`, `verb`). So we have **named** assemblies.
- **Statements:** `activate(dog)`, `project(LEX, SEM)`, `associate(subject, verb, clause)`, `merge(A, B, C)`.
- **Control flow:** `repeat until stable { project(LEX, LEX) }`, `if active(question) then project(SEM, QUERY) else project(SEM, DECL)`.
- **Abstraction:** `def recognize_object(sensory): ... return object_assembly` — a "function" that is really a subgraph (a pattern of projections) that maps sensory input to an object assembly. So we have **procedural abstraction** — name a subgraph and "call" it.

**Compilation:** The high-level program (a sequence of statements with control flow) is **compiled** to:
- A **graph** of areas and connectomes (the "object code").
- A **run-time** that executes the graph (run dynamics until fixed point, or for T steps; optionally apply Hebbian on association/merge edges).

So the **compiler** (the thing that takes high-level to assembly calculus) could be:
- **Hand-written:** We design the graph and the schedule (which projections when) to implement the high-level program.
- **Learned:** A meta-system (e.g. another network, or RL) learns to produce the graph and/or schedule from high-level specs (e.g. "parse this sentence" → produce a sequence of project/associate that implements parsing).

The **assembly calculus** is then the **instruction set** and **runtime** of this higher-level language. The higher-level language is the **cognitive** or **symbolic** layer — goals, plans, parse trees, propositions — and the assembly calculus is the **machine** that runs it.

---

## 7. Abstraction hierarchy

We can stack abstractions:

**Level 0: Neurons**  
Spikes, synapses, membrane potentials. The "hardware."

**Level 1: Assemblies and the calculus**  
Assemblies = groups of neurons that act as units. Operations = project, associate, merge, select. This is the **first** computational abstraction above neurons — the "assembly language" for the brain. **Programs** at this level = graphs of areas and connectomes; **execution** = dynamics to fixed point.

**Level 2: Cognitive / symbolic operations**  
"Recognize," "bind," "infer," "plan step," "parse." Each of these is implemented by **one or more** assembly operations (a subgraph). So level 2 is **high-level** statements; they **compile to** level 1 (assembly operations). **Programs** at this level = cognitive programs (e.g. "see object → recognize → plan grasp → execute"); **execution** = run the compiled assembly graph.

**Level 3: Goals, propositions, full sentences**  
"The ball is red." "Pick up the cup." These are **data** or **specifications** that level 2 operations operate on. They might **compile to** level 2 (e.g. "pick up the cup" → plan step 1, 2, 3; each step is a level-2 operation), and level 2 compiles to level 1. So we have a **tower**: Level 3 (goals, language) → Level 2 (cognitive ops) → Level 1 (assembly calculus) → Level 0 (neurons).

**Assemblies** are the **critical** level: they're the first level where we have **discrete** computational units (assemblies) and **compositional** operations (project, associate, merge). Below that, we have continuous dynamics; above that, we have "symbolic" or cognitive structure that **maps down** to assemblies. So **assemblies are the neural programming language**; the levels above are **languages that compile to it**.

---

## 8. Type-like structure and compositionality

**Types:** In programming languages, types constrain what can be combined (e.g. "function from int to int"). Do assemblies have **types**?

- **Areas as types:** We could say area LEX has "type" = lexical assembly; area SEM has "type" = semantic assembly; area MOTOR has "type" = motor assembly. So **projection LEX → SEM** is like a function from "lexical" to "semantic" type. **Association** (A, B → C) might require A and B to be of compatible types (e.g. subject and verb) and C to be of a "bound" type (e.g. clause). So we have **type-like** structure: not enforced by a type checker, but by the **topology** (we only have connectomes from certain areas to certain areas) and by **learning** (Hebbian only strengthens co-occurring patterns that "make sense").
- **Compositionality:** Merge and association are **compositional**: the result (C) "contains" or "binds" A and B. So the **type** of C could be thought of as the **product** or **binding** of the types of A and B. That's **compositionality** at the type level: complex types are built from simpler types by association/merge. So we have a **type structure** that mirrors the **compositional** structure of the operations — and that's exactly what you need for a **compositional** semantics (e.g. for language: the meaning of "red ball" is the merge/binding of "red" and "ball").

**Category theory:** We could formalize this. **Objects** = areas (or assembly types). **Morphisms** = projections (or connectomes). **Association** could be a product-like operation (A × B → C); **merge** could be a coproduct-like or sum-like operation. So the assembly calculus might have a **categorical** structure — and then we could prove things about compositionality and expressiveness. That's a possible direction for theory.

---

## 9. Compiler phases (if we take the metaphor seriously)

If we have a **high-level** neural programming language (level 2 or 3) that compiles to the assembly calculus (level 1), what are the "phases" of the compiler?

1. **Parsing:** The high-level program (e.g. a goal, a sentence, a plan) is **parsed** into an abstract representation (e.g. a tree or a graph of cognitive operations). So we have a **syntax** for the high-level language.
2. **Semantic analysis:** Check that the operations are well-formed (e.g. the right "types" of assemblies are combined). That might be trivial if we don't have strict types; or it could correspond to "is this plan feasible?" or "does this parse make sense?"
3. **Optimization:** Can we **optimize** the graph? E.g. eliminate redundant projections, or reorder operations so we converge faster. That's like compiler optimization — same semantics, cheaper execution.
4. **Code generation:** Map the high-level representation to a **graph of areas and connectomes** (and optionally a **schedule** — which projections to run when). That's the **codegen** phase: produce the "object code" (the connectome and the execution plan).
5. **Runtime:** Execute the object code (run the dynamics). Optionally **recompile** (Hebbian) during execution — so the "runtime" includes the learning compiler that updates the connectome.

So the **neural compiler** has the same **logical** structure as a classical compiler: parse → analyze → optimize → codegen → run. The **target** is the assembly calculus; the **source** is the high-level cognitive/symbolic program.

---

## 10. Turing completeness and expressiveness

**Question:** Can the assembly calculus compute everything a Turing machine can?

- We have **finite state** (k winners per area, finite number of areas) — so **bounded** state. A Turing machine has **unbounded** tape. So with **finite** areas and **fixed** k, we might **not** be Turing complete — we're more like a **finite automaton** or a **bounded** recurrent network.
- If we allow **unbounded** context (e.g. we can add new areas as we process longer input, or we have external memory that can grow), we might get **Turing completeness**. Or if we allow **unbounded** recurrence (the fixed point could depend on arbitrarily long history in principle), we might get more expressiveness.
- Even if **not** Turing complete, we might be **expressively powerful** for **cognitive** tasks: language (which might not need full Turing power for comprehension and production), planning (bounded depth), perception-action loops. So the **practical** question is: is the assembly calculus **expressive enough** for the tasks we care about? That might not require Turing completeness — it might require "enough" compositionality and recursion for language and planning.

So: **Turing completeness** is an open (and interesting) theoretical question; **expressiveness** for cognitive tasks is the more relevant empirical question.

### 10.1 What the Turing simulations (`turing_simulations.py`) actually test

The codebase has two simulations that probe **Turing-machine-like** behavior: **larger_k** and **turing_erase**. They don't implement a full TM; they test **primitives** that a TM would need.

**larger_k (lines 17–69):**
- **Setup:** Stimulus → Area A (small assembly, k); Area B (larger assembly, bigger_factor × k). A projects to B; B projects to A; A has no self-plasticity, B has self-plasticity.
- **Phase 1:** Stim → A, then A → A until stable. So A forms a **stable assembly** (attractor).
- **Phase 2:** Stim → A, then **reciprocal** A ↔ B until stable. So A and B **interact**; B gets "written" by A (and vice versa).
- **Measured:** Overlap of A before vs after interacting with B. So: **does A's identity persist** when A is coupled to a **larger** area B?
- **TM interpretation:** We're testing **heterogeneous "cells"** — different areas can have different **k** (different "symbol capacity" or "cell size"). In a TM, tape cells hold symbols; here "cell" = area, "symbol" = assembly pattern. So **larger_k** probes: can we have **different-sized** assemblies (like different "symbol sets" or "cell capacities") and still get stable interaction? And does the **smaller** assembly (A) retain its identity when coupled to a **larger** one (B)? That matters if we want **state** (small, precise) vs **tape** (larger, more robust?) to coexist.

**turing_erase (lines 72–145):**
- **Setup:** Stimulus → A; areas A, B, C (all same size, bigger_factor × k). A projects to B and C; B and C project back to A; A has weak self-plasticity.
- **Phase 1:** Stim → A, A → A until stable. So A forms a **stable assembly**.
- **Phase 2:** Stim → A, **reciprocal A ↔ B** until stable. So we **"write"** B with A's pattern (A drives B; B drives A; they settle). We save **A_after_proj_B** (A's state after coupling with B).
- **Phase 3:** Stim → A, **reciprocal A ↔ C** until stable. So we **"write"** C with A's pattern, and A **changes** (A is now coupled to C instead of B). We save **A_after_proj_C**.
- **Phase 4:** **Read B** without stimulus: project A → B (empty stimulus). So we "read" B by projecting current A into B. We compare **B_after_erase** (B's state after we've "left" B and done A↔C) to **B_before_erase** (B's state from Phase 2). So: **did B "remember" what we wrote, or did B get "erased"** when we did A↔C?
- **Measured:** (1) Overlap(B_after_erase, B_before_erase) — **tape persistence**: does B retain its content when we're not writing to it? (2) Overlap(A_after_proj_B, A_after_proj_C) — how much did A change when we switched from B to C?
- **TM interpretation:** **Write** = project A → B (B takes on a pattern influenced by A). **Leave** = stop projecting to B, do other computation (A↔C). **Read** = project A → B again (or observe B's saved state). For a TM we need: **non-volatile tape** — when we move the head away, the cell must still hold what we wrote. So **turing_erase** is a **direct test** of **tape persistence**: after "writing" to B and "leaving" (doing A↔C), does B still hold the old pattern? **High** B_overlap ⇒ B "remembers" (non-volatile). **Low** B_overlap ⇒ B "forgot" or got overwritten (volatile or interference). So the simulation is probing **memory** and **erasure** — the core tape operations of a TM.

### 10.2 What a Turing machine needs (and how assemblies map)

| TM requirement | Assembly analogue | What we have / need |
|----------------|-------------------|----------------------|
| **Unbounded tape** | Unbounded memory | **Bounded** if we have finitely many areas. **Unbounded** if: (a) **tape = time** — one area's **history** over time (unbounded sequence of states); or (b) **tape = space** — unbounded number of areas (dynamic growth). |
| **Read current cell** | Observe area's winners | **Have:** We can read an area's k winners (the "symbol" at that "cell"). |
| **Write to current cell** | Project to area | **Have:** Project assembly X → area B makes B's state depend on X. So we can "write" a pattern to B. |
| **Move left/right** | Change "current cell" | **Space tape:** "Current cell" = which area we're reading/writing. "Move" = switch to neighbor area (need a **linear chain** of areas: … T_{i-1}, T_i, T_{i+1} …). **Time tape:** "Current cell" = state at time t. "Move" = advance t (and optionally write by setting state at t+1). So we **have** "move" if we use time as the tape dimension. |
| **Finite state control** | Finite set of assemblies (or one "state" area) | **Have:** Finitely many areas, each with k winners ⇒ finite state. We can designate one area (or a set) as "state" and the rest as "tape." |
| **Conditional transition** | If state=X and tape=Y then write Z, move, go to state W | **Need:** We must **choose** which projection runs based on **current** state and **current** tape symbol. That's **conditional** projection — which we said requires **gating** or **competition**. So we **don't** have it as a primitive; we need an extension (gating area, or competition between alternative projections). |

So from the simulations we **have**: **write** (project), **read** (observe winners), **heterogeneous areas** (larger_k), and a **test of tape persistence** (turing_erase). We **don't yet have**: **unbounded tape** (we have finite areas) and **conditional transition** (we need gating).

### 10.3 Two paths to Turing completeness

**Path A: Tape = unbounded number of areas (spatial tape)**  
- **Tape** = a **linear chain** of areas T_0, T_1, T_2, … (unbounded: we can add T_{i+1} when we "move right" past the end).  
- **State** = one (or a few) areas with a finite set of assemblies encoding TM state.  
- **Read** = project "head" (current state + current tape area T_i) → readout that gives "symbol at T_i."  
- **Write** = project "state + new symbol" → T_i (overwrite T_i).  
- **Move** = change "current" from T_i to T_{i-1} or T_{i+1} — i.e. **which** area we project to next. That requires **conditional** projection: if state says "move right," next projection targets T_{i+1}; if "move left," target T_{i-1}. So we need **gating** (or a meta-controller that selects which area is the "current" tape cell).  
- **Unbounded:** We need to **create** new areas (new tape cells) when we move right past the end. So we need **dynamic topology** — add area T_{max+1} when needed. That's a significant extension (the "program" or runtime must be able to create new areas).

**Path B: Tape = unbounded time (temporal tape)**  
- **Tape** = the **sequence** of states of **one** area (e.g. T) over time: T(0), T(1), T(2), … So the "tape" has **unbounded length** (unbounded time steps).  
- **State** = one area S with finite assemblies (TM state).  
- **Read** = at time t, the "symbol at current position" is T(t). We **have** that (current winners of T).  
- **Write** = at time t+1, we set T(t+1) by **projecting** something (determined by state and T(t)) to T. So the "write" is: state S(t) + tape symbol T(t) → compute next state S(t+1), next symbol T(t+1), and "move" (which is implicit: we always "move" to t+1).  
- **Conditional transition** = we need: **if** (S(t), T(t)) = (q, a) **then** project assembly encoding (q', b, L/R) so that S(t+1) = q' and T(t+1) = b (and "move" = L or R). For **temporal** tape, "move left" = use T(t-1) instead of T(t) as "current symbol" (so we need to **store** T(t-1) somewhere — e.g. another area "prev_T") or "move right" = advance to t+1 (default). So "move left" in a temporal tape is **non-trivial** — we need to **remember** the previous tape symbol. That might require **two** areas: T_curr and T_prev, and we "shift" (T_curr → T_prev, new write → T_curr) each step. Then "current symbol" = T_curr; "move right" = shift and write new symbol to T_curr; "move left" = shift so T_prev becomes T_curr (and we need to "write" back to T_prev — so we need a **stack** or **two stacks** to simulate a tape with move left/right in time). Actually, a **tape** with left/right moves is hard to simulate with **only** time (time only goes forward). So we need either **spatial** tape (Path A) or **two stacks** (two areas that we push/pop) to simulate a tape. Two stacks can simulate a tape (classic result). So **Path B variant:** tape = **two stacks** (two areas, each holding a **sequence** of symbols — we push/pop by projection?). That gets messy. Simpler: **Path A** (spatial tape, unbounded areas) or **Path B with no move** (tape = one-way infinite sequence in time; TM with **no left move** = finite automaton with output, or we allow "move left" by storing history in **another** area — so we have "tape so far" in one area's **history** and "current position" = we're at the end; "move left" = we need to retrieve the previous symbol from history, which requires either unbounded state or another tape). So the cleanest path to Turing completeness is **Path A**: unbounded **spatial** tape (unbounded areas) + **conditional** projection (gating).

**Summary:**  
- **turing_erase** and **larger_k** show we have **write**, **read**, **persistence test**, and **heterogeneous k**.  
- For **Turing completeness** we need: (1) **Unbounded tape** — either unbounded areas (Path A) or a clever encoding with time + stacks (Path B). (2) **Conditional transition** — gating or competition so that "if state=X and symbol=Y then …" can be implemented.  
- The simulations are **first steps**: they test **memory persistence** (does B retain what we wrote?) and **heterogeneous capacity** (can A and B with different k coexist?). If B_overlap in turing_erase is **high**, we have **non-volatile memory** — a necessary (but not sufficient) condition for a TM tape. If **low**, we have a **problem** (tape "forgets") or we need a different encoding (e.g. write to a **different** area each time, and use **time** as the address — so tape = sequence of areas T_1, T_2, … and we only write to T_t at time t; then we never "overwrite" and we don't need persistence of one cell across time, we need unbounded areas).

### 10.4 What to test next (implications for experiments)

1. **Report and interpret turing_erase overlaps** — Run turing_erase and record B_overlap (B_after_erase vs B_before_erase). **High** overlap ⇒ B "remembers" (non-volatile) ⇒ one necessary condition for tape-like memory holds. **Low** overlap ⇒ B is volatile or A→B projection when stimulus is empty overwrites B; then we need a design where "read" doesn't overwrite (e.g. read = copy B → readout area without projecting back to B).  
2. **Conditional projection (gating)** — Design a minimal experiment: two "state" assemblies (q0, q1) and two "tape symbol" assemblies (0, 1). Implement "if q0 and 0 then write 1 and go to q1" by **gating**: e.g. an area that receives (q0, 0) and projects to "next state" q1 and "write" 1. So we need **association** (q0, 0) → (q1, 1) and a way to **route** the output to the right "tape cell" (which requires knowing "current cell" — so we need either spatial tape with current index, or temporal tape). That's the **minimal** TM step; implementing it in assemblies would be a concrete "conditional transition" test.  
3. **Unbounded tape** — For Path A: implement **dynamic area creation** (add a new area when "move right" past the end). For Path B: implement **two stacks** (two areas; push = merge/write, pop = project to readout?) and show we can simulate a tape. Both are significant extensions.  
4. **Formal claim** — Once we have (i) non-volatile memory (turing_erase B_overlap high, or design that avoids overwrite on read), (ii) conditional transition (gating), and (iii) unbounded tape (dynamic areas or two stacks), we could **claim** Turing completeness (with a written construction). Until then, the simulations **support** that we have **tape-like primitives** (write, read, persistence test) and **heterogeneous** areas; they don't yet give Turing completeness.

### 10.5 Bottom line for Turing completeness

- **turing_simulations.py** probes **TM-like primitives**: **larger_k** = heterogeneous assembly sizes (state vs tape capacity?); **turing_erase** = **write** (A→B), **leave** (A↔C), **read** (A→B again) and **persistence** (does B remember?).  
- For **Turing completeness** we need: **unbounded tape** (unbounded areas or time+stacks), **conditional transition** (gating), and **non-volatile tape** (B_overlap high, or read that doesn't overwrite).  
- The simulations **imply**: we're testing the **right things** (memory, erase, persistence, heterogeneous k). Next steps: (1) interpret and report B_overlap and A_overlap from turing_erase; (2) implement **gating** (conditional projection) in a minimal TM-step experiment; (3) decide on **tape encoding** (spatial vs temporal) and implement unbounded tape if we want a full Turing-completeness result.  
- Even **without** Turing completeness, the calculus can be **expressively powerful** for cognitive tasks; Turing completeness would be a **theoretical** result that sharpens the "neural programming language" claim.

---

## 11. Relation to other "neural programming" ideas

- **Neural Turing machines (NTM):** External memory + controller. The assembly calculus doesn't have an explicit "tape," but we could **add** an external memory (e.g. a store of assembly states at different "addresses") and let the dynamics **read/write** via projection. So we could **embed** something like NTM in the assembly calculus by adding memory areas and projections to/from them.
- **Differentiable programming:** Programs that are differentiable so we can backprop. We've discussed STE for assemblies — so we can make the assembly calculus **differentiable** (in the backward pass) and then we have "differentiable assembly programs." So we're in the same spirit as differentiable programming: **programs** (graphs of operations) that are **trainable**.
- **Program synthesis:** Automatically generate programs from specs. The "neural compiler" could be a **program synthesizer**: given a high-level spec (e.g. "parse sentences"), **synthesize** the assembly graph (and/or the connectome) that implements it. That could be done by **learning** (e.g. RL or gradient-based) to produce the graph/weights that achieve the spec.

So the assembly calculus sits in the same **conceptual space** as these ideas — it's a **constrained** (discrete, local, fixed primitives) but **compositional** neural programming substrate.

---

## 12. Summary: implications

1. **Assembly calculus = target language / IR** — The "machine code" or "assembly language" for neural computation. Primitives (project, associate, merge, select), composition (graphs), execution (dynamics to fixed point).
2. **Program = connectome (and/or sequence of ops); data = activity** — With learning = recompilation (Hebbian changes the program).
3. **Areas = modules/functions** — Same "body" (top-k); graph defines the "calls." Recurrence = recursive call; fixed point = return value.
4. **Control flow** — Sequencing = temporal order; loops = recurrence (built-in); conditionals = competition or gating (emergent or with small extension).
5. **Higher-level language** — Cognitive/symbolic layer (goals, plans, parse trees) **compiles to** assembly calculus. The **neural compiler** is the mapping from high-level to connectome + dynamics.
6. **Abstraction hierarchy** — Neurons (L0) → assemblies (L1) → cognitive ops (L2) → goals/language (L3). Assemblies are the **first** computational abstraction and the **target** for everything above.
7. **Type-like structure** — Areas or assembly "types"; compositionality (merge/association) gives product-like structure. Possible **categorical** formulation.
8. **Compiler phases** — Parse → analyze → optimize → codegen → run (with optional recompile at runtime via Hebbian).
9. **Turing completeness** — Open; expressiveness for cognitive tasks is the more relevant question.
10. **Learning = recompilation** — The brain is both the **compiler** (it rewrites its connectome) and the **runtime** (it runs the dynamics). So assemblies are not just a programming language — they're a **self-modifying** programming language, where execution **recompiles** the program.

---

**Bottom line:** The assembly calculus is a **neural programming language** (primitives, composition, execution semantics) and the **target** of a **neural compiler** (high-level cognitive/symbolic programs compile to assembly graphs). It is a **higher-level computational abstraction** than neurons (level 1 in a tower) and can support **even higher** levels (cognitive ops, goals, language) that compile down to it. **Learning** is **recompilation** — the program (connectome) is rewritten by experience. This view gives a clear **conceptual** and potentially **formal** (syntax, semantics, types, compiler phases) framework for what the assembly calculus *is* and how it fits into a larger story of neural and cognitive computation.
