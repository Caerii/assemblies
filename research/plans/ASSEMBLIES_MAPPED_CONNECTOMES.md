# Applying the assembly calculus to mapped connectomes (C. elegans, Drosophila)

**Goal:** Explore whether and how the **assembly calculus** (projection, association, merge, Hebbian, top-k) could be **applied** to **mapped connectomes** — real wiring diagrams from organisms whose brains have been (largely) fully reconstructed. The two canonical examples are **C. elegans** and **Drosophila**. This is a brainstorm and stub.

---

## 1. Why mapped connectomes matter

The assembly calculus is usually studied with **random** connectivity (Bernoulli(p) within and between areas). Real brains have **specific** wiring. If we take a **real** connectome (who is connected to whom, and optionally synaptic weights or strengths) and run **assembly-style dynamics** on it — projection (sum inputs, pick top-k per region), Hebbian plasticity, association, merge — we can ask:

- Does this real graph **support** assembly-like computation? Do stable assemblies form? Do projection, association, and merge "work" on this topology?
- Can we **predict** or **explain** neural activity or behavior by interpreting the dynamics as assembly operations?
- Does the **structure** of the connectome (motifs, area-to-area fibers, hierarchy) match what the assembly calculus would "want" for projection, association, merge?

So **applying** the system to C. elegans and Drosophila means: **real connectome as graph** + **assembly calculus as dynamics**. We're not claiming the brain "is" the assembly calculus; we're asking whether the calculus is a good **model** or **language** for reasoning about these mapped brains.

---

## 2. C. elegans

### 2.1 Connectome facts (numbers and sources)

- **Neurons:** 302 in adult hermaphrodite (282 somatic + 20 pharyngeal). **118 neuron classes** by morphology and connectivity (White et al. 1986).
- **Synapses:** ~5,000–6,400 chemical synapses, ~900–2,000 gap junctions, ~1,500–2,000 neuromuscular junctions. Whole-network assemblies exist (e.g. Varshney et al. 2011); ~75% synaptic reproducibility between animals. Cook et al. 2019 provided updated whole-animal connectomes for both sexes (Nature 2019).
- **Landmark:** White et al. (1986), "The structure of the nervous system of the nematode *Caenorhabditis elegans*," *Phil. Trans. R. Soc. Lond. B* — first complete connectome of any animal. Reconstructed from EM serial sections; chemical synapses and gap junctions both mapped.

### 2.2 Anatomy and natural "areas"

- **Somatic (282):** Nerve ring (head "brain"), ventral cord, tail ganglia. **Sensory:** ~68 in head (olfactory, gustatory, mechanosensory, thermosensory); processes to nose tip, axons to nerve ring. **Interneurons:** receive from sensory in nerve ring; **command interneurons** (e.g. AIY, AIZ, AVA) integrate sensory and drive locomotion (reversals, turns, chemotaxis). **Motor:** ventral cord (many postembryonic); e.g. SMD, RIV, SMB encode movement features.
- **Pharyngeal (20):** Separate subsystem for feeding.
- **Useful partitions for assembly "areas":** (1) Sensory / interneurons / motor. (2) Finer: sensory classes, command interneurons (AIB, AIY, AIZ, AVA, RIB, etc.), ventral-cord motor classes. (3) Nerve ring vs ventral cord vs tail. Literature and WormAtlas give neuron–class and region annotations.

### 2.3 Data sources and programmatic access

- **WormAtlas:** [wormatlas.org](https://www.wormatlas.org) — "The Mind of a Worm"; anatomy, neuron classes, connectivity references.
- **OpenWorm Connectome Toolbox:** [openworm.org/ConnectomeToolbox](https://openworm.org/ConnectomeToolbox/) — adjacency matrices, graph views, multiple datasets (chemical, electrical, extrasynaptic). Data in CSV/Excel and other formats.
- **owmeta:** [github.com/openworm/owmeta](https://github.com/openworm/owmeta) — Python library for C. elegans anatomy and connectivity; programmatic load of neurons, synapses, and connectivity.
- **WormNeuroAtlas / Full1:** Symmetry-corrected and other derived connectome datasets available via the Toolbox.

### 2.4 How we could apply the assembly calculus

- **Graph:** Use actual connectivity (adjacency or weights from synapse counts) from White-derived or Varshney/Cook assemblies. No random sampling — fixed real topology.
- **Areas:** Define from anatomy (e.g. sensory / command interneurons / motor) or from 118 classes; each area = set of neuron IDs; edges from the map.
- **Dynamics:** Per area: sum weighted input from active neighbors → **top-k** (or threshold) → optional **Hebbian** on weights. Run same assembly dynamics on the real graph.
- **Stimuli:** Drive sensory neurons (e.g. chemosensory, mechanosensory) corresponding to a stimulus; observe which interneurons and motor neurons "win" and whether trajectories match known circuits (e.g. AIY for chemotaxis, AVA for reversal).

### 2.5 Challenges

- **Scale:** n = 302 → k ≈ √302 ≈ 17; small but workable. Few assemblies per area.
- **Gap junctions:** Treat as separate edge type or ignore initially; chemical-only is a clear first step.
- **Plasticity:** Wiring largely fixed. Run with fixed weights (from synapse counts) to test activation dynamics; optionally add Hebbian on a subset to model slow plasticity.
- **Validation:** Compare to calcium imaging, optogenetics (e.g. AIY drive → chemotaxis), or behavioral state (reversal, turn). Brain-wide representations and state-dependent gating have been studied (e.g. state-dependent network interactions at command level).

### 2.6 Opportunity

C. elegans is the only organism with a **complete** connectome. If assembly dynamics on this graph produce interpretable "assemblies" that align with known circuits (avoidance, chemotaxis, command interneurons), the calculus becomes a candidate **language** for small, non-cortical brains.

---

## 3. Drosophila

### 3.1 Connectome facts (numbers and sources)

- **Hemibrain (2020):** Scheffer et al., eLife 2020 — "A connectome and analysis of the adult *Drosophila* central brain." ~**25,000 neurons**, ~**20 million** synaptic connections in roughly half of the central brain. Dense EM reconstruction (8 nm slices); ~64M postsynaptic densities, ~9.5M T-bars. Landmark for methods (prep, imaging, alignment, segmentation). Freely available online.
- **FlyWire whole brain (2024):** Nature 2024 — "Whole-brain annotation and multi-connectome cell typing of *Drosophila*." ~**139,000–140,000 neurons**, ~**50 million** chemical synapses in an adult female brain. Complete whole-brain connectome with neurotransmitter predictions. ~8,453 annotated cell types (3,643 overlapping with hemibrain, 4,581 new from regions outside hemibrain). Consensus cell-type atlas with hierarchical annotations (neuronal classes, cell types, hemilineages).
- **Scale:** Whole-brain n ≈ 1.4×10⁵ matches our simulation scale (numpy_sparse, cuda_implicit). Per-region n can be thousands (e.g. mushroom body Kenyon cells ~2,500).

### 3.2 Anatomy and natural "areas"

- **Mushroom body:** ~2,500 **Kenyon cells** (small neurons); cell bodies atop calyx (sensory input), axons through pedunculus to output lobes (α, β, γ, δ, α'). Laminar/stratified organization; 7 Kenyon cell subtypes + 3 other intrinsic types; ~20 extrinsic neuron types. Traditionally olfactory (~90% Kenyon cells receive olfactory input); connectomics shows ~8% receive predominantly **visual** input from optic lobe (sparse, distributed, combinatorial). Function: learning, memory, associative coding.
- **Central complex:** Navigation, compass, motor control; well-defined neuropils (e.g. protocerebral bridge, fan-shaped body, ellipsoid body).
- **Optic lobe:** Vision; layered neuropils (lamina, medulla, lobula, lobula plate).
- **Useful partitions for assembly "areas":** (1) Neuropil-based: mushroom body (Kenyon cells, input neurons, output neurons), central complex subregions, optic lobe layers. (2) Cell-type-based: use FlyWire/hemibrain cell-type annotations (8k+ types) to define areas or competition pools.

### 3.3 Data sources and programmatic access

- **FlyWire:** [flywire.ai](https://flywire.ai) — complete whole-brain connectome; 140k+ neurons, 50M+ synapses; neurotransmitter info. **Codex** = web-based FlyWire browser.
- **fafbseg (Python):** [fafbseg-py.readthedocs.io](https://fafbseg-py.readthedocs.io) — programmatic access: `get_synapses()`, `get_connectivity()`, `get_synapse_counts()`, `get_adjacency()`, `get_transmitter_predictions()`. Neuroglancer URL encode/decode for visualization.
- **seung-lab/FlyConnectome (GitHub):** CAVE API, segmentation queries, mesh rendering; Jupyter notebooks for access patterns.
- **Neuroglancer / cloudvolume:** Volumetric EM and meshes; subvolume reading, per-segment mesh download.

### 3.4 How we could apply the assembly calculus

- **Graph:** Use FlyWire or hemibrain connectivity (adjacency or synapse-count weights). Edges = chemical synapses; optionally filter by neurotransmitter.
- **Areas:** Define from anatomy (e.g. Kenyon cells as one area, mushroom body output neurons as another, central complex regions, optic lobe layers) or from cell-type hierarchy. Each area = set of neuron/segment IDs; edges from the map.
- **Dynamics:** Same as C. elegans: sum input → top-k per area → optional Hebbian. Stimuli = e.g. odor (drive olfactory inputs to mushroom body), visual (drive optic lobe), or fictive motor/place patterns.
- **Scale:** n per area can be 10³–10⁵; our engines can handle it. May coarsen or run on a subgraph (e.g. mushroom body + its inputs/outputs) first for tractability.

### 3.5 Challenges

- **Completeness:** Whole brain is now mapped (FlyWire 2024); hemibrain remains a smaller, well-validated subvolume. Can run on full FlyWire or on hemibrain-only subset.
- **Cell-type identity:** ~1/3 of hemibrain-proposed cell types could not be reliably re-identified in FlyWire; use consensus annotations and validated types where possible.
- **Plasticity:** Drosophila has known plasticity (mushroom body, dopaminergic reinforcement). Hebbian in our model can be compared or combined with known plasticity loci.
- **Validation:** Compare to calcium imaging, behavior (odor preference, place memory, navigation). Do assembly "winners" or trajectories correlate with recorded activity or behavioral state?

### 3.6 Opportunity

Drosophila has rich behavior (learning, navigation, social) and a scale (n ≈ 10⁵) that matches our simulations. Testing assembly dynamics on the fly connectome links the calculus to a real, experimentally tractable brain and to existing models (e.g. mushroom body as associative memory). FlyWire’s whole-brain scope and Python APIs (fafbseg, adjacency/connectivity) make loading and partitioning tractable.

---

## 4. What "applied" could mean in practice

1. **Load connectome:**  
   - **C. elegans:** Use [owmeta](https://github.com/openworm/owmeta) or OpenWorm Connectome Toolbox exports (CSV/Excel adjacency or edge list).  
   - **Drosophila:** Use [fafbseg](https://fafbseg-py.readthedocs.io) (`get_adjacency()`, `get_connectivity()`, `get_synapse_counts()`) or FlyConnectome/CAVE API. Represent as directed graph with optional weights (e.g. synapse counts).
2. **Partition into areas:** Define areas from anatomy (C. elegans: sensory/inter/motor; fly: mushroom body, central complex, optic lobe layers) or from cell-type annotations (FlyWire 8k+ types). Each area = set of neuron/segment IDs; edges within and between areas from the map.
3. **Run assembly engine:** Use our Brain/Engine with **custom connectome** — replace random Bernoulli(p) with the real adjacency (and optionally real or normalized weights). One step = project (sum inputs, top-k per area), optionally Hebbian update. Stimulate "sensory" or input neurons; run for T steps.
4. **Analyze:** Do stable assemblies form? Do they correspond to known circuits or behaviors? Can we "decode" stimulus or state from assembly activity? Compare to calcium imaging, optogenetics, or behavior where available.

**Implementation note:** Our current code assumes **random** connectivity from p and area sizes. To use a **mapped** connectome we need a path to load an explicit graph (adjacency or edge list) and use it as the connectome instead of sampling — e.g. "Brain from connectome" or "Engine from adjacency matrix."

---

## 5. Whole-brain emulation: how the assembly calculus could enable it

**Whole-brain emulation** here means: run the **entire** mapped connectome with **assembly-calculus dynamics** (projection, top-k, Hebbian), drive it with **sensory input** encoding the environment, and **read out** motor or internal state so that the simulated brain produces behavior (or internal trajectories) that can be compared to the real organism. Optionally **close the loop**: motor output → body/environment simulator → next sensory input. The assembly calculus is a candidate **dynamics engine** for such an emulation — same wiring, abstract but well-defined update rule — without requiring spike-level or biophysical detail.

### 5.1 Why the assembly calculus is a plausible emulation engine

- **Graph-agnostic dynamics:** Projection (sum inputs, top-k per area) and Hebbian (co-activity → strengthen) are defined on **any** directed graph. So the **whole** connectome — every neuron, every synapse — is the graph; we partition it into "areas" (competition pools) and run one round of dynamics per step. No need to invent connectivity; we use the real map.
- **Discrete, deterministic (given inputs):** One step = one round of: (1) for each area, sum weighted input from active pre-synaptic neurons; (2) pick top-k (or threshold) as "winners"; (3) optionally update weights by Hebbian. Given initial activity and inputs, the trajectory is well-defined. That makes emulation **reproducible** and **testable**.
- **Input–output interface:** **Input:** Map the environment (or experimental stimulus) to **which sensory neurons are active** — e.g. odor A → drive olfactory sensory assembly; touch → drive mechanosensory assembly. **Output:** Read **which motor (or command) neurons are active** after T steps and decode to behavior — e.g. ventral-cord motor activity → muscle activation → movement; or command interneurons → reversal vs forward. So we don't emulate ion channels; we emulate **which assemblies are active** and interpret that as sensory encoding and motor decoding.
- **Scale:** C. elegans (302 neurons) is trivial for our engines; Drosophila (139k) is in range (numpy_sparse, cuda_implicit). So **whole-brain** in the sense of "every neuron in the map" is computationally feasible.
- **Learning (optional):** If we allow Hebbian on (a subset of) the connectome, the emulated brain can **change** with experience — e.g. odor–reward associations in the fly mushroom body. That supports **behavioral** emulation over time, not just static input–output.

So: **whole-brain emulation** = real connectome (graph) + assembly dynamics (projection, top-k, Hebbian) + sensory encoding (environment → sensory neuron activity) + motor decoding (motor neuron activity → behavior) + optional body/environment loop.

### 5.2 The emulation loop

1. **Environment / stimulus:** Current state of the world (e.g. chemical gradient, touch, odor, visual scene).
2. **Sensory encoding:** Map environment to **sensory neuron activity**. E.g. odor concentration at nose → which olfactory sensory neurons (or their assembly) are "on"; touch location → which mechanosensory neurons. This can be hand-designed (e.g. receptive fields) or learned; for testing, we can use known tuning (e.g. from C. elegans or fly literature).
3. **Drive:** Set the activity of sensory (or input) neurons to the encoded state — i.e. set "winners" in sensory areas to the neurons that should fire for this stimulus.
4. **Run dynamics:** For T steps, run assembly calculus on the **whole** connectome: project (sum inputs, top-k per area), optionally Hebbian. T can be 1 (feedforward) or many (recurrent); for behavior, we typically need recurrence so that command and motor areas integrate over time.
5. **Read out:** After T steps, read the activity of **motor** neurons (or command interneurons that drive motor). Decode to **motor output** — e.g. which muscles contract, or movement direction (forward, reverse, turn). Again, mapping can be from literature (e.g. which motor neurons correspond to forward vs reverse in C. elegans).
6. **Body / environment (optional):** Feed motor output into a **simulator** (physics, chemistry, body model) that updates the environment (e.g. worm moves, chemical gradient changes). Next timestep: go to (1) with the new environment. That closes the loop and yields **embodied** whole-brain emulation.

If we skip (6) and only do (1)–(5) with **fixed** stimuli, we get "brain in a vat" emulation — same dynamics, but no closed loop. That's still useful for comparing assembly trajectories to recorded neural data (e.g. calcium imaging) under the same stimuli.

### 5.3 C. elegans: full-organism emulation

- **Scope:** Somatic nervous system (282) + optionally pharyngeal (20). Full connectome from White/Varshney/Cook; chemical synapses (and optionally gap junctions) as graph.
- **Sensory encoding:** Map stimulus (odor, touch, temperature) to sensory neuron activity using known receptor/tuning (e.g. AWC for odor, PLM/ALM for touch). One stimulus → one (or a few) sensory assemblies "on."
- **Motor decoding:** Ventral-cord motor neurons (e.g. A-type, B-type) drive muscles. Read which motor neurons are winners after dynamics; map to muscle activation (e.g. from literature or OpenWorm muscle model). That gives locomotion (forward, reverse, turn).
- **Loop:** Plug into a **body + environment** simulator (e.g. OpenWorm-style: worm body, fluid, chemical gradient). Sensory encoding reads from environment; motor decoding drives body; environment updates. Run for many timesteps. **Validation:** Does the emulated worm exhibit **chemotaxis** (move toward attractant), **avoidance** (reverse from repellent), **thermotaxis**, or other known behaviors? Compare to real C. elegans or to OpenWorm with other brain models.
- **Relation to OpenWorm:** [OpenWorm](https://openworm.org) aims for whole-organism C. elegans simulation. Their brain model has varied (e.g. conductance-based, or simplified). An **assembly-calculus brain** would be an alternative: same connectome, same sensory/motor interface, but our dynamics. Comparing behavior (e.g. chemotaxis accuracy, reversal frequency) between assembly-calculus emulation and other models (or real worms) would test whether the calculus is sufficient for functional emulation.

### 5.4 Drosophila: whole-brain and "brain in a vat"

- **Scope:** Full FlyWire connectome (~139k neurons) or a subset (e.g. hemibrain ~25k, or mushroom body + central complex + optic lobe). Areas from anatomy (Kenyon cells, MB output, central complex, optic lobe layers, etc.).
- **Sensory encoding:** Map odor (or visual scene) to olfactory (or visual) input neuron activity — e.g. odor identity → which olfactory receptor neurons (ORNs) or projection neurons (PNs) are active. Fly connectomics and physiology give tuning.
- **Motor / internal readout:** Read central complex (e.g. heading, steering) or mushroom body output (e.g. valence, choice) or descending neurons → decode to **behavior** (turn, speed, or choice in a T-maze). For full embodied emulation we'd need a fly body/arena simulator; that's harder than C. elegans. A more tractable first step: **brain in a vat** — stimulate with odor (or visual) sequences, run dynamics, read assembly trajectories and compare to calcium imaging (e.g. mushroom body, central complex) under the same stimuli. That tests whether assembly dynamics **reproduce** internal state trajectories, even without closing the loop.
- **Learning:** Drosophila mushroom body has dopaminergic reinforcement. If we allow Hebbian on MB synapses, the emulated fly can **learn** odor–reward associations. Then we can ask: does the emulated fly show **conditioned preference** (e.g. approach odor that was paired with reward)? That would be a strong functional test of whole-brain-style emulation (at least for the learned behavior).

### 5.5 What we need to build

- **Engine from connectome:** Load real graph (adjacency or edge list, optionally weights); partition into areas; run projection + top-k + Hebbian. No random sampling. (Already outlined in §4.)
- **Sensory encoder:** Module that maps environment/stimulus to sensory neuron IDs (or sensory assembly). For C. elegans: e.g. odor → AWC, ASH, etc.; for fly: odor → ORN/PN activity. Can start with lookup tables from literature.
- **Motor decoder:** Module that maps motor (or command) neuron activity to motor output. For C. elegans: motor winners → muscle activation (e.g. forward/reverse/turn). For fly: descending or central-complex activity → movement or choice. Again, literature or simplified model.
- **Optional body/environment:** For C. elegans, interface to an existing body simulator (e.g. OpenWorm) or a minimal one (point mass + gradient). For fly, harder; can defer to "brain in a vat" first.
- **Validation pipeline:** Same stimulus → compare (1) emulated assembly trajectory, (2) recorded neural activity (e.g. calcium), or (1) emulated behavior, (2) real behavior. Metrics: overlap of active neurons, correlation of trajectories, or behavioral success rate (e.g. chemotaxis index, choice accuracy).

### 5.6 Fidelity and caveats

- **We are not claiming spike-accurate or biophysical emulation.** We emulate at the level of **which neurons (assemblies) are active** each step. Real brains have continuous membrane potentials, delays, neuromodulation, gap junctions. We have discrete steps and top-k. So **functional** or **behavioral** emulation is the goal: does the same input produce similar output (or similar internal trajectories) to the real organism? That's a **hypothesis** — the assembly calculus might be sufficient for behavior-relevant computation even though it's an abstraction.
- **Timing:** One assembly step might correspond to several ms of real time (e.g. one round of recurrent activity). We may need to match step count to behavioral timescales (e.g. C. elegans reversal in ~1 s → many steps).
- **Gap junctions / neuromodulation:** Omitted or simplified in the first pass. Adding them (e.g. gap junctions as undirected edges, or modulatory gains) could improve fidelity later.

### 5.7 Summary

The assembly calculus **could** allow whole-brain emulation by (1) using the **mapped connectome** as the only connectivity, (2) running **projection + top-k + Hebbian** as the only dynamics, (3) **encoding** environment into sensory activity and **decoding** motor activity into behavior, and (4) optionally **closing the loop** with a body/environment simulator. For C. elegans, that means a full-organism emulation (brain + body) is in principle within reach — same connectome, our dynamics, plus sensory/motor interface and a worm simulator. For Drosophila, whole-brain "brain in a vat" (stimulate, run, read trajectories) is tractable at 139k neurons; embodied emulation would require a fly body/arena model. **Validation** is key: do we get the right behaviors or the right internal dynamics? If yes, the assembly calculus would stand as a candidate **minimal sufficient** dynamics for functional whole-brain emulation of these organisms.

---

## 6. View from an assemblies theorist: what we're actually claiming and where it's fragile

The following is a **critical** pass from the perspective of someone who works on the **theory** of the assembly calculus (Papadimitriou et al., Dabagia et al., etc.). It sharpens what we're assuming, what we're not guaranteed, and what would falsify or constrain the approach.

### 6.1 The theory assumes random connectivity; real connectomes are not random

- **What the theory proves:** Projection, association, merge, capacity, and phase diagrams are derived for **random** connectivity — Bernoulli(p) within areas and between areas (fibers). The proofs rely on specific scaling (e.g. k²p, n, β) and on the **typical** properties of random graphs (expansion, concentration). We have **no** theorem that says "for any graph, assembly operations form stable assemblies with the desired semantics."
- **Implication:** Applying the calculus to a **mapped** connectome is an **extrapolation** into a regime (real, highly structured topology) where we have **no theoretical guarantee**. The real graph may have motifs, hierarchy, and sparse/dense patches that are nothing like Bernoulli(p). So we're making an **empirical bet**: maybe the calculus is **robust** enough that it still "works" on real wiring. If it doesn't (e.g. no stable assemblies, or wrong behavior), that **falsifies** the sufficiency of the calculus for that brain (or that partition). If it does, we gain evidence that the calculus generalizes beyond the random-graph regime. An assemblies theorist would insist we state this explicitly: **mapped connectomes are a test of robustness, not an application of a theorem.**

### 6.2 Areas and k are imposed by us; the brain doesn't hand them to us

- **In the theory:** "Areas" are given; each has n neurons and we pick top-k. The fiber between A and B is a random bipartite graph. So the **partition** and **k** are part of the model.
- **In C. elegans and Drosophila:** There are no literal "areas" in the same sense — we **choose** a partition (sensory / inter / motor, or by neuropil, or by cell type). Different partitions yield **different** dynamics. So "apply assembly calculus to the connectome" is underspecified: we must say **which** partition and **which** k per area. For small regions (e.g. a handful of command interneurons), k might be 1 or 2; for large regions (e.g. Kenyon cells), k might be √n or less. There is no **canonical** choice from the theory; the theory assumes areas and k are given.
- **Implication:** Every partition is a **different model**. Validation (or falsification) is partition-relative. An assemblies theorist would ask: is there a **theory-motivated** or **anatomy-unique** partition? Or do we have to search over partitions and report which ones work? The latter weakens the claim that "the assembly calculus explains this brain."

### 6.3 Scaling (k, n, p) may not hold on real graphs

- **In the theory:** Capacity and stability depend on scaling laws (e.g. k ≈ √n, k²p in a range, β bounded). We tune p and k in simulations to sit in the "good" regime.
- **On a mapped connectome:** We don't have p — we have a **fixed** graph. "p" could be interpreted as effective connection probability in a region, but it's derived from the graph, not chosen. And **k** is chosen by us per area; for small areas (e.g. n_area = 10), k > n_area is meaningless, so we must set k ≤ n_area. So we're no longer in the parameter regime the theory was developed for. Heterogeneous area sizes and fixed topology mean we **cannot** freely tune k²p or n. The theorist would say: we're running the **same update rule** (sum input, top-k, Hebbian) but in a **different regime**; we have no guarantee that assemblies will form, stabilize, or have capacity. We need **empirical** checks (e.g. do we get stable, stimulus-specific assemblies for at least one partition?).

### 6.4 Projection and association on real "fibers"

- **In the theory:** Projection A→B works because the **random** fiber from A to B has enough connectivity that the k winners in A project to a set of neurons in B that, after competition, form a **new** assembly in B that "represents" A. The randomness guarantees (with high probability) the right expansion and overlap properties.
- **On the real connectome:** The "fiber" from sensory to inter (or inter to motor) is the **actual** wiring — it may be sparse, biased, or hierarchical. There is **no** theorem that says this wiring will support projection in the sense of the calculus (i.e. that the activity in B will be a stable, distinct assembly that we can interpret as "copy of A"). So we're **assuming** that the real graph's connectivity is "good enough." If it isn't (e.g. sensory activity doesn't drive a clean assembly in command interneurons), projection **fails** in our model — and we learn that this brain (or this partition) doesn't implement projection the way the theory assumes. The theorist would want a **minimal benchmark**: e.g. "sensory stimulus S → does a stable, S-specific assembly form in area B?" If no, we need a different partition or we give up on projection for this graph.

### 6.5 What would falsify the approach?

- **Assembly instability:** If, for a given partition and k, activity never stabilizes (chaos, or trivial all-zero/all-active), then the calculus does not produce the kind of attractor dynamics the theory predicts. That would suggest the real graph (or our partition) is not in a "calculus-like" regime.
- **Stimulus–output mismatch:** If the same stimulus produces completely different assembly trajectories (or motor readouts) than recorded neural data or behavior, then either (a) our dynamics are wrong, (b) our sensory encoding or motor decoding is wrong, or (c) the assembly calculus is insufficient for this brain. We'd need to disentangle (a)–(c); e.g. fix encoding/decoding and see if dynamics alone can match.
- **Behavioral failure:** If the emulated worm doesn't chemotax, or the emulated fly doesn't show conditioned preference, then **functional** emulation fails. We're not claiming spike accuracy — we're claiming **behavioral** sufficiency. So behavioral failure is a direct falsification of that claim (for that partition, that encoding/decoding).
- **Partition sensitivity:** If the result (stable assemblies, correct behavior) depends **strongly** on an ad hoc partition and k, and no anatomy-based choice is unique, then the claim "the assembly calculus explains this brain" is weak — we're fitting a model (partition + k) to data rather than deriving it. The theorist would want at least one **anatomy-motivated** partition that works without heavy tuning.

### 6.6 What we're not claiming (and what we need to prove)

- **We're not claiming:** The brain **is** the assembly calculus (literally). We're not claiming spike-accurate or biophysical emulation. We're not claiming that the theory (which is for random graphs) **implies** success on mapped connectomes.
- **We are claiming (as a hypothesis):** The assembly calculus might be a **sufficient abstraction** for **functional** (behavioral) emulation — i.e. that the relevant computation for behavior can be captured at the level of assemblies and our operations, so that running the same dynamics on the real graph + encoding/decoding yields the right input–output (or internal) trajectories. That is an **empirical** claim. The only way to test it is to run the emulation and compare. **Failure** is informative: it tells us the calculus is insufficient (or the partition/encoding/decoding is wrong) and may point to what's missing (e.g. gap junctions, neuromodulation, sequence machinery, or different dynamics).

### 6.7 Gaps the theory doesn't address (for mapped connectomes)

- **Non-random graphs:** We have no theory of assembly dynamics on **structured** graphs (e.g. small-world, hierarchical, or real connectomes). Developing such a theory (e.g. sufficient conditions on graph properties for projection/association to "work") would strengthen the foundations for applied work.
- **Heterogeneous areas:** Theory often assumes uniform n and k. Real partitions have areas of very different sizes. We need either area-dependent k and scaling, or a theory that allows heterogeneity.
- **Inhibition:** k-winners-take-all is a **stand-in** for inhibition (competition). Real brains have explicit inhibitory neurons. In C. elegans and fly, inhibition is present. Are we implicitly folding it into "top-k" (so that only k neurons win per area), or are we missing a role for inhibition (e.g. disinhibition, gating)? The theorist would flag this as a possible gap.
- **Sequence and time:** For **whole-brain emulation** of behavior that depends on **temporal** integration (e.g. odor over time, or sequence of stimuli), one-step projection might not be enough. Dabagia et al. add **sequences** (replay, memorization) and **LRIs** for FSM/Turing. So we may need to add sequence machinery (and possibly LRIs) to the emulation engine, not just static projection + top-k + Hebbian. The theorist would ask: for C. elegans chemotaxis or fly odor learning, do we need **sequence** operations, or is one-step (or few-step) recurrence enough? That's organism- and behavior-specific.

### 6.8 Summary for the theorist

Applying the assembly calculus to mapped connectomes (and using it for whole-brain emulation) is **not** a direct application of existing theory — it's an **extrapolation** and an **empirical test**. The theory gives us a **dynamics** (projection, top-k, Hebbian) and a **language** (assemblies, projection, association, merge), but it does **not** guarantee that real connectomes will support these operations. We must **choose** a partition and k; we must **encode** and **decode**; we must **run** and **compare**. Success would support the claim that the calculus is robust and sufficient for functional emulation in these organisms; failure would constrain or falsify that claim and direct us toward what's missing. An assemblies theorist would insist on **explicit falsification criteria**, **minimal benchmarks** (e.g. stability, stimulus-specific assemblies, behavior), and **honesty** that we're testing a hypothesis, not applying a theorem.

---

## 7. Why applying to mapped connectomes is a good fit (empirical angle)

- **Same dynamics:** Projection, top-k, Hebbian are **local** and well-defined on **any** graph. So we can run them on a mapped connectome once we have a partition. The theory doesn't guarantee success, but the **operation** is defined.
- **Interpretability:** If assemblies on the real graph align with known neurons or circuits, the calculus gives a **vocabulary** for describing computation. If they don't, we learn something.
- **Predictions and tests:** We can predict trajectories or behavior and compare to data. Falsifiable.
- **Bridge:** If it works, the calculus becomes a candidate description of real brains; if it fails, we know we need more (or different) structure.

---

## 8. Open questions and caveats

- **Area definition:** How to partition neurons into "areas" for top-k? Literature-based anatomy (C. elegans: sensory/inter/motor; fly: mushroom body, central complex, etc.) is one option. Data-driven clustering of the connectome is another. Different partitions may give different assembly interpretations.
- **Weights:** Mapped connectomes often give synapse counts or presence. We could use counts as initial weights, or binary edges. Hebbian would then **modify** these over simulated time.
- **Gap junctions / neuromodulation:** C. elegans and Drosophila have gap junctions and neuromodulation. Our model is chemical synapses + Hebbian. We could ignore non-chemical or treat them separately; that's a simplification.
- **Validation:** The gold standard would be comparing simulated assembly activity to recorded neural activity or behavior. That requires access to data and a clear protocol (same stimuli, same time scale). Without that, we can still ask structural questions (does the connectome have the "right" motifs for projection/association?) or qualitative ones (do assemblies stabilize? do they look interpretable?).

---

## 9. Key references (for implementation and citation)

**C. elegans**

- White et al. (1986). The structure of the nervous system of the nematode *Caenorhabditis elegans*. *Phil. Trans. R. Soc. Lond. B* 314, 1–340. (First complete connectome.)
- Varshney et al. (2011). Structural properties of the *Caenorhabditis elegans* neuronal network. *PLoS Comput. Biol.* (Whole-network chemical + gap junction assemblies; topology.)
- Cook et al. (2019). Whole-animal connectomes of both *Caenorhabditis elegans* sexes. *Nature* 571, 63–71. (Updated connectomes both sexes; quantitative matrices.)
- WormAtlas: [wormatlas.org](https://www.wormatlas.org). OpenWorm Connectome Toolbox: [openworm.org/ConnectomeToolbox](https://openworm.org/ConnectomeToolbox). owmeta: [github.com/openworm/owmeta](https://github.com/openworm/owmeta).

**Drosophila**

- Scheffer et al. (2020). A connectome and analysis of the adult *Drosophila* central brain. *eLife* 9, e57443. (Hemibrain: ~25k neurons, ~20M synapses.)
- Dorkenwald et al. / FlyWire Consortium (2024). Whole-brain annotation and multi-connectome cell typing of *Drosophila*. *Nature* 632, 575–582. (FlyWire whole brain: ~139k neurons, ~50M synapses; cell-type atlas.)
- FlyWire: [flywire.ai](https://flywire.ai). fafbseg (Python): [fafbseg-py.readthedocs.io](https://fafbseg-py.readthedocs.io). FlyConnectome (GitHub): seung-lab/FlyConnectome.

---

**Bottom line:** Yes — our system **can** be applied to C. elegans and Drosophila. Use the **mapped connectome** as the graph (fixed topology, optionally real weights), define **areas** from anatomy or clustering, and run **assembly calculus dynamics** (projection, top-k, Hebbian) on top. That would test whether the calculus is a useful model for real, fully-mapped brains and could link our theory to the only two organisms with (nearly) complete connectomes. Implementation would require a path to load real connectomes into our engine (custom connectome instead of random); the rest is the same dynamics we already have.
