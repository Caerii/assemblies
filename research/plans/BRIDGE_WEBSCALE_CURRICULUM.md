# Bridge: Assembly calculus ↔ web-scale data and training curricula

**Goal:** How to connect the assembly calculus to **web-scale data** and **training curricula**; what this implies; and whether it can do **next-token prediction** (and how).

**Flagship title:** **Assembly Foundation Models** (or a more memorable variant below) — the most complete overview of this research program: assembly calculus (projection, association, merge, Hebbian, top-k) + scaling + curriculum + web-scale data + next-token prediction + differentiable path (STE) + theory (SM, ML/linear algebra, complex systems) + foundation-model pre-train-and-adapt vision. One title for the full arc from cortical assemblies to foundation-style models.

---

## Why "Attention is all you need" works (and what our equivalent is)

**Why that title sticks:**

1. **It names the *mechanism* that defines the paradigm.** The paper didn’t invent attention; it said: you can drop recurrence and convolution and build *only* on attention. So the title is a **sufficiency claim** — this one mechanism is *enough* for the whole architecture.
2. **"All you need" is deliberately provocative.** It implies: (a) simplicity (one thing suffices), (b) a break from the past (you don’t need the old machinery), (c) a new norm. So it’s both a scientific claim and a **rhetorical move** — declare the new paradigm in one sentence.
3. **It’s about the *operation*, not the application.** The title doesn’t say "Transformers for Language." It names the **core operation** (attention) and claims that’s the foundation. The paper is framed as *architectural* / *mechanistic*.
4. **It invites a contrast.** "All you need" implies "you thought you needed X, Y, Z — you don’t." So the reader immediately thinks: what did we think we needed? Recurrence. The title says: just attention. **Contrast without naming the loser.**
5. **Minimal formula.** One word (attention) + one frame ("is all you need"). Short, grammatical, easy to say.

**So for us, the *equivalent* is:**

Name the **one mechanism or unit** that defines *our* paradigm and claim it’s **sufficient** — and implicitly contrast with what the dominant paradigm uses (attention, dense, backprop-only).

- **Their** core: attention (dense, all-to-all selection by similarity).
- **Ours** core: assemblies (sparse, k winners, projection + Hebbian). Or, more narrowly: **top-k** (the selection op), **sparsity** (the property), or **winners** (who’s active).

So the *structural* parallel is: **"X is all you need"** where X is our building block. Options for X:

| X | What it names | Contrast with | Pros / cons |
|---|----------------|----------------|-------------|
| **Assemblies** | The representation unit (Papadimitriou’s term) | Attention (their unit) | Strong: same rhythm as "Attention is all you need"; "assemblies" = our paradigm. Risk: sounds derivative. |
| **Top-k** | The core nonlinearity (selection) | Softmax / dense attention | Sharp: *operation*-level parallel; ML people know "top-k." Surprise: "That thing we use for inference is the *whole* architecture?" |
| **Sparsity** | The property (k of n) | Dense | One word; clear contrast. Less specific than "assemblies." |
| **Winners** | Who’s active (winner-take-all) | "Everyone participates" (attention) | Evocative; "winners take all" is a known phrase — could play on it. |
| **Projection** | The core op (W @ active → top-k) | Attention (Q,K,V) | Precise but less iconic than "assemblies" or "top-k." |

**Deeper twist: don’t mirror the formula, mirror the *effect*.**

- **Liberation:** "You thought you needed attention (dense, expensive). You don’t — **sparse** (or **assemblies**) is enough." So: **Assemblies are all you need** or **Sparsity is all you need**.
- **Contrast in four words:** **Assemblies, not attention** — no "all you need"; just the swap. Memorable, direct.
- **Minimal / cryptic:** **All you need is k** — "k" = the magic number (sparsity). Very short; sticks. Might be too cryptic for a first paper.
- **Wordplay:** **Winners take all** — the mechanism (winner-take-all) in a familiar phrase. Memorable; could be read as "our selection rule is sufficient."

**Refined shortlist (by effect):**

1. **Assemblies are all you need** — Direct parallel; paradigm unit; "assemblies" = our building block. Same rhythm as the original.
2. **Top-k is all you need** — Operation-level parallel; surprising for ML readers; names the *selection* that replaces dense attention.
3. **Assemblies, not attention** — Maximum contrast in four words; no formula, just the swap.
4. **Sparsity is all you need** — One-word contrast with "dense"; clear and short.
5. **Winners take all** — Wordplay; memorable; "winners" = our selection. Softer sufficiency claim.
6. **Assembly Foundation Models** — Not "all you need" style; instead names the *goal* (foundation models from assemblies). Best for the **overview** paper; less punchy than (1)–(4) but clear and scoped.

**Recommendation:** For the **most complete overview** paper, **Assembly Foundation Models** is the right *descriptive* title (what the paper is). For a **memorable, paradigm-declaring** title that "goes hard" like "Attention is all you need," use **Assemblies are all you need** (same formula, bold claim) or **Top-k is all you need** (same formula, operation-level surprise). If you want maximum contrast without repeating the formula: **Assemblies, not attention.**

**Caveat: "All you need" = sufficiency.** You shouldn't use **Assemblies are all you need** unless assemblies can do *everything* that matters for the claim — e.g. next-token prediction at scale, web-scale pre-training, adaptation, and (if you're claiming parity) the same applications as attention-based foundation models. Until you've shown or seriously argued that, the title overclaims. Safer options until then: **Assembly Foundation Models** (what we're building; no sufficiency claim), **Toward Assembly Foundation Models** (explicitly aspirational), or **Assemblies, not attention?** (provocative but a question, not a flat "all you need"). If you do use "Assemblies are all you need," frame it as the **thesis** of the paper ("we argue that assemblies are all you need; we present evidence X, Y, Z") rather than established fact — and be ready to back it up.

---

## 1. Can it do next-token prediction?

**Yes.** Mechanically:

- **Context** = previous tokens (words or subwords). Map each token to an assembly (or an input that drives an assembly): e.g. token id → stimulus to a "LEX" area, or token embedding → projection into assembly space. You already have this pattern in `gpu_language_learner`: `activate_word` + `project('LEX','LEX')` for context, then score words by **overlap** with current LEX activation → `predict_next_word`.
- **Readout:** From the current "context assembly" (state of LEX after processing context), produce a distribution over the **next token**. Two options:
  - **Overlap-based (current):** Score each vocab item by overlap of its assembly with the context assembly; normalize to get a distribution. No extra parameters; works with Hebbian-only learning.
  - **Differentiable readout:** Context assembly (e.g. k winner indices or a dense vector from assembly state) → linear (or MLP) → logits over vocab size → softmax → cross-entropy loss on next token. Train the readout (and optionally W via STE) to minimize next-token loss.
- **Training:** Next-token prediction = cross-entropy(predicted next token, actual next token). So the **bridge** is: same assembly dynamics; the **training signal** comes from web-scale text (next token given context). You run dynamics on context → get context assembly → readout → logits → loss.

So: **next-token prediction is already in the design** (e.g. `predict_next_word`); the step to "web-scale" is (1) tokenization and vocab at scale, (2) mapping tokens to assembly inputs at scale, (3) a readout trained (or used) with next-token loss on huge data, (4) optional curriculum over that data.

### 1b. Completion first, then instruction/chat (GPT-style trajectory)

**Early GPT-2 and GPT-3** were trained only for **token completion** (next-token prediction on raw text). Chat and instruction tuning came **after** base training: supervised fine-tuning (SFT) on (instruction, response) or chat-format sequences, then (for GPT-3.5+) RLHF or similar. So the trajectory was: **base = completion-only** → **then** add chat/instruction on top.

**We can follow the same trajectory.**

1. **Phase 1 — Completion-only base:** Train the assembly model on **next-token prediction** only, on **completion-style** sequences (raw text, no special chat or instruction format). Same as early GPT: the model sees continuations of text; the only objective is predict the next token. Curriculum (length, frequency, domain) applies here. At the end of Phase 1 you have a **base** assembly language model that does completion.
2. **Phase 2 — Instruction/chat tuning:** After the base works, add **instruction or chat** tuning. Options:
   - **SFT on (instruction, response) pairs:** Train on sequences like `[Instruction] ... [Response] ...`; same assembly dynamics and readout; the model learns to continue in a "helpful" way when the context is an instruction. You can fine-tune the readout (and optionally W via STE) on these sequences.
   - **Chat-format sequences:** Train on multi-turn dialogue (e.g. `User: ... Assistant: ...`); again, same engine, different data stream. The model learns to produce assistant-style continuations after user turns.
   - **Optional:** RLHF or other preference-based training on top of SFT, same as in the transformer world — if we have a reward signal and a way to backprop or approximate gradients through the assembly readout.

So the **trajectory** is the same as GPT: **foundation first (completion only), then instruction/chat**. We don't need to train for chat from day one; we get a completion-capable base, then adapt it to instruction/chat with a second phase of training on the right kind of sequences. The assembly calculus (projection, association, merge, Hebbian, top-k) stays the same; only the **data** and possibly the **loss schedule** change in Phase 2.

---

## 2. Bridge: Web-scale data → Assembly calculus

**Three links:**

### A. Data → tokens → assembly inputs

- **Web-scale data** = huge text (or multimodal) corpora. **Tokenize** (subwords, words, or sentences) so you have sequences of token ids.
- **Token → assembly:** Each token type (or token id) must **drive or map to** an assembly. Options:
  - **One assembly per token:** Assign each vocab type a fixed assembly (e.g. random k neurons, or learned embedding → project to area). Context = sequence of token assemblies; you form a "context assembly" by projection/association/merge over recent tokens. Scale: vocab size = number of assembly types you maintain (could be 50k–1M+); you may use hashing or embedding to keep it sparse.
  - **Embedding → assembly:** Token embedding (learned vector) → linear projection → top-k in an area = "token assembly" for that step. Then context = run dynamics on the sequence of these assemblies (e.g. recurrent projection, or merge of last L tokens). The embedding + projection can be trained with the readout via next-token loss.
- So **web-scale** means: (1) tokenization at scale (e.g. SentencePiece/BPE on a large corpus); (2) a mapping from token id (or embedding) to assembly or assembly input; (3) batching and streaming so the assembly engine sees long sequences (or windows) of token-driven inputs.

### B. Training objective = next-token (or related) loss

- **Objective:** Minimize cross-entropy(next token | context). Context is processed by assembly dynamics; readout maps context assembly → logits over vocab.
- **Implied:** You need a **differentiable** path from loss back to (a) readout weights and (b) optionally W. So either:
  - **Readout-only:** Dynamics stay Hebbian (or fixed); only the readout (assembly state → vocab logits) is trained with gradients. Context representation is given by the assembly dynamics; you only learn how to map it to next token.
  - **Full STE:** Dynamics use STE so d(loss)/d(W) exists; you train both W and readout so that dynamics form "good" context representations for next-token prediction. Same dynamics (hard top-k); training shapes W and readout.
- **Other objectives:** You can also use related losses (e.g. span prediction, contrastive, multi-task) as long as the loss is a function of assembly state and you have gradients to the readout (and optionally W).

### C. Curriculum = ordering of data (and tasks)

- **Curriculum** = the order in which the model sees data (or tasks). For web-scale + assembly calculus:
  - **By sequence length:** Short sequences first (e.g. 1–5 tokens), then longer (paragraph, document). Lets the assembly system learn stable context assemblies on short context before long.
  - **By frequency / domain:** High-frequency tokens or one domain first (e.g. simple text, then Wikipedia, then code). Matches your existing curriculum idea (stages by vocab size, complexity) but applied to web-scale token streams.
  - **By task:** Next-token only first; then add span prediction, QA, or embodied tasks (if you have that data). Same assembly engine; curriculum chooses which loss(es) to optimize when.
- So the **bridge** for curriculum is: **same assembly dynamics**, but the **data stream** and **loss schedule** are curriculum-controlled (easy→hard, narrow→broad). Your existing curriculum docs (embodied stages, MHC, vocab stages) can sit **on top** of this: e.g. Stage 1 = short, high-freq, next-token only; Stage 2 = longer, more vocab; Stage 3 = multi-task or embodied if you have that data.

---

## 3. What does this imply?

**Implications:**

1. **Assembly calculus as representation engine** — The "brain" does not change: projection, association, merge, Hebbian, top-k. Its job is to turn **context** (sequence of token-driven inputs) into a **context assembly**. The readout (and optionally W) is trained so that this representation predicts the next token (or other objective). So assemblies are the **internal representation** for a next-token (or foundation-style) model.

2. **Scale and engineering** — To use web-scale data you need: (1) tokenization and vocab at scale; (2) efficient mapping token → assembly input (e.g. embedding table + projection, or hashing); (3) batching and long-context handling (sliding window, recurrence, or hierarchical); (4) a readout that can output over a large vocab (e.g. 50k–500k). Your CUDA assembly kernels give you speed; the bridge adds the **data pipeline** and **readout** so that the same kernels are fed from web-scale corpora and trained with next-token loss.

3. **Curriculum is first-class** — You are not training on a random shuffle of the internet. You **order** the data (and possibly the loss) by difficulty/length/domain. That aligns with your existing curriculum philosophy (stages, MHC) and with evidence that curriculum helps both biological and artificial systems. So "web-scale" + "assembly calculus" naturally implies **curriculum-based web-scale training**.

4. **Next-token prediction is the core objective** — Same as LLMs: next-token prediction on large text is a scalable, self-supervised objective that forces the system to build useful context representations. So the assembly-based model **can** do next-token prediction; the bridge is the data pipeline + readout + (optionally) STE for W. Whether it **matches** transformer LLMs in quality depends on capacity, context length, and scale—but the **mechanism** (context → dynamics → context assembly → readout → next-token logits) is valid.

5. **Foundation-style path** — Pre-train on web-scale next-token (with curriculum); then adapt (fine-tune readout, add tasks, or use in-context Hebbian for new tokens). So this bridge is a step toward an **assembly-based foundation model**: same dynamics, web-scale curriculum-trained, then adaptable.

---

## 4. Minimal recipe (summary)

| Step | What |
|------|------|
| 1 | **Tokenize** web-scale text → sequences of token ids (e.g. BPE/subword, vocab 50k–500k). |
| 2 | **Map token → assembly input:** Either one assembly per token type, or token embedding → projection → top-k in an area. |
| 3 | **Context assembly:** For each position, run assembly dynamics on previous tokens (e.g. activate token assemblies in order, project LEX→LEX or merge last L; recurrence or window). |
| 4 | **Readout:** Context assembly (winners or dense vector) → linear/MLP → logits over vocab. |
| 5 | **Loss:** Cross-entropy(next token, logits). Train readout (and optionally W with STE). |
| 6 | **Curriculum:** Order data by length, frequency, or domain; optionally order tasks (next-token first, then others). |

**Can it do next-token prediction?** Yes: context tokens → assembly dynamics → context assembly → readout → next-token logits → cross-entropy. The bridge is the **data pipeline** (web-scale tokens → assembly inputs) and the **readout** (assembly state → vocab logits), with curriculum controlling the order of training. The assembly calculus itself stays the same; you are using it as the **representation engine** for a next-token (and potentially foundation) model.

---

## 5. How much more efficient would assembly foundation models be?

**Caveat:** There is no built assembly foundation model yet; the numbers below are **theoretical / comparative** from the assembly calculus (sparse, recurrent, O(log n) convergence) and standard transformer scaling. Real speedups depend on implementation, batching, and matching quality.

### Compute (FLOPs) per token

- **Transformer (decoder):** Attention is **O(L² d)** per layer (L = context length, d = model dim); FFN is O(L d²). So **per token**, with L-token context: O(L d) from attention (one row) × layers, plus FFN. **Total over L tokens:** O(L² d) per layer dominates → **quadratic in context length**.
- **Assembly (recurrent):** One “step” = projection O(n) + top-k O(n log k). With **T steps per token** (e.g. T ≈ 3–6 from your scaling: convergence in O(log n) steps), **per token** cost is O(T · n log k). Over L tokens: **O(L T n log k)** → **linear in context length**. No L².
- **So:** If n ≈ d (same “width”), assembly is **O(L T n log k)** vs transformer **O(L² d × layers)**. For **long context** (L large), assembly wins: **linear vs quadratic in L**. Order-of-magnitude: e.g. L = 100k, d = n = 4k, T = 5, k = 1k → assembly ~ L T n log k ≈ 100k × 5 × 4k × 10 ≈ 2e10 vs transformer L² d ≈ 100k² × 4k ≈ 4e13 per layer — **~1000× fewer FLOPs per layer** for that L (and assembly has no L² term at all when processing one more token).

### Memory (context)

- **Transformer:** KV cache = **O(L d)** (and often O(L²) for full attention matrix unless optimized). **Linear in context length.**
- **Assembly (recurrent):** Context is in the **current assembly state**: k winners per area, fixed number of areas. So **O(areas × k)** — **constant in L**. No need to store L tokens of activations if you run recurrently; you only keep the current “context assembly.”
- **So:** **Constant vs linear in L** — assembly can have **O(1) context memory** in context length vs O(L) for transformers. For L = 100k, that’s a huge win (no 100k-length KV cache).

### Parameters

- **Transformer:** Dense: **O(d²)** per layer (attention + FFN) × num_layers. Billions for large models.
- **Assembly:** **Sparse** W: O(p n²) per connectome with p ≪ 1 (e.g. p ≈ 0.01–0.05), or **implicit** (hash-based) so **no stored W** — only the hash seed and k winners. So **far fewer parameters** if using implicit connectivity; with stored sparse W, still **p n²** vs dense n².
- **So:** **Sparsity** gives **1/p** fewer parameters than dense (e.g. 20–100× for p = 0.01–0.05); **implicit** can give **zero** stored weights for the recurrent core.

### Training

- **Forward:** Same as above — assembly forward is O(L T n log k) vs transformer O(L² d × layers). So **training FLOPs per batch** can be much lower for assembly for long sequences.
- **Backward:** With STE, assembly has a backward; cost is similar in spirit to forward (same graph). So **training time** can still be dominated by the **linear-in-L** assembly forward vs **quadratic** transformer, i.e. **potential large speedup for long-context training**.

### Summary (order-of-magnitude)

| Axis | Transformer | Assembly | Win |
|------|-------------|----------|-----|
| Compute vs L | O(L² d) per layer | O(L T n log k) | **Linear vs quadratic in L** — big win for long context |
| Context memory | O(L d) | O(k) recurrent | **Constant vs linear in L** |
| Parameters | O(d²) dense per layer | O(p n²) or 0 (implicit) | **Sparse / implicit** — 20–100× or more fewer params |
| Per-step time (your CUDA) | — | Sub-ms to few ms for n ≤ 10M | Already **100s–1000s Hz** per projection |

**Bottom line:** Assembly foundation models could be **much more efficient** on **long context** (linear compute and constant context memory vs quadratic and linear), and **parameter-efficient** (sparse or implicit). The largest gains are for **large L**; for short context the gap shrinks. Exact “how much more efficient” depends on matching quality and implementation — but the **scaling** (linear vs quadratic in L, constant vs linear memory) is the structural advantage.

---

## 5b. Could assemblies support language models that "understand" billions of words of context?

**Short answer:** Yes, in a **different sense** than transformers — not by "attending over 1B tokens at once" (no one can do that), but by **training on billions of words** so the connectome encodes that exposure, and **retrieving** relevant assemblies when given the current input. So "billions of words of context" = **implicit** (in the weights) + **retrieval-based** (current input activates the right assemblies from that huge training history).

### Two meanings of "long context"

- **Transformer-style:** Context = the **explicit** token sequence in the current window (e.g. last 128k or 1M tokens). The model **attends** over that sequence. "Billions of words" in this sense = literally 1B tokens in the window → **infeasible** (compute, memory). No architecture does that.
- **Assembly-style:** Context = (1) **Training exposure** — the model has **seen** billions of words; the **connectome** (weights) has been updated (Hebbian or STE) so that co-occurring or related patterns have stronger links. (2) **Current state** — at inference, the "context" is the **current assembly state** (winners in each area) plus the connectome. When you give a **new** input (e.g. a query, or a long document processed in chunks), the dynamics **retrieve** (activate) the assemblies that are relevant to that input from the trained connectome. So the model doesn't "hold" 1B tokens in a buffer — it **compresses** that information into the connectome during training, and **retrieves** what's relevant at inference. That's **associative / retrieval-based** long context.

### How assemblies could "understand" billions of words

1. **Train on billions of words:** Stream a huge corpus (e.g. 1B tokens or more) through the assembly model: token → assembly input, run dynamics, Hebbian (or STE) update. The connectome **evolves** so that frequently co-occurring or related token assemblies become strongly associated. So the "context" of billions of words is **implicit** in the weights — the model has "absorbed" that statistics.
2. **At inference, retrieve, don't attend:** Given a **current** input (e.g. a question, or a 10k-token document processed in chunks), you don't feed 1B tokens. You feed the **current** input; the dynamics **activate** the assemblies that match or associate with that input. So the "relevant context" from the training corpus is **retrieved** by association (pattern completion, propagation), not by literal attention over 1B tokens. That's similar in spirit to **retrieval-augmented** LMs (RAG): you don't put the whole corpus in the context window; you **retrieve** relevant pieces. In assemblies, "retrieval" is **built into** the dynamics — the connectome *is* the retrieval structure.
3. **Recurrent processing of long documents:** If the **current** context is a long document (e.g. 1M tokens), you can process it **recurrently**: feed tokens (or chunks) one-by-one, run dynamics, update state. The **state** (current winners + connectome) at the end is a **compressed** representation of the document. So "understanding" the document = having a state that was shaped by processing it; then when you ask a question, you run dynamics from that state + the question and read out. No need to re-attend over 1M tokens — the state is **O(areas × k)**, constant in document length. So assemblies can "understand" **very long documents** (millions of tokens) in the sense of **compressing** them into a fixed-size state by recurrence; "billions of words" of **training** context is in the connectome; "millions of words" of **current** document context can be compressed into state by recurrent processing.

### Capacity: can the connectome hold "billions" of patterns?

- **Theory:** Assembly capacity (number of stable assemblies, or associations) is bounded by n, k, p, and Hebbian regime. Typical analyses give **exponential in k** or **polynomial in n** under certain scaling. So "billions" of **distinct** semantic patterns (e.g. billions of n-grams or concepts) might require very large n (and/or many areas) and a lot of Hebbian updates. Whether capacity **scales** to billions is an **open** question — we'd need to check theory (e.g. Papadimitriou, Dabagia) and/or run scaling experiments.
- **Compression:** We may not need **billions of distinct** assemblies — we need the connectome to encode **statistics** (e.g. which token co-occurrences are strong, which concepts associate). So the "information" in billions of words might be **compressed** into a smaller number of strong associations. That's more like **dense** retrieval (many patterns share overlapping assemblies) than like a literal 1B separate assemblies.
- **Pragmatic path:** Start with **millions** of words (or tokens) of training; measure capacity and quality. Scale up; if capacity or quality plateaus, we learn the limits. If it scales, we push toward billions.

### Sequence and order

- **Order matters in language.** Assemblies excel at **association** (A and B linked); for **strict order** (e.g. long-range dependency, word order) we may need **sequence** machinery (Dabagia et al.: sequence memorization, replay). So for "understanding" billions of words of **ordered** context, we might need (a) huge associative capacity (connectome trained on 1B words) and (b) sequence operations for order-sensitive tasks (e.g. "what was the 5th word in the document?"). For many **semantic** tasks (e.g. "what is this document about?", "answer from the document"), association + retrieval may suffice — we don't need to store exact order of 1B words, we need to retrieve the right content. So the answer is **task-dependent**: pure association + retrieval for semantic "understanding"; sequence machinery for strict order.

### Summary

- **Could neural assemblies be used to make language models that understand billions of words of context?** **Yes, in a specific sense:** (1) **Train** on billions of words so the connectome encodes that exposure (associations, statistics). (2) **At inference**, the "context" is not a 1B-token buffer — it's **retrieved** by the current input (query or document) activating the relevant assemblies from the trained connectome. (3) For **current** long documents (e.g. millions of tokens), process **recurrently** and compress into fixed-size state; then query against that state. So "billions of words of context" = **implicit in the weights** (training) + **retrieval at inference** + **recurrent compression** for the current long document. That's a **different** (and potentially more scalable) kind of long context than transformer-style "attend over L tokens." **Open:** capacity scaling to billions of patterns, and whether **quality** of understanding matches or beats transformer long-context models; **efficiency** (linear in L, constant state) is already a structural advantage for assemblies.

---

## 6. State space models: disadvantages and how assembly calculus relates

**Yes — assembly foundation models can have disadvantages similar to SSMs**, because they share the same **structural** idea: **fixed-size recurrent state** and **linear-in-L** compute/memory. The key is *where* we align with SSMs (and their downsides) and *where* we differ.

### What state space models are (briefly)

- **SSMs** (S4, Mamba, etc.): Linear recurrence **h_t = A h_{t-1} + B x_t**, output **y_t = C h_t** (or with input-dependent A, B). **Fixed-size latent state** h; one update per token; **O(L)** compute and memory in context length.
- **Advantages:** Linear scaling; long context without L²; fast inference.
- **Disadvantages (often cited):**
  1. **Fixed-state bottleneck:** Context is compressed into a fixed-size vector h. You **can't store** L tokens' worth of arbitrary information in a fixed-size state — so **information is lost** or **blurred** over long context.
  2. **No direct position-based access:** Attention can "attend to position 37,482" directly. SSMs don't have **index-based** access; retrieval is through the recurrent state, so **needle-in-haystack** or **exact position** recall can be harder.
  3. **Weaker in-context learning (in some benchmarks):** Tasks that need **selective retrieval** (e.g. "what was said 50k tokens ago?") or **many distinct lookups** can favor attention, which has O(L²) but **direct** access to every position.
  4. **Linear recurrence:** Expressiveness is limited by linear mixing (even with input-dependent SSMs); attention is inherently **nonlinear** over positions (softmax over scores).

So the **tradeoff** is: SSMs get **efficiency** (O(L), small state) but risk **limited retrieval** and **information bottleneck** compared to attention.

### How assembly calculus relates

- **Same high-level shape:** We have a **fixed-size recurrent state** — **k winners per area** (and a fixed number of areas). Context is processed **recurrently** (token by token, or window by window); we don't store L tokens of activations, we **update** the assembly state. So **O(L)** compute, **O(k)** context memory — same **scaling** as SSMs, same **fixed-state** idea.
- **So we *do* share the main structural downside:** A **fixed-size state** (k winners) cannot store L tokens' worth of arbitrary information. We **compress** context into k neurons (per area). So **yes** — we can have the same kind of **disadvantage** as SSMs: **limited capacity** for fine-grained or position-heavy retrieval from very long context.

### Where we differ from SSMs

- **Nonlinear recurrence:** Our update is **top-k(W @ state + input)** — **nonlinear** (selection, winner-take-all), not linear like h = A h + B x. So we have **nonlinear dynamics** and **attractor** behavior (fixed points, basins). That can be **more expressive** than a linear recurrence for **pattern completion** and **content-based** structure.
- **Associative, not position-based:** We don't have "position 37,482"; we have **content** — assemblies that **associate** (A + B → C), **merge**, and **compete**. So "retrieval" in our setting is **associative** (cue → dynamics → attractor that represents that info), not "read index i." So:
  - **Position-heavy tasks** ("what is the 10th word?", "recall token at index 50k") — we're **similar to SSMs**: no direct index; we can be **worse** than attention.
  - **Content / associative tasks** ("what was said about the capital of France?", "complete the pattern") — we're **different from linear SSMs**: we're built for **association** and **pattern completion**, so we might be **better** than linear SSMs (and sometimes competitive with attention) for **content-based** retrieval.
- **Sparse state:** Our state is **k indices** (winners), not a dense vector. So the "bottleneck" is **k slots** that can **persist** and **compete** — more like a **set of active patterns** than a blended vector. That might help **preserve** a few distinct "threads" (e.g. k assemblies) rather than one blended h; but we still can't store L distinct positions.

### Summary: disadvantages we share vs don't

| SSM disadvantage | Do we have it? | Note |
|------------------|----------------|------|
| Fixed-size state bottleneck | **Yes** | We compress context into k winners; can't store L tokens' worth of arbitrary info. |
| No direct position-based access | **Yes** | We don't have "attend to position i"; we have content/association. |
| Weaker needle-in-haystack / exact position recall | **Possibly** | Depends on whether the "needle" is retrievable by **content** (cue → attractor) or only by **position**; we're better at content. |
| Linear recurrence limit | **No** | We have **nonlinear** (top-k, Hebbian) recurrence; more expressive for pattern completion and attractors. |
| Associative / content-based retrieval | **Different** | We're **designed** for association and pattern completion; can be **better** than linear SSMs here, closer to "content-addressable" memory. |

**Bottom line:** Assembly foundation models **will** have some of the **same disadvantages as SSMs** (fixed state, no direct position index, possible limits on position-heavy long-context retrieval). They **differ** in having **nonlinear**, **associative** recurrence, so they may do **better** on **content-based** long-context tasks and **worse** on **position-heavy** ones. The **relation** is: we're in the same **efficiency class** (O(L), fixed state) as SSMs, so we share the same **tradeoff** (efficiency vs direct access); we're not a free lunch. The **design choice** is whether **associative** retrieval (assembly-style) is enough for your use case, or whether you need **position** retrieval (where attention or hybrid designs still win).

---

## 7. Overcoming the limitations: toward transformers, diffusion, and CARD

**Goal:** How to **overcome** the assembly/SSM-like limitations (fixed state, no position access, weaker position-heavy retrieval) so assembly foundation models can compete with **transformers**, **diffusion language models**, or **causal diffusion** (e.g. [CARD](https://arxiv.org/abs/2601.22031)). And what the **CARD** paper implies for our design.

### 7.1 What CARD does (and what it implies)

**CARD (Causal Autoregressive Diffusion)** [arXiv:2601.22031](https://arxiv.org/abs/2601.22031) unifies:

- **Training efficiency of ARMs** (autoregressive): good data efficiency, next-token objective.
- **High-throughput inference of diffusion**: **dense, per-token supervision in a single forward pass** (not one token at a time); **dynamic parallel decoding** at inference (generate variable-length token sequences based on confidence, with KV-caching).

**Mechanisms:**

1. **Causal diffusion under a strictly causal attention mask** — The diffusion process is reformulated so that each position only depends on past tokens. So you get **dense supervision** (loss at every position) in **one forward pass**, like diffusion, but **causal** like ARMs.
2. **Soft-tailed masking** — Preserves **local context** and stabilizes optimization (causal diffusion can be unstable).
3. **Context-aware reweighting** — From signal-to-noise principles; reweights loss by position/SNR.
4. **Dynamic parallel decoding** — Uses KV-cache; generates **multiple tokens in parallel** when the model is confident, instead of strictly one-by-one. So inference latency can drop (fewer steps for the same sequence).

**Results:** Outperforms discrete diffusion baselines; **~3× training latency reduction** vs block diffusion; ARM-level data efficiency + **parallel generation** benefits.

**Implications for assembly models:**

1. **Dense supervision in one pass** — CARD gets per-token loss at *every* position in one forward pass (causal). We currently process context **recurrently** (one token at a time) so training is sequential over L. We could do **causal assembly in parallel**: run assembly dynamics for *every prefix* (tokens 1..t for each t) in one pass with a **causal mask** (position t only sees 1..t). That gives "assembly state at position t" for all t in one forward pass, then **dense loss** (e.g. next-token at every position). So **training efficiency** like CARD: one pass = all positions, better gradient flow.
2. **Preserve local context** — CARD's soft-tailed masking keeps local context from being blurred. We need the same: **local assembly** (e.g. an area that only receives input from the last W tokens) or **hierarchical assemblies** (local + global) so local context isn't lost in the fixed-state bottleneck.
3. **Parallel decoding** — CARD decodes multiple tokens in parallel when confident. We could do **confidence-based parallel decoding**: when readout entropy is low (assembly state is "stable"), predict the next few tokens in parallel and verify with assembly dynamics; or **speculative assembly** (draft + verify).
4. **KV-cache / reuse** — CARD uses KV-cache for fast decoding. Our analogue: **cache assembly states** at a subset of positions (e.g. every 64 tokens) for hierarchical readout or retrieval, so we don't recompute full recurrence from scratch when we need "context at position i."

So CARD suggests: **causal + dense supervision + local context preservation + parallel decoding** are the levers that make efficient-but-capable LMs. We can mirror those in the assembly setting.

### 7.2 How to overcome assembly/SSM limitations (concrete directions)

**A. Be more like transformers (direct access, in-context learning)**

1. **Hybrid: add limited attention** — Add a **small attention** over a **sliding window** of recent assembly states (or recent token embeddings) so we have **local** direct access — O(window²), not O(L²). The recurrent assembly core stays O(L); attention gives position access for the **last W tokens**. So we keep efficiency but recover "needle in haystack" for **recent** context.
2. **Hierarchical assemblies** — Multiple areas with different "receptive fields": e.g. **local** (last 10 tokens), **mid** (last 100), **global** (full context compressed). Readout can **combine** or **attend over** these scales. So we get **multi-scale** context; recent/fine-grained info lives in local, long-range in global. Addresses "information blurring" by preserving local context explicitly (like CARD's soft-tailed masking).
3. **External memory / retrieval** — A **compressed memory** (e.g. key-value store) storing assembly states (or token embeddings) at a **subset** of positions; **content-based retrieval** (assembly state as query) to "recall" relevant items. So we get selective access without full L² attention — **content-addressable** retrieval that assemblies are good at, plus the ability to pull in specific long-range info.
4. **Larger or dynamic state** — Increase **k** or **number of areas** so the bottleneck holds more information (at cost of compute). Or **dynamic k**: use more winners when context is long or when uncertainty is high.

**B. Be more like diffusion / CARD (dense supervision, parallel decoding)**

1. **Causal assembly in parallel (CARD-like)** — For training: compute **assembly state at every position t** in one forward pass. For each t, state_t = assembly_dynamics(tokens[1..t]) with **causal** dependency (no future). Implement by: **unrolling** the recurrence in parallel (like a transformer decoder where each "layer" is an assembly step), or by **prefix caching** (compute state for prefix 1..t for t = 1,2,...,L in one batched pass). Then apply **dense loss** (e.g. next-token cross-entropy at every t) with optional **context-aware reweighting** (e.g. by assembly stability or SNR). So we get **dense per-token supervision in one pass** — same benefit as CARD for training efficiency.
2. **Soft-tailed / local context** — **Local assembly area**: only receives input from the last W tokens (window); **global** area gets projected from local or from full recurrence. So local context is **preserved** (like CARD's soft-tailed masking); global handles long-range. Reduces optimization instability and information loss for recent tokens.
3. **Parallel decoding** — **Confidence-based**: when readout entropy < threshold, predict next **K** tokens in parallel (e.g. greedy or beam), then run assembly dynamics for the chosen continuation and verify. Or **speculative**: small draft model proposes next few tokens; assembly model runs dynamics on the proposal and accepts/refines. So we get **variable-length parallel generation** like CARD, reducing inference latency.
4. **Reweighting (CARD-like)** — **Context-aware reweighting** of the loss: e.g. by **assembly stability** (reward positions where the assembly converged) or by **signal-to-noise** (e.g. reweight by inverse variance of readout). Improves training stability and data efficiency.

**C. Be more like diffusion language models (generative process)**

- Diffusion LMs **denoise** from noise to tokens over steps. We could have an **assembly diffusion** variant: **forward** = add noise (e.g. perturb winners, or blend with random assembly); **reverse** = assembly dynamics that "denoise" (converge to a clean assembly state). Train the reverse process to predict clean state from noisy. Then **sampling** = run reverse dynamics from noise. That would align our **dynamics** (recurrence, attractors) with the **generative process** of diffusion (multi-step denoising). More speculative; would need a clear formulation of "assembly noise" and "assembly denoising."

### 7.3 Summary: what to implement first

| Direction | What | Why |
|-----------|------|-----|
| **Hierarchical / local assembly** | Local area (last W tokens) + global area (full recurrence) | Preserve local context; address blurring; CARD-like soft-tailed effect. |
| **Causal assembly in parallel** | Assembly state at every position t in one pass; dense loss | Training efficiency like CARD; dense gradients; 3×-style speedup potential. |
| **Hybrid attention** | Small attention over last W assembly states or tokens | Recover position access for **recent** context; needle-in-haystack for local. |
| **Parallel decoding** | Confidence-based or speculative multi-token generation | Inference latency like CARD; variable-length parallel decode. |
| **Context-aware reweighting** | Reweight loss by stability or SNR | Training stability; data efficiency. |

**Bottom line:** To make assembly foundation models **competitive with transformers or diffusion/CARD**, we don't need to abandon assemblies — we need to **add** or **reformulate** along the dimensions above: (1) **preserve local context** (hierarchical/local assembly, soft-tailed effect); (2) **dense causal supervision** (assembly state at every position in one pass, like CARD); (3) **limited position access** where it matters (hybrid attention over recent context, or external memory); (4) **parallel decoding** (confidence-based or speculative). CARD shows that **causal + dense supervision + local preservation + parallel decoding** is a viable paradigm for efficient LLMs; we can instantiate that in the assembly setting and keep the **efficiency** (O(L), sparse, associative) while closing the gap on **position-heavy** and **training/inference** efficiency.

---

## 8. Implications if it works: much more data-efficient models?

**If the assembly dynamical system actually works and can be used at scale, does it mean we can make *much* more data-efficient models?**

**Yes — it *could* mean that.** The dynamical system (sparse, Hebbian, associative, attractor) gives **strong inductive biases** that transformers largely lack. If those biases align with how language and concepts work, the model could reach **similar quality with far less data**, or **better quality with the same data**. Here’s why.

### 8.1 Why assembly foundation models could be much more data-efficient

1. **Strong inductive bias (structure, not just scale)**  
   Assemblies have **built-in structure**: (a) **sparsity** — only k “concepts” active at once; (b) **Hebbian** — co-active neurons strengthen; (c) **association** (A + B → C) and **merge** — composition by construction; (d) **attractor dynamics** — convergence to stable states. So the hypothesis space is **constrained** toward “concepts + associations + attractors.” Transformers have weak inductive bias (attention over everything); they rely on **scale and data** to discover structure. If assemblies’ bias matches reality (concepts, composition, stability), they could **need far less data** to learn good representations — same quality with **10× or more less data** is a plausible target, not a guarantee.

2. **Sparse representations → less overfitting, better generalization**  
   Sparse coding and sparse representations often **generalize better** from fewer examples — fewer effective parameters, less memorization. Assembly state is **k winners** (and sparse W). So **per example**, the model might overfit less and **transfer** better, i.e. **more data-efficient** (same data → better validation; or same validation with less data).

3. **Attractors = one concept, many inputs**  
   If the dynamics **converge** to attractors, many inputs (noisy, paraphrased, varied) map to the **same** attractor (concept). So the model learns **one stable representation per concept** instead of one representation per token combination. That’s **concept-level** learning with **example-level** data — i.e. **more data-efficient** concept acquisition.

4. **Curriculum is first-class**  
   We’re already designing **curriculum** (easy→hard, short→long, high-freq→rare). Curriculum learning is known to improve **data efficiency** — the right order of examples reduces sample complexity. So assembly + curriculum could be **more data-efficient** than “random shuffle of the internet” for the same final capability.

5. **Hebbian = local learning from every example**  
   Hebbian updates are **local** (co-activation → strengthen). So **every** example updates the connectivity in a useful way (co-occurring assemblies get stronger), without needing a global loss. That could mean **each example does more useful work** than in a pure backprop model where only the final loss gradient flows. Combined with a readout (or STE) trained on a task loss, we might get **better sample efficiency** — fewer examples to reach the same loss.

6. **Compositionality and reuse**  
   Association and merge are **compositional** — same assembly for “dog” can be reused in “dog runs,” “big dog,” etc. So we learn **reusable building blocks**; transfer and few-shot could be better. That’s **data efficiency across tasks** — learn a concept once, reuse everywhere.

7. **Effective capacity vs data**  
   Transformers are heavily overparameterized and need huge data to generalize. Assemblies are **sparse** (k active, sparse W) — **effective capacity** may be smaller and better matched to “concepts + relations.” That can mean **better generalization from less data** (as long as capacity is sufficient for the task).

### 8.2 What “much more data-efficient” could mean in practice

- **Same quality with less data** — e.g. same perplexity or downstream performance with **5–10× (or more) less** pre-training data.
- **Same data, better quality** — e.g. better perplexity or few-shot with the same corpus.
- **Faster convergence** — fewer **epochs** or **steps** to reach the same loss (each example is used more effectively).
- **Better few-shot / in-context** — if assembly representations are more compositional and stable, few-shot and transfer could improve (data efficiency at **adaptation** time).

### 8.3 Caveat

We don’t have **empirical proof** yet — no assembly foundation model trained at scale and compared to transformers on data-efficiency curves. So “much more data-efficient” is a **plausible implication** of the dynamical system’s **inductive biases**, not an established fact. The **claim** to test is: *with matched quality, assembly foundation models require significantly less data (or fewer steps) than transformer baselines.* That’s a central **hypothesis** if the system works.

### 8.4 Bottom line

**If the assembly dynamical system actually works and can be used at scale, it *could* mean we can make much more data-efficient models** — because of strong inductive bias (sparsity, Hebbian, association, attractors), sparse representations (less overfitting), attractor-based concept learning (one concept, many inputs), curriculum, and compositional reuse. The **implication** is: **same capability with far less data**, or **better capability with the same data**, is a realistic target to aim for and to **validate empirically** once we have a working assembly foundation model.

---

## 9. Further insights: what this implies for you

Synthesis of deeper threads and how they should shape the program.

### 9.1 Core vs periphery: keep the identity clear

**Tension:** The calculus is motivated by **cortical assemblies** (sparse, Hebbian, local). Foundation models are **engineered** for scale (data, compute, quality). If we add hybrid attention, dense causal training, parallel decoding, curriculum, STE — are we still "assemblies" or "assemblies + a lot of transformer-like machinery"?

**Insight:** Treat **assemblies as the *core representation engine*** (sparse, recurrent, associative, attractor). The transformer-like pieces (attention over recent context, dense loss, parallel decode) are **interfaces** or **readout layers** that make that core usable at scale. So the *identity* of the system stays "assembly-based"; we're not abandoning the core, we're making it **deployable**. That keeps the story clean: core = assembly dynamics; periphery = what we need to match transformer/diffusion efficiency and benchmarks.

### 9.2 What would falsify the program?

**Data efficiency:** If we build an assembly foundation model and it's **not** more data-efficient (same data → worse quality, or we need **more** data for same quality), that suggests the inductive biases are **wrong** or **insufficient** for language. So the **falsifiable prediction** is: *assembly vs transformer data-efficiency curves* (quality vs data size, matched compute).

**Compute efficiency:** If at long context we need **more** FLOPs than we thought to match quality (e.g. T or n blows up), the O(L) advantage is eaten by something else. So the **falsifiable prediction** is: *assembly vs transformer FLOPs (and memory) to reach same quality at long L*.

Being explicit about **what would falsify** the program tells you what to **measure first** and what to **report** — and keeps the claim scientific.

### 9.3 The first win might be narrow

We may not get "assembly foundation model beats transformer everywhere." We might get:

- **Long context** — assembly wins on compute/memory for large L, with acceptable quality.
- **Low data** — assembly wins on data efficiency in a **particular regime** (e.g. small corpus, curriculum, or few-shot).
- **Embodied / multimodal** — assembly wins because association and merge map naturally to "bind perception to action" or "bind modalities."

**Insight:** Find the **narrow regime** where assembly's biases give a **clear win** (long context? low data? embodied?) and **establish that first**, then broaden. Don't require "beat transformer on everything" for the first paper — require "beat transformer on X" where X is the regime assembly is designed for.

### 9.4 Theory before scale

We have **empirical** results (convergence, scaling, phase diagram, association, merge) but "needs theory" (Q02, Q03). If we go straight to "assembly foundation model" without **one derived theory result** (phase boundary, convergence rate, or capacity), we're building on sand — we won't know *why* it works or when it will break.

**Insight:** **One derived theory result** is a **prerequisite** for confidence in scaling up. It doesn't have to be complete; it has to be **testable** (e.g. "mean-field predicts critical p_c; we measure p_c and compare"). That connects to PRIORITIES_AND_GAPS: do the derivation **before** or **in parallel with** scaling experiments.

### 9.5 Attractor claim requires attractor demo

Q21 (autonomous recurrence not fully supported) and Q10 (noise robustness) are **blockers** for the "attractor dynamics" story. If we claim "attractor dynamics" but can't run pure recurrence or show recovery from perturbation, the claim is weak.

**Insight:** **Fix Q21/Q10 before** claiming "assembly foundation models" or "data-efficient attractor-based models" in a paper. Otherwise we're claiming attractors without being able to demonstrate them. That's a **sequencing** insight: validate the core dynamics (autonomous recurrence, recovery) before scaling to foundation-model scale.

### 9.6 Data efficiency vs compute efficiency (measure both)

We've argued assembly could be **both** more data-efficient (inductive bias) and more compute-efficient (O(L), sparse). They're **different**: data efficiency = fewer examples to reach quality; compute efficiency = fewer FLOPs (or less memory) per token. We could be compute-efficient but data-hungry, or data-efficient but compute-heavy.

**Insight:** **Measure both** separately. Report: (a) *quality vs data size* (same compute budget) — data efficiency; (b) *quality vs FLOPs (and memory) at long L* — compute efficiency. Being clear which one we're optimizing for first (and which one we're claiming) avoids confusion and overclaiming.

### 9.7 Biological inspiration vs engineered system

Cortical assemblies in biology are **not** trained on web-scale text with next-token loss. They're shaped by development, embodied experience, and local plasticity. So "assembly foundation model" is **inspired by** biology but **not** a model of how the brain learns language.

**Insight:** We're building an **engineered** system that uses **assembly-like** dynamics (sparse, Hebbian, associative) for **foundation-model** goals (scale, data, quality). The biological connection is **motivation and plausibility**, not a claim that we're replicating the brain. Stating that clearly avoids overclaiming and keeps the narrative honest.

### 9.8 Binding and compositionality: a mechanistic angle

Association (A + B → C) and merge are **binding** operations: they bind two assemblies into one. In cognitive science, the "binding problem" is how the brain binds features (color, shape, location) into objects. Assemblies give a **concrete** binding mechanism (Hebbian co-activation → shared assembly).

**Insight:** Assembly foundation models could be a **testbed for binding** in language — e.g. bind "red" + "ball" → "red ball" assembly. If that works, it's not just efficiency; it's a **mechanistic** account of compositionality. That's a deeper **scientific** angle for papers and for cognitive/neuro audiences: we're not only building efficient models; we're offering a **mechanism** for how composition could work (binding via association/merge).

### 9.9 Failure modes to watch

- **Quality plateau** — assembly quality doesn't scale with n or data the way we hope; we need n or T so large that O(n log k) is no longer cheap.
- **Overfit to curriculum** — model generalizes poorly when we change data order or domain; curriculum becomes a crutch.
- **Position-heavy tasks dominate benchmarks** — needle-in-haystack and exact-position recall are essential for LLM benchmarks; hybrid attention isn't enough and we can't close the gap.
- **Training instability** — dense causal assembly training (CARD-like) or STE leads to instability; we need more tricks than we thought.

Being explicit about **failure modes** helps prioritize what to **validate first** and what to **monitor** when scaling.

### 9.10 A minimal path to impact

A concrete **sequence** that gets from where you are to a defensible "assembly foundation model" claim:

1. **Fix Q21/Q10** — Demonstrate autonomous recurrence and recovery from perturbation (attractor demo). Without this, the core claim is weak.
2. **One derived theory result** — Phase boundary, or convergence rate, or capacity bound; compare to experiments. Gives "why it works" and "when it might break."
3. **Small-scale data-efficiency test** — Build a **small** assembly language model (small n, small vocab, curriculum) and measure **data-efficiency** vs a small transformer: same data size, same compute budget → compare quality (e.g. perplexity, downstream). That's the first **falsifiable** test of "much more data-efficient."
4. **If (3) holds** — Scale up (larger n, web-scale data) and add CARD-like and hybrid pieces (dense causal training, local assembly, parallel decode, optional hybrid attention).
5. **Long-context benchmark** — Assembly vs transformer at L = 10k–100k on compute, memory, and quality. Establishes the **compute-efficiency** claim for long context.

That path has **clear milestones** and **falsifiable** steps. The first **publishable** win could be (3) — "assembly language model is more data-efficient than transformer at small scale" — even before full foundation-model scale.

### 9.11 Bottom line

The implications for you: (1) **Keep core = assembly, periphery = interface** so the identity is clear. (2) **Define falsifiable predictions** (data-efficiency curves, compute-efficiency at long L) and measure them. (3) **Aim for a narrow first win** (long context, or low data, or embodied) rather than "beat transformer everywhere." (4) **Get one theory result** before or alongside scale. (5) **Fix Q21/Q10** before big attractor/foundation claims. (6) **Measure data and compute efficiency separately.** (7) **Frame biology as inspiration**, not replication. (8) **Use binding/compositionality** as a mechanistic scientific angle. (9) **Watch failure modes** and validate the weak links first. (10) **Follow a minimal path**: attractor demo → theory → small data-efficiency test → scale → long-context benchmark.
