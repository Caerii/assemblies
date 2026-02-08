# Control speed from CUDA assembly simulation

**Goal:** Use existing CUDA benchmarks to derive the **maximum control-loop frequency (Hz)** you can get when each control step is one assembly projection round (sensory → motor, one or several areas).

---

## Biological realism of the control rate

Your CUDA assembly step is **fast enough for biologically plausible closed-loop control**:

- **Cortical "decision" / association:** ~10–100 ms per cycle in vivo → **10–100 Hz**.
- **Spinal / subcortical reflex and motor:** ~1–10 ms → **100–1000 Hz**.
- **Your assembly step:** sub-ms to a few ms for n ≤ 10M → **hundreds to tens of thousands Hz** per projection round.

So you can run an assembly-based controller at **100–1000 Hz** (or higher for smaller brains) and still be **within or above** the timescales of cortical and spinal loops. That gives you "fast biologically realistic" control in the sense of **loop rates**, not just single-neuron spike rates. The assembly calculus then operates at a **population / assembly timescale** (one projection ≈ one "macro-step"), which aligns with the idea that behavior is driven by assemblies evolving on ~10–100 ms, not by every single spike.

---

## 1. What "one step" means for control

For motor control, **one control step** = one round of:

1. Feed sensor/state into assembly areas (or use current winners).
2. Run one `project()` (or equivalent) so all relevant areas update (sensory → association → motor).
3. Read out motor assembly → continuous command (e.g. joint torques).

So **time per control step** = time for one full projection round (all areas that participate in the control loop). That determines **control rate** = 1 / (time per step) in Hz.

---

## 2. Benchmark data you already have

### 2.1 Consolidated CUDA kernels (one area, one projection)

From `cpp/cuda_kernels/tests/algorithmic_improvements_test_20250923_113504.json` (optimized kernels, single-area "step"):

| Neurons (n) | Active (k) | Top-k | Time per step (ms) | Steps/sec (Hz) |
|-------------|------------|-------|---------------------|----------------|
| 100,000     | 1,000      | 100   | 0.00246             | **405,771**    |
| 1,000,000   | 10,000     | 1,000 | 0.0354              | **28,286**     |
| 10,000,000  | 100,000    | 1,000 | 0.354               | **2,829**      |
| 100,000,000 | 1,000,000  | 1,000 | 3.535               | **283**        |

So **one area, one projection**: sub-millisecond for n ≤ 10M; a few ms for n = 100M.

### 2.2 Full brain step (multiple areas)

- `cpp/tests/test_ms_per_step.py` targets **UltraOptimizedCUDABrainV2** with several areas and reports:
  - `avg_step_time` (s) → ms per step = `avg_step_time * 1000`
  - `steps_per_second` = 1 / `avg_step_time` → **control rate in Hz**

Example scales there: 50k–1M neurons, 3–5 areas. So "one step" = one full brain update (all areas). That's exactly one control step if the whole loop is: sense → project once → read out motor.

### 2.3 CuPy (single-area projection)

- `src/lexicon/cupy_assembly_kernels.py`: n = 10k, 100k, 1M, 10M; k = 50; reports **ms/projection** and **projections/sec**.
- `src/lexicon/cupy_assembly_kernels_batched.py`: 8 areas, n = 1M, k = 50; reports total ms per round and **proj/sec** (for 8 areas together).

So you can get **time per projection per area** and **time per full multi-area round** from these.

---

## 3. Deriving control speed

- **One area in the loop:**  
  Control rate (Hz) = steps_per_second from that area's benchmark (e.g. 28,286 Hz at n=1M from consolidated kernels).

- **N_area areas in the loop:**  
  Assume time per control step ≈ sum of per-area projection times (or use a full-brain benchmark). Then:
  - **Control rate (Hz) = 1 / (time_per_step_sec)**  
  - If each area costs `t_area` ms: `time_per_step_sec = N_area * t_area / 1000`, so  
  - **Control rate ≈ 1000 / (N_area * t_area)** Hz.

Using the consolidated kernel numbers (single area):

- n = 100k:  t ≈ 0.0025 ms → **400k / N_area** Hz.
- n = 1M:   t ≈ 0.035 ms  → **28.6k / N_area** Hz.
- n = 10M:  t ≈ 0.35 ms   → **2.86k / N_area** Hz.
- n = 100M: t ≈ 3.54 ms   → **282 / N_area** Hz.

Example: 5 areas, n = 1M each → t_step ≈ 0.18 ms → **≈ 5,500 Hz** control rate.  
So **control speed is in the kilohertz range** for million-neuron, few-area brains; **hundreds of Hz** for tens of millions of neurons or many areas.

---

## 4. Comparison to typical robotics control rates

| Application              | Typical rate | Period   |
|--------------------------|-------------|----------|
| Slow / whole-body        | 50–100 Hz   | 10–20 ms |
| Standard servo/robot     | 100–500 Hz  | 2–10 ms  |
| Fast (e.g. arms, legs)   | 500–1000 Hz | 1–2 ms   |
| Very fast (force/contact)| 1–5 kHz     | 0.2–1 ms |

So:

- **n ≤ 1M, few areas (e.g. 3–5):** You get **thousands of Hz** from the kernels above → control rate is **not** limited by assembly step time; you're easily above 1 kHz and can match or exceed "very fast" control.
- **n ~ 10M, several areas:** Still **hundreds–low thousands Hz** → fine for 100–500 Hz and often 1 kHz.
- **n ~ 100M, many areas:** **Tens to low hundreds Hz** → suitable for 50–200 Hz control; still in the "standard" robotics range.

So **you can derive the speed of control** directly from your CUDA "time per step" (or time per projection per area): **control rate (Hz) = 1 / (time per control step in seconds)**.

---

## 5. How to state it in a paper or doc

- **Measured:** "On a single GPU, one assembly projection step for an area of n = 10^6, k = 10^3 takes ~0.035 ms (consolidated optimized kernels)."
- **Derived:** "A control step consisting of one projection round across 5 such areas therefore takes ~0.18 ms, yielding a **maximum control rate of ~5.5 kHz**; in practice we run the robot at 1 kHz to leave headroom for sensing and actuation."
- **Scaling:** "Control rate scales as 1/(N_area × t_area(n,k)); for our kernels, t_area is sub-ms for n ≤ 10^7, so assembly-based control can run at 100–1000 Hz for brains with millions to tens of millions of neurons and several areas."

---

## 6. What to run to get your own numbers

1. **Per-area time (CUDA):**  
   Run the consolidated kernel benchmark (or equivalent) for your chosen (n, k) and read **optimized_time_ms** → steps/sec = 1000 / optimized_time_ms.

2. **Full brain step (multi-area):**  
   Run `test_ms_per_step.py` (or your full brain benchmark) with the intended n, k, and number of areas → use **avg_step_time** and **steps_per_second** as **time per control step** and **control rate (Hz)**.

3. **CuPy path:**  
   Run `cupy_assembly_kernels.benchmark()` and/or the batched benchmark for your (n, k, num_areas) → use reported ms per projection and projections/sec; for N_area areas, control rate ≈ (projections/sec) / N_area if each control step does one projection per area, or use the batched "total ms per round" as time per control step.

Then: **Control rate (Hz) = 1 / (time per control step in seconds)**.
