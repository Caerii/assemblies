# Spiking Foundations Analysis

## What We Have

### 1. Assembly Calculus (`brain.py`)

**Model Type:** Rate-based, discrete time

**Computation:**
```
At each step:
  input_i = sum_j(w_ji * active_j)
  winners = argmax_k(input)
  w_ji *= (1 + beta)  for active pairs
```

**Strengths:**
- Fast simulation
- Mathematically tractable
- Sparse efficiency

**Limitations:**
- No temporal dynamics
- No attractor states
- No autonomous activity

### 2. Spiking Simulation (`neural_testing/neuron_sim.py`)

**Model Type:** Leaky Integrate-and-Fire with STDP

**Computation:**
```
Continuous time:
  dV/dt = -(V - V_rest)/tau + I/C
  
  if V > threshold:
    spike, V = V_reset
    
  STDP:
    dw = A+ * exp(-dt/tau+)  if pre before post
    dw = A- * exp(dt/tau-)   if post before pre
```

**Features:**
- Membrane potential dynamics
- Refractory periods
- STDP (LTP and LTD)
- Conductance-based synapses
- Homeostatic plasticity
- E/I balance (80/20)

**Parameters:**
| Parameter | Value |
|-----------|-------|
| Neurons | 200 |
| dt | 0.5e-5 s |
| Threshold | -45 mV |
| Reset | -65 mV |
| Rest | -70 mV |
| STDP A+ | 0.015 |
| STDP A- | -0.015 |
| STDP tau | 20 ms |

### 3. Hodgkin-Huxley (`cpp/python_implementations/`)

**Model Type:** Biophysical ion channels

**Computation:**
```
dV/dt = (I - I_Na - I_K - I_L) / C

I_Na = g_Na * m³h * (V - E_Na)
I_K = g_K * n⁴ * (V - E_K)
I_L = g_L * (V - E_L)

dm/dt = alpha_m(V)(1-m) - beta_m(V)m
dh/dt = alpha_h(V)(1-h) - beta_h(V)h
dn/dt = alpha_n(V)(1-n) - beta_n(V)n
```

**Status:** Implemented but not integrated with Assembly Calculus

---

## The Hierarchy of Models

```
ABSTRACT ←————————————————————————→ BIOPHYSICAL

Assembly     LIF        Izhikevich    Hodgkin-
Calculus                              Huxley
   ↓          ↓            ↓            ↓
Binary    Membrane    2-variable    Ion channel
neurons   potential   dynamics      dynamics
   ↓          ↓            ↓            ↓
Fast      Medium      Slow          Very slow
```

---

## What's Missing for True Dynamics

### Current Assembly Calculus Problems

1. **No persistence:** Assemblies vanish without input
2. **No temporal order:** Can't learn sequences from timing
3. **No pattern completion:** Partial cues don't work
4. **No attractors:** Weights don't create stable states

### What Spiking Would Add

1. **Persistence:** Membrane potential decays slowly, activity can persist
2. **Temporal coding:** STDP learns causal relationships
3. **Attractors:** Strong internal connections maintain patterns
4. **Oscillations:** Rhythmic activity enables binding

---

## The Gap to Bridge

### Current State
```
Assembly Calculus          Spiking Neurons
      ↓                          ↓
Fast, no dynamics         Slow, has dynamics
      ↓                          ↓
   brain.py              neural_testing/neuron_sim.py
```

### Target State
```
Spiking Assembly Calculus
         ↓
Assembly structure + Spiking dynamics
         ↓
- k-winners (from Assembly Calculus)
- Membrane potentials (from LIF)
- STDP learning (from spiking)
- Sparse connectivity (for efficiency)
```

### What This Would Enable

| Capability | Current | With Spiking |
|------------|---------|--------------|
| Pattern recognition | ✅ | ✅ |
| Associations | ✅ | ✅ |
| Persistence without input | ❌ | ✅ |
| Sequence learning | ❌ | ✅ |
| Pattern completion | ❌ | ✅ |
| Attractor dynamics | ❌ | ✅ |

---

## Mathematical Foundation for Spiking Assemblies

### LIF with Assembly Constraint

For each neuron i in area A:
```
dV_i/dt = -(V_i - V_rest)/tau + I_i/C

I_i = I_ext + sum_j(w_ji * s_j(t))

s_j(t) = synaptic kernel after spike

if V_i > threshold:
  spike
  V_i = V_reset
```

### Assembly Formation

Instead of instant winner-take-all, use soft competition:
```
Inhibitory current:
  I_inh = -g_inh * (sum of all activity in area)

This creates effective k-winners through dynamics, not selection.
```

### STDP for Assembly Learning

```
For each synapse w_ji:
  if neuron j spikes at t_j and neuron i spikes at t_i:
    dt = t_i - t_j
    
    if dt > 0:  # causal (j before i)
      w_ji += A+ * exp(-dt/tau+)
    else:       # anti-causal (i before j)
      w_ji += A- * exp(dt/tau-)
```

This naturally strengthens connections within assemblies (neurons that fire together) and weakens others.

---

## Roadmap

### Phase 1: Understand Current Code
- [x] Document Assembly Calculus primitives
- [x] Document spiking neuron code
- [x] Identify the gap

### Phase 2: Design Spiking Assemblies
- [ ] Define LIF + soft-WTA dynamics
- [ ] Design STDP rule for assemblies
- [ ] Plan sparse implementation

### Phase 3: Implement
- [ ] Create SpikingArea class
- [ ] Integrate with Brain class
- [ ] Test attractor formation

### Phase 4: Validate
- [ ] Test persistence without input
- [ ] Test pattern completion
- [ ] Test sequence learning

---

## Key Equations Summary

### Assembly Calculus (Current)
```
winners = argmax_k(W @ active)
W *= (1 + beta)  for active pairs
```

### LIF Dynamics (Needed)
```
dV/dt = -(V - V_rest)/tau + I/C
spike when V > threshold
```

### STDP (Needed)
```
dw/dt = A+ exp(-dt/tau+) - A- exp(dt/tau-)
```

### Soft WTA (Needed)
```
I_inh = -g_inh * sum(activity)
```

---

## Next Steps

1. **Prototype:** Create minimal spiking assembly simulation
2. **Test:** Can assemblies persist without input?
3. **Iterate:** Tune parameters for stable attractors
4. **Scale:** Optimize for larger networks

