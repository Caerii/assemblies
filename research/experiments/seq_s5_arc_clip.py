"""POST HOC diagnostic (Addendum 5 of PREREG_s5_cliff_anatomy.md; labelled): relocation is the CLIP.

import inspect, sys
sys.path.insert(0, 'F:/Github/assemblies'); sys.path.insert(0, 'F:/Github/assemblies/research/experiments')
import numpy as np, torch
from neural_assemblies.core.brain import Brain
from neural_assemblies.core.torch_engine._hashed_fsm import HashedArcFSM
from neural_assemblies.programs.word_problems import GROUPS, word_problem_fsm
from seq_s5_word_problem import BETA, K, ORGAN_P, REFRACTED, sizes

group = GROUPS['Z60']()
states, symbols, transitions = word_problem_fsm(group)
n_arc, n_state = sizes(group, len(symbols))
w_max = inspect.signature(Brain).parameters['w_max'].default
seeds = [42, 43, 44, 45]
P = 30
fsm = HashedArcFSM(seeds, states, symbols, transitions, n_arc=n_arc, n_state=n_state,
                   k=K, p=ORGAN_P, beta=BETA, refracted_strength=REFRACTED, w_max=w_max,
                   norm_init=False, max_potentiations=256, prefix='_wp', zero_or_size=False)
arcs = {}
for pres in range(P):
    for (fr, sym), to in fsm.table.items():
        fsm.train_transition(sym, fr, to)
        if pres == 14:
            arcs[(fr, sym)] = fsm.arc.winners.clone()
lost_pot, kept_pot, lost_fib, kept_fib, lost_bias, kept_bias = [], [], [], [], [], []
lost_base, kept_base, lost_raw, kept_raw = [], [], [], []
for (fr, sym), to in fsm.table.items():
    fsm.arc.inhibit(); fsm.cue_state(fr)
    fsm.sym.set_words(fsm._idx(sym, fsm.symbol_index))
    fsm.core.conjoin([fsm.state_arc, fsm.sym], freeze=True)
    test = fsm.arc.winners
    a15 = arcs[(fr, sym)]
    g = fsm.symbol_index[sym]
    blk = fsm.blocks[fsm.state_index[fr]]
    for b in range(len(seeds)):
        t = set(test[b].tolist()); o = set(a15[b].tolist())
        pot = fsm.sym.pot[g, b]                                   # [n_arc]
        C = fsm.state_arc.C[b][blk].float()                       # [k, n_arc] counts from the block
        fib = C.mean(0)                                           # mean count per arc column
        bias = fsm.arc.bias[b]
        base = fsm.sym.base[g, b]
        raw = torch.zeros(len(seeds), n_arc, device='cuda'); fsm.state_arc.contribute(raw, fsm.state.winners); fsm.sym.contribute(raw); raw = raw[b]
        for j in o - t:
            lost_pot.append(int(pot[j])); lost_fib.append(float(fib[j])); lost_bias.append(float(bias[j])); lost_base.append(float(base[j])); lost_raw.append(float(raw[j]))
        for j in o & t:
            kept_pot.append(int(pot[j])); kept_fib.append(float(fib[j])); kept_bias.append(float(bias[j])); kept_base.append(float(base[j])); kept_raw.append(float(raw[j]))
lp, kp = np.array(lost_pot), np.array(kept_pot)
print(f'P={P}: lost {len(lp)} kept {len(kp)} (of {len(lp)+len(kp)} presentation-15 arc neurons)')
print(f'  symbol-stimulus potentiation count: lost median {np.median(lp):.0f} p10 {np.percentile(lp,10):.0f}  |  '
      f'kept median {np.median(kp):.0f} p90 {np.percentile(kp,90):.0f}  |  '
      f'lost with count >= 31 (clip): {(lp>=31).mean()*100:.1f}%  kept >= 31: {(kp>=31).mean()*100:.1f}%')
print(f'  state->arc mean count from the block: lost {np.median(lost_fib):.1f}  kept {np.median(kept_fib):.1f}')
print(f'  accumulated bias: lost median {np.median(lost_bias):.1f}  kept median {np.median(kept_bias):.1f}')

lb, kb = np.array(lost_base), np.array(kept_base); lr, kr = np.array(lost_raw), np.array(kept_raw)
hi = 20.0 * max(1.0, K * ORGAN_P)
print(f'  stimulus BASE (present rows): lost median {np.median(lb):.1f} p10 {np.percentile(lb,10):.1f} | kept median {np.median(kb):.1f} p90 {np.percentile(kb,90):.1f}')
print(f'  clipped at P=30 (base * 1.1^30 >= {hi:.0f}, i.e. base >= {hi/1.1**30:.1f}): lost {(lb*1.1**30>=hi).mean()*100:.1f}%  kept {(kb*1.1**30>=hi).mean()*100:.1f}%')
print(f'  test-time raw drive: lost median {np.median(lr):.0f}  kept median {np.median(kr):.0f};  net (raw - bias): lost {np.median(lr-np.array(lost_bias)):.0f}  kept {np.median(kr-np.array(kept_bias)):.0f}')
