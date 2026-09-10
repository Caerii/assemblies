"""POST HOC diagnostic (Addendum 5 of PREREG_s5_cliff_anatomy.md; labelled): arc drift across presentations.

Post hoc diagnostic for Addendum 5 (labelled as such): does the ARC
assembly of a (state, symbol) pair DRIFT across presentations under
refraction, so that the arc at test is not the arc that was potentiated?
Z60, 4 brains, presentations 15 and 30: overlap of the frozen test-time arc
with the arc at each training presentation of the same pair, plus the
STATE drive of the block's weakest member and of the best outsider."""
import inspect, json, os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, _ROOT); sys.path.insert(0, os.path.join(_ROOT, 'research', 'experiments'))
import numpy as np, torch
from neural_assemblies.core.brain import Brain
from neural_assemblies.core.torch_engine._hashed_fsm import HashedArcFSM
from neural_assemblies.programs.word_problems import GROUPS, word_problem_fsm
from seq_s5_word_problem import BETA, K, ORGAN_P, REFRACTED, sizes
from _results import results_path
OUT = {}

group = GROUPS['Z60']()
states, symbols, transitions = word_problem_fsm(group)
n_arc, n_state = sizes(group, len(symbols))
w_max = inspect.signature(Brain).parameters['w_max'].default
seeds = [42, 43, 44, 45]
for P in (15, 30):
    fsm = HashedArcFSM(seeds, states, symbols, transitions, n_arc=n_arc, n_state=n_state,
                       k=K, p=ORGAN_P, beta=BETA, refracted_strength=REFRACTED, w_max=w_max,
                       norm_init=False, max_potentiations=256, prefix='_wp', zero_or_size=False)
    arcs = {}   # (state, sym) -> list over presentations of [B, k] arc winners
    for _ in range(P):
        for (fr, sym), to in fsm.table.items():
            fsm.train_transition(sym, fr, to)
            arcs.setdefault((fr, sym), []).append(fsm.arc.winners.clone())
    # test-time arc for every pair, frozen; overlap with each presentation's arc
    ov_by_age = np.zeros((P,)); n = 0
    weak, best = [], []
    for (fr, sym), to in fsm.table.items():
        fsm.arc.inhibit(); fsm.cue_state(fr)
        fsm.sym.set_words(fsm._idx(sym, fsm.symbol_index))
        fsm.core.conjoin([fsm.state_arc, fsm.sym], freeze=True)
        test = fsm.arc.winners
        for i, a in enumerate(arcs[(fr, sym)]):
            hit = (test.unsqueeze(2) == a.unsqueeze(1)).any(2).float().mean(1)   # [B]
            ov_by_age[i] += float(hit.mean())
        n += 1
        drive = torch.zeros(len(seeds), n_state, device='cuda')
        fsm.arc_state.contribute(drive, test)
        tgt = fsm.state_index[to]
        blk = drive[:, tgt * K:(tgt + 1) * K]
        out = drive.clone(); out[:, tgt * K:(tgt + 1) * K] = -1
        weak.append(blk.min(1).values.cpu().numpy()); best.append(out.max(1).values.cpu().numpy())
    ov_by_age /= n
    weak = np.concatenate(weak); best = np.concatenate(best)
    g = (1 + BETA) ** P
    print(f'P={P} gain {g:.1f}: test-arc overlap with presentation #1..#{P}: '
          + ' '.join(f'{v:.2f}' for v in ov_by_age))
    print(f'   block weakest member drive: mean {weak.mean():.1f} min {weak.min():.1f} p5 {np.percentile(weak,5):.1f}; '
          f'best outsider drive: mean {best.mean():.1f} max {best.max():.1f} p95 {np.percentile(best,95):.1f}; '
          f'pairs with outsider >= weakest: {(best >= weak).mean()*100:.2f}%', flush=True)
    OUT[str(P)] = dict(overlap_by_presentation=[float(v) for v in ov_by_age], weak_min=float(weak.min()), weak_mean=float(weak.mean()), best_max=float(best.max()), best_mean=float(best.mean()), collide=float((best >= weak).mean()))
    del fsm; torch.cuda.empty_cache()
json.dump(OUT, open(results_path('sequence', 'seq_s5_arc_drift.json'), 'w'), indent=1)
