"""The corrected measurement cannot turn agreement-word signal into distractor carry."""
from copy import deepcopy

import pytest
from research.experiments.temporal_observations import chain_observation_manifest, chain_arc_contrasts

SENTENCES = [
    ['does','dog','sees','cat','it','bird','doesnt'],
    ['is','dogs','likes','cats','itself','birds','isnt'],
    ['do','dog','see','cat','they','bird','dont'],
    ['are','dogs','like','cats','themselves','birds','arent'],
]


def fake_transducer(mutation=None):
    """A clock that refuses a tick unless the preceding emit advanced carry."""
    from types import SimpleNamespace
    torch = pytest.importorskip('torch')

    class Clock:
        B, seeds, n_arc, k, device = 2, [11, 17], 64, 3, 'cpu'

        def __init__(self):
            self.word_index = {word: i for i, word in enumerate(sorted(set(sum(SENTENCES, []))))}
            self.steps = [0, 0]
            self.pending = False
            self.calls = []
            for name in ('lex', 'arc', 'state', 'out'):
                setattr(self, name, SimpleNamespace(bias=torch.zeros(2, 64)))
            for name in ('lex_arc', 'state_arc', 'arc_state', 'arc_out'):
                count = torch.zeros(2, 4, 4, dtype=torch.int8)
                setattr(self, name, SimpleNamespace(counts=lambda count=count: count, check=lambda: None))
            self.reg = self.reg_arc = None
            self.S = SimpleNamespace(pot=torch.zeros(2, 64, dtype=torch.int64))
            self.G = SimpleNamespace(pot=None)
            self.Gs = []

        def reset(self, mask):
            for brain, reset in enumerate(mask.tolist()):
                if reset:
                    self.steps[brain] = 0

        def tick(self, words, *, rounds, freeze):
            assert not self.pending and freeze and rounds == 2
            self.pending = True
            self.calls.append(words)
            self.arc.winners = torch.tensor([[3 * step + j for j in range(3)] for step in self.steps])
            if mutation == 'counts':
                self.arc_out.counts()[0, 0, 0] += 1
            elif mutation == 'bias':
                self.arc.bias[0, 0] += 1
            elif mutation == 'stimulus':
                self.S.pot[0, 0] += 1

        def emit(self):
            assert self.pending
            self.pending = False
            self.steps = [step + 1 for step in self.steps]

    return Clock()


def test_capture_preserves_ragged_corpora_boundaries_and_advances_carry():
    from research.experiments.temporal_observations import capture_chain_arcs
    clock = fake_transducer()
    corpora = [SENTENCES, SENTENCES[:3]]
    reports = capture_chain_arcs(clock, corpora, gap=1, rounds=2)
    assert [report['seed'] for report in reports] == [11, 17]
    assert [len(report['frames']) for report in reports] == [24, 18]
    assert all(words[1] == -1 for words in clock.calls[18:])
    for report in reports:
        for frame in report['frames']:
            assert frame['neurons'] == [3 * frame['position'] + j for j in range(3)]
    assert not clock.pending


@pytest.mark.parametrize('mutation', ['counts', 'bias', 'stimulus'])
def test_capture_rejects_learned_state_mutation(mutation):
    from research.experiments.temporal_observations import capture_chain_arcs
    with pytest.raises(RuntimeError, match='changed learned tensors'):
        capture_chain_arcs(fake_transducer(mutation), [SENTENCES, SENTENCES], gap=1, rounds=2)


@pytest.mark.parametrize('invalid', ['seeds', 'batch', 'vocabulary', 'corpus', 'pairs'])
def test_capture_preflight_rejects_before_any_tick(invalid):
    from research.experiments.temporal_observations import capture_chain_arcs
    clock = fake_transducer()
    corpora = [SENTENCES, SENTENCES]
    if invalid == 'seeds':
        clock.seeds = [11, 11]
    elif invalid == 'batch':
        corpora.pop()
    elif invalid == 'vocabulary':
        clock.word_index.pop('dog')
    elif invalid == 'corpus':
        corpora[1] = [['unknown']]
    else:
        corpora[1] = SENTENCES[:2]
    with pytest.raises(ValueError):
        capture_chain_arcs(clock, corpora, gap=1, rounds=2)
    assert clock.calls == []


@pytest.mark.gpu
@pytest.mark.parametrize('state_mode', ['copy', 'induced'])
def test_cuda_capture_matches_manual_frozen_sentence_schedule(state_mode):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA required for real transducer capture')
    from neural_assemblies.core.torch_engine import _fused_cuda
    if _fused_cuda.load() is None:
        pytest.fail(f'CUDA present but fused extension unavailable: {_fused_cuda.last_error()}')
    from neural_assemblies.core.torch_engine._hashed_transducer import HashedTransducer
    from research.experiments.temporal_observations import capture_chain_arcs
    vocab = sorted(set(sum(SENTENCES, [])))
    transducer = HashedTransducer([11, 17], vocab, n=64, k=3, p=.2, organ_p=.3,
                                 state_mode=state_mode, max_potentiations=256)
    transducer.ground(rounds=2)
    transducer.train_sentence(SENTENCES[0], rounds=2)
    corpora = [SENTENCES, SENTENCES[:3]]
    reports = capture_chain_arcs(transducer, corpora, gap=1, rounds=2)
    # Independent, sentence-at-a-time schedule checks frame identity and values.
    manual = [[], []]
    for sentence_id in range(4):
        transducer.reset()
        for position in range(6):
            words = [transducer.word_index[corpus[sentence_id][position]]
                     if sentence_id < len(corpus) else -1 for corpus in corpora]
            transducer.tick(words, rounds=2, freeze=True)
            transducer.emit()
            arcs = transducer.arc.winners.cpu().tolist()
            for brain, corpus in enumerate(corpora):
                if sentence_id < len(corpus):
                    manual[brain].append(arcs[brain])
    assert [[frame['neurons'] for frame in report['frames']] for report in reports] == manual


def frames(carry=False):
    result=[]
    for expected in chain_observation_manifest(SENTENCES,gap=1).values():
        frame={key:value for key,value in expected.items() if key!='role'}
        group = 3 if expected['subject_number']=='pl' and (carry or expected['role']!='NOUN') else 0
        frame['neurons']=list(range(group,group+3))
        result.append(frame)
    return result


def measure(observations):
    return chain_arc_contrasts(SENTENCES,observations,gap=1,n=6,k=3)


def test_agreement_signal_is_separate_from_absent_distractor_signal():
    report=measure(frames())
    assert report['frame_count']==24 and report['sentence_count']==4
    assert [row['contrast'] for row in report['positions']]==[1.,0.,1.,0.,1.,0.]
    assert [row['position'] for row in report['positions'] if row['is_distractor']]==[1,3,5]
    assert all(row['same_pair_count']==2 and row['different_pair_count']==4 for row in report['positions'])


def test_real_distractor_signal_moves_each_position_and_frame_order_is_irrelevant():
    observed=frames(carry=True)
    expected=measure(observed)
    assert all(row['contrast']==1. for row in expected['positions'])
    assert measure(list(reversed(observed)))==expected


@pytest.mark.parametrize('change',['missing','duplicate','sentence','position','token','subject','neurons','range','bool','extra'])
def test_corrupt_or_incomplete_frames_fail_instead_of_silent_pooling(change):
    observed=frames()
    if change=='missing':
        observed.pop()
    elif change=='duplicate':
        observed.append(deepcopy(observed[0]))
    elif change=='sentence':
        observed[0]['sentence_id']=99
    elif change=='position':
        observed[0]['position']=True
    elif change=='token':
        observed[0]['token']='do'
    elif change=='subject':
        observed[0]['subject_number']='pl'
    elif change=='neurons':
        observed[0]['neurons']=[0,0,1]
    elif change=='range':
        observed[0]['neurons']=[0,1,6]
    elif change=='bool':
        observed[0]['neurons']=[False,1,2]
    else:
        observed[0]['is_distractor']=True
    with pytest.raises(ValueError):
        measure(observed)


@pytest.mark.parametrize('change',['gap','role','agreement','unknown','empty'])
def test_invalid_corpus_cannot_define_a_mislabelled_manifest(change):
    corpus=deepcopy(SENTENCES)
    gap=1
    if change=='gap': gap=2
    elif change=='role': corpus[0][1]='sees'
    elif change=='agreement': corpus[0][2]='see'
    elif change=='unknown': corpus[0][1]='unknown'
    else: corpus=[]
    with pytest.raises(ValueError):
        chain_observation_manifest(corpus,gap=gap)


def test_single_subject_population_has_no_contrast_and_is_not_dropped():
    corpus=SENTENCES[:2]
    observed=[row for row in frames() if row['sentence_id']<2]
    with pytest.raises(ValueError,match='same- and different-subject'):
        chain_arc_contrasts(corpus,observed,gap=1,n=6,k=3)


@pytest.mark.parametrize('gap',[1,2,3,6])
def test_manifest_matches_real_generated_corpus_without_global_mode_changes(gap,monkeypatch):
    from research.experiments.study4 import ntp_agree
    monkeypatch.setattr(ntp_agree,'GAP',gap)
    corpus=ntp_agree.generate_chain(12,123)
    manifest=chain_observation_manifest(corpus,gap=gap)
    assert len(manifest)==12*(3+3*gap)
    assert sum(row['role']=='NOUN' for row in manifest.values())==12*3*gap
    observed=[]
    for row in manifest.values():
        frame={key:value for key,value in row.items() if key!='role'}
        frame['neurons']=[0,1,2]
        observed.append(frame)
    report=chain_arc_contrasts(corpus,observed,gap=gap,n=6,k=3)
    assert all(row['contrast']==0 for row in report['positions'])
    assert sum(row['is_distractor'] for row in report['positions'])==3*gap
