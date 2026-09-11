"""Position-specific observations for the registered agreement-chain corpus.

Specification: research/notes/sequence/AUDIT_temporal_position_pooling.md#replacement-observation-contract
"""
from itertools import combinations
from hashlib import sha256
from math import fsum

from neural_assemblies.core.registration import validate_area_registration, validate_round_count
from research.experiments.study4.ntp_agree import CHAIN_CLASSES


TOKEN_CLASSES = {word: name.split('_') for name, words in CHAIN_CLASSES.items() for word in words}
FRAME_FIELDS = {'sentence_id', 'position', 'token', 'subject_number', 'neurons'}


def _learned_state_digest(transducer):
    """Hash count matrices, stimulus potentiations and refraction charges.

    This checks endpoint equality of learned tensors, not hermetic execution or
    transient writes later undone. Winner state and stimulus cursors must advance.
    Transfer one tensor at a time; do not duplicate the whole brain on the host.
    """
    digest = sha256()
    tensors = []
    for name in ('lex_arc', 'state_arc', 'arc_state', 'arc_out', 'reg_arc'):
        fiber = getattr(transducer, name)
        if fiber is not None:
            fiber.check()
            tensors.append((name, fiber.counts()))
    for name in ('lex', 'arc', 'state', 'out', 'reg'):
        area = getattr(transducer, name)
        if area is not None:
            tensors.append((name, area.bias))
    stimuli = [('S', transducer.S), ('G', transducer.G)]
    stimuli += [(f'Gs.{i}', item) for i, item in enumerate(transducer.Gs)]
    if transducer.reg is not None:
        stimuli.append(('F', transducer.F))
    tensors += [(name, stimulus.pot) for name, stimulus in stimuli]
    for name, tensor in tensors:
        digest.update(name.encode('utf-8') + b'\0')
        if tensor is None:
            digest.update(b'none\0')
        else:
            digest.update(f'{tensor.dtype}:{tuple(tensor.shape)}\0'.encode('ascii'))
            host = tensor.detach().cpu().contiguous().numpy()
            digest.update(memoryview(host).cast('B'))
            del host
    return digest.hexdigest()


def capture_chain_arcs(transducer, corpora, *, gap, rounds, state_blind=False):
    """Capture complete, position-labelled arcs under frozen transducer readout.

    Specification: research/notes/sequence/AUDIT_temporal_position_pooling.md#replacement-observation-contract

    One corpus per brain, in seed order. Reset at each sentence boundary, then
    tick and emit (which advances carry). No teacher forcing. Unequal corpus
    lengths use idle rows; idle winners are not observations. This leaves runtime
    winner state changed and checks learned tensor equality before returning.
    """
    import torch

    rounds = validate_round_count(rounds)
    if type(state_blind) is not bool:
        raise TypeError('state_blind must be a bool')
    n, k = validate_area_registration('ARC', transducer.n_arc, transducer.k)
    seeds = list(transducer.seeds)
    if (len(corpora) != transducer.B or len(seeds) != transducer.B or not seeds
            or any(type(seed) is not int or seed < 0 for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError('capture needs one corpus per unique nonnegative brain seed')
    schedules = []
    for sentences in corpora:
        manifest = chain_observation_manifest(sentences, gap=gap)
        labels = [manifest[(i, 0)]['subject_number'] for i in range(len(sentences))]
        if len(set(labels)) < 2 or len(labels) == len(set(labels)):
            raise ValueError('capture needs same- and different-subject sentence pairs')
        if any(word not in transducer.word_index for sentence in sentences for word in sentence):
            raise ValueError('capture corpus contains tokens outside the transducer vocabulary')
        schedules.append(list(manifest.values()))
    before = _learned_state_digest(transducer)
    frames = [[] for _ in seeds]
    with torch.no_grad():
        for step in range(max(map(len, schedules))):
            active = [schedule[step] if step < len(schedule) else None for schedule in schedules]
            boundary = torch.tensor([item is not None and item['position'] == 0 for item in active],
                                    dtype=torch.bool, device=transducer.device)
            if bool(boundary.any()):
                transducer.reset(boundary)
            words = [transducer.word_index[item['token']] if item is not None else -1 for item in active]
            live = torch.tensor([item is not None for item in active], dtype=torch.bool,
                                device=transducer.device)
            if state_blind:
                transducer.state.inhibit_rows(live)
            transducer.tick(words, rounds=rounds, freeze=True)
            # Snapshot before emit so carry advancement cannot rewrite this frame.
            winners = transducer.arc.winners.detach().cpu().tolist()
            transducer.emit()
            for brain, item in enumerate(active):
                if item is not None:
                    frame = {key: value for key, value in item.items() if key != 'role'}
                    frame['neurons'] = winners[brain]
                    frames[brain].append(frame)
    if _learned_state_digest(transducer) != before:
        raise RuntimeError('frozen temporal capture changed learned tensors; discard this run')
    return [{'seed': seed, 'frames': observed,
             'analysis': chain_arc_contrasts(sentences, observed, gap=gap, n=n, k=k),
             'learned_state_digest': before}
            for seed, sentences, observed in zip(seeds, corpora, frames)]


def chain_observation_manifest(sentences, *, gap):
    """Expected processed tokens, keyed by explicit sentence and position identity."""
    gap = validate_round_count(gap)
    roles = ['AUX'] + ['NOUN'] * gap + ['VERB'] + ['NOUN'] * gap + ['PRON'] + ['NOUN'] * gap + ['TAG']
    if not sentences:
        raise ValueError('observation corpus must contain sentences')
    manifest = {}
    for sentence_id, sentence in enumerate(sentences):
        if len(sentence) != len(roles):
            raise ValueError('sentence does not match the declared chain gap')
        classes = []
        for token in sentence:
            if not isinstance(token, str) or token not in TOKEN_CLASSES:
                raise ValueError('unknown chain token')
            classes.append(TOKEN_CLASSES[token])
        if [role for role, number in classes] != roles:
            raise ValueError('token roles do not match the chain template')
        subject = classes[0][1]
        if any(number != subject for role, number in classes if role != 'NOUN'):
            raise ValueError('agreement tokens must match the subject number')
        # Last token is the prediction target, not an observed input step.
        for position, token in enumerate(sentence[:-1]):
            manifest[(sentence_id, position)] = {
                'sentence_id': sentence_id, 'position': position, 'token': token,
                'subject_number': subject, 'role': roles[position]}
    return manifest


def chain_arc_contrasts(sentences, frames, *, gap, n, k):
    """One brain's per-position pair means; pairs are not independent brain seeds.

    Require the complete expected frame inventory and exactly k distinct neuron IDs.
    Token roles and subject labels are derived from the corpus, not trusted metadata.
    """
    n, k = validate_area_registration('ARC', n, k)
    manifest = chain_observation_manifest(sentences, gap=gap)
    observed = {}
    for frame in frames:
        if not isinstance(frame, dict) or set(frame) != FRAME_FIELDS:
            raise ValueError('arc frame fields must match the observation contract')
        if any(type(frame[key]) is not int for key in ('sentence_id', 'position')):
            raise ValueError('frame identities must be integer sentence/position keys')
        key = (frame['sentence_id'], frame['position'])
        if key not in manifest or key in observed:
            raise ValueError('unexpected or duplicate arc frame identity')
        expected = manifest[key]
        if any(frame[name] != expected[name] for name in ('token', 'subject_number')):
            raise ValueError('arc frame token/subject differs from the corpus manifest')
        neurons = frame['neurons']
        if (not isinstance(neurons, (list, tuple)) or len(neurons) != k
                or any(type(value) is not int or not 0 <= value < n for value in neurons)
                or len(set(neurons)) != k):
            raise ValueError('arc frames need exactly k distinct in-range neuron IDs')
        observed[key] = frozenset(neurons)
    if set(observed) != set(manifest):
        raise ValueError('missing arc frames; incomplete sentences must not be dropped')
    positions = []
    for position in sorted({position for _, position in manifest}):
        same, different = [], []
        for left, right in combinations(range(len(sentences)), 2):
            a, b = (left, position), (right, position)
            overlap = len(observed[a] & observed[b]) / k
            target = same if manifest[a]['subject_number'] == manifest[b]['subject_number'] else different
            target.append(overlap)
        if not same or not different:
            raise ValueError('each position needs both same- and different-subject sentence pairs')
        # Stable arithmetic for within-brain pair means; never a seed-level interval.
        same_mean, different_mean = fsum(same) / len(same), fsum(different) / len(different)
        role = manifest[(0, position)]['role']
        positions.append({'position': position, 'role': role, 'is_distractor': role == 'NOUN',
                          'same_pair_count': len(same), 'different_pair_count': len(different),
                          'same_mean': same_mean, 'different_mean': different_mean,
                          'contrast': same_mean - different_mean})
    return {'positions': positions, 'sentence_count': len(sentences), 'frame_count': len(observed)}
