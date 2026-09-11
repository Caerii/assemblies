"""Position-specific observations for the registered agreement-chain corpus.

Specification: research/notes/sequence/AUDIT_temporal_position_pooling.md#replacement-observation-contract
"""
from itertools import combinations
from math import fsum

from neural_assemblies.core.registration import validate_area_registration, validate_round_count
from research.experiments.study4.ntp_agree import CHAIN_CLASSES


TOKEN_CLASSES = {word: name.split('_') for name, words in CHAIN_CLASSES.items() for word in words}
FRAME_FIELDS = {'sentence_id', 'position', 'token', 'subject_number', 'neurons'}


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
