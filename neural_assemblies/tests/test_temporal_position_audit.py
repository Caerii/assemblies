"""A pooled score can pass without any distractor-specific representation."""
import pytest
from research.experiments import seq_a3_transducer as study


def test_agreement_word_signal_can_fake_a_distractor_contrast():
    labels=['sg','sg','pl','pl']
    common=set(range(6))
    sg,pl=set(range(6)),set(range(6,12))
    positions={0:[(label,common) for label in labels],
               1:[(label,common) for label in labels],
               2:[(label,sg if label=='sg' else pl) for label in labels]}
    distractor=study._historical_pooled_arc_overlaps({0:positions[0],1:positions[1]})
    pooled=study._historical_pooled_arc_overlaps(positions)
    assert distractor['same']-distractor['diff']==0.
    assert pooled['same']-pooled['diff']==.5


def test_old_distractor_helper_refuses_misleading_label():
    with pytest.raises(ValueError,match='position-specific'):
        study._distractor_overlaps({})


def test_mechanism_request_stops_before_gpu_construction():
    with pytest.raises(ValueError,match='TM-9 mechanism collection is invalid'):
        study.a3_hashed([1,2,3],n_arc=60,beta=.1,collect_arcs=True)


def test_chain_processed_positions_include_informative_non_distractors():
    import ntp_agree
    old_chain,old_gap=ntp_agree.CHAIN,ntp_agree.GAP
    try:
        ntp_agree.use_chain(True,gap=2)
        sentence=ntp_agree.generate_chain(1,123)[0]
        roles=[ntp_agree.CLASS[word].split('_')[0] for word in sentence[:-1]]
        assert roles[1:3]==['NOUN','NOUN']
        assert roles[3]=='VERB' and roles[6]=='PRON'
        # The historical helper includes positions3 and6 because both are nonzero.
    finally:
        ntp_agree.use_chain(old_chain,gap=old_gap)
