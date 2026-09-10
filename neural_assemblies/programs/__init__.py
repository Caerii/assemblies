"""
Literature-aligned programs on top of Assembly Calculus.

Layer 3 in the runtime stack: parsers, planners, FSM/TM demos, statistical
learning protocols, and causal binding — built on ``Brain`` + ``FiberCircuit``.
"""

from .rule_parser import RuleParser, parse_sentence
from .learn import learn_class_assembly, learn_separable_classes
from .fsm_learn import learn_fsm_from_sequences, build_fsm_from_traces
from .tm_demo import MinimalTMDemo
from .markov_coin import (
    CoinFlipModel,
    MarkovChainModel,
    train_markov_from_sequences,
)
from .colt_mnist_numpy import ColtMnistResult, run_colt_mnist_numpy
from .colt_mnist_brain import ColtMnistBrainResult, run_colt_mnist_brain
from .mod3_fsm import Mod3FsmResult, run_mod3_fsm_demo
from .nemo_fsm import AlternatingMarkovNetwork, NemoArcFSM, NemoMarkovPFA
from .arc_markov import ArcMarkovNetwork, ArcMarkovProtocol
from .direct import direct_bind, measure_directional_asymmetry, validate_direct_do_calculus
from .planning import (
    BlocksWorldAC,
    BlocksWorldPlanner,
    apply_strips_plan,
    bfs_plan,
    make_toy_problem,
    make_three_block_problem,
    make_four_block_problem,
)

__all__ = [
    "RuleParser",
    "parse_sentence",
    "learn_class_assembly",
    "learn_separable_classes",
    "learn_fsm_from_sequences",
    "build_fsm_from_traces",
    "MinimalTMDemo",
    "CoinFlipModel",
    "MarkovChainModel",
    "train_markov_from_sequences",
    "ColtMnistResult",
    "run_colt_mnist_numpy",
    "ColtMnistBrainResult",
    "run_colt_mnist_brain",
    "Mod3FsmResult",
    "run_mod3_fsm_demo",
    "NemoArcFSM",
    "ArcMarkovNetwork",
    "ArcMarkovProtocol",
    "NemoMarkovPFA",
    "AlternatingMarkovNetwork",
    "direct_bind",
    "measure_directional_asymmetry",
    "validate_direct_do_calculus",
    "BlocksWorldAC",
    "BlocksWorldPlanner",
    "apply_strips_plan",
    "bfs_plan",
    "make_toy_problem",
    "make_three_block_problem",
    "make_four_block_problem",
]
