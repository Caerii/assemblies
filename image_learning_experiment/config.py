# Configuration for the CIFAR-10 Brain Simulation Experiment

import numpy as np

# Constants
LOW_LEVEL = "LOW_LEVEL"
MID_LEVEL = "MID_LEVEL"
HIGH_LEVEL = "HIGH_LEVEL"
CLASS_AREA = "CLASS_AREA"

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
CLASS_INDICES = {
    "airplane": 0,
    "automobile": 1,
    "bird": 2,
    "cat": 3,
    "deer": 4,
    "dog": 5,
    "frog": 6,
    "horse": 7,
    "ship": 8,
    "truck": 9
}

MNIST_INDICES = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9
}

# Data Loading & Preprocessing
DATA_ROOT = "./data" # Relative path from the workspace root
BATCH_SIZE = 64*2
AUGMENT_DATA = True
SHUFFLE_DATA = True

# --- Simulation Parameters ---
NUM_TRAIN_SAMPLES = 5000  # Number of training images to use
NUM_TEST_SAMPLES = 1000   # Number of testing images to use
SEED = 42                # Random seed for reproducibility

# --- Data Parameters ---
CIFAR10_PATH = 'data/cifar-10-batches-py'
# PREPROCESS_TRAIN_TARGET_SIZE = 3072 # Original flattened size (32*32*3)
# PREPROCESS_TEST_TARGET_SIZE = 3072
RF_SIZE = 4
RF_STRIDE = 2
_NUM_RF_PER_DIM = ((32 - RF_SIZE) // RF_STRIDE) + 1 # Private calculation variable
NUM_RECEPTIVE_FIELDS = _NUM_RF_PER_DIM * _NUM_RF_PER_DIM

PREPROCESS_TRAIN_TARGET_SIZE = NUM_RECEPTIVE_FIELDS
PREPROCESS_TEST_TARGET_SIZE = NUM_RECEPTIVE_FIELDS
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Brain Structure & Initialization
BRAIN_P = 0.01 # Initial connection probability
BRAIN_SEED = 0 # RNG Seed for brain initialization

# --- Brain Area Configuration ---
# n: number of neurons in the area
# k: number of winners in k-WTA
# p: initial connection probability (if generating connections)
# inhibited: initial inhibition state
# plastic: whether incoming synapses are subject to Hebbian learning
# explicit: whether neurons have fixed representations (input/output) or emerge
AREAS = {
    'LOW_LEVEL':  {'n': NUM_RECEPTIVE_FIELDS, 'k': int(NUM_RECEPTIVE_FIELDS * 0.15), 'inhibited': False, 'plastic': False, 'explicit': True}, # k adjusted proportionally (15%)
    'MID_LEVEL':  {'n': 512, 'k': 50, 'inhibited': False, 'plastic': True, 'explicit': False},
    'HIGH_LEVEL': {'n': 256, 'k': 25, 'inhibited': False, 'plastic': True, 'explicit': False},
    'CLASS_AREA': {'n': len(CLASS_NAMES),  'k': 1,  'inhibited': False, 'plastic': True, 'explicit': True}, # One winner per class, n matches num classes
}

# Learning Process
LEARNING_RATE_BETA = 0.001 # Kept low from decay experiments
ENABLE_DECAY = True
SYNAPTIC_DECAY_RATE = 0.0005 # Explicitly set based on last experiment

# Evaluation
NUM_EVAL_SAMPLES = 100

# Animation
ANIMATION_OUTPUT_DIR = "animations" # Relative path from the workspace root
ANIMATION_FPS = 10 
ANIMATION_SNAPSHOT_RATE = 50 # Save a snapshot every N training samples 

# Options for reinforcement modifications
ASSOCIATIVE_REINFORCEMENT = True # Strengthen connections between actual HIGH winners and true CLASS label
ENABLE_LTD = True               # Weaken connections from HIGH winners to incorrect CLASS assemblies
LTD_FACTOR = 0.5                # Factor for LTD weakening (relative to beta)
ENABLE_NORMALIZATION = True     # Apply L2 normalization after LTP/LTD updates

# WEIGHT_CLIP_MAX = 5.0 # Max weight if clipping is enabled (Currently disabled) 

# --- Animator Parameters ---
SAVE_ANIMATIONS = True
ANIMATION_INTERVAL = 1  # Interval between frames in milliseconds
ANIMATION_FRAMES_PER_EPOCH = 100 # Number of frames (training steps) per animation video
ANIMATION_FPS = 10 # Frames per second for the output video
ANIMATION_SNAPSHOT_RATE = 50 # Save a snapshot every N training samples (e.g., 50 means save frame 0, 50, 100...) 
                                # Must be a divisor of NUM_TRAIN_SAMPLES if you want the last frame
OUTPUT_DIR = "animations/experiments"
EXPERIMENT_NAME = "experiment_rf_1" # Subdirectory for this experiment's outputs
PLOT_CONNECTOME = True # Whether to generate connectome heatmaps at the end 