import numpy as np

# Assuming config.py is in the parent directory
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
import config

# Import the base Brain class from the existing brain_module
from brain_module.core import Brain
from brain_module.area import Area

class CIFAR10Brain:
    def __init__(self, areas_config, p_connect, seed=None):
        """
        Initializes the Brain specific to the CIFAR-10 task configuration.

        Args:
            areas_config (dict): Configuration dictionary for brain areas from config.py.
            p_connect (float): Initial connection probability between areas.
            seed (int, optional): Random seed for reproducibility.
        """
        self.brain = Brain(areas_config, p_connect, seed=seed)
        self.class_names = config.CLASS_NAMES
        self._validate_config()
        self._initialize_areas()
        self.label_assemblies = {}  # Mapping from labels to neuron assemblies in CLASS_AREA area
        self._initialize_assemblies()
        self.activation_fig = None # Added for update_activations if used externally
        self.activation_ax = None  # Added for update_activations if used externally

    def _validate_config(self):
        """Basic validation of critical config parameters."""
        if 'CLASS_AREA' not in self.brain.areas:
            raise ValueError("Configuration must include a 'CLASS_AREA'")
        if self.brain.areas['CLASS_AREA'].n != len(self.class_names):
            raise ValueError(f"CLASS_AREA size ({self.brain.areas['CLASS_AREA'].n}) "
                             f"must match number of classes ({len(self.class_names)})")
        if 'LOW_LEVEL' not in self.brain.areas:
            raise ValueError("Configuration must include a 'LOW_LEVEL' area for input")
        if self.brain.areas['LOW_LEVEL'].n != config.PREPROCESS_TRAIN_TARGET_SIZE:
             raise ValueError(f"LOW_LEVEL area size ({self.brain.areas['LOW_LEVEL'].n}) must match "
                              f"PREPROCESS_TRAIN_TARGET_SIZE ({config.PREPROCESS_TRAIN_TARGET_SIZE})")

    def _initialize_assemblies(self):
        """Initializes label assemblies for all classes."""
        class_area = self.brain.areas['CLASS_AREA']
        # Use config for CLASS_INDICES
        for class_name, index in config.CLASS_INDICES.items():
            start = index * class_area.k
            assembly = np.arange(start, start + class_area.k)
            self.label_assemblies[class_name] = assembly

        # Validate the keys in label_assemblies
        assert set(self.label_assemblies.keys()) == set(config.CLASS_INDICES.keys()), \
            "[DEBUG] Mismatch in label assemblies initialization!"

        print("[DEBUG] Initialized assemblies for all classes.")
        print(f"[DEBUG] Class assemblies: {self.label_assemblies}")

    def _initialize_areas(self):
        """Initialize brain areas for image processing using AREA_CONFIG."""
        for area_name, area_params in config.AREA_CONFIG.items():
            self.brain.add_area(
                area_name,
                n=area_params['n'],
                k=area_params['k'],
                beta=area_params['beta'],
                explicit=area_params['explicit']
            )

        # Print initial connectome statistics
        for src_area in self.brain.connectomes:
            for dst_area in self.brain.connectomes[src_area]:
                connectome = self.brain.connectomes[src_area][dst_area]
                mean_weight = np.mean(connectome)
                max_weight = np.max(connectome)
                min_weight = np.min(connectome)
                print(f"[DEBUG] Initial connectome {src_area} -> {dst_area}: mean={mean_weight}, max={max_weight}, min={min_weight}")

    def update_activations(self, class_area, label_assemblies):
        """Update the activation bar chart in real-time."""
        # This method might need matplotlib if called externally
        # import matplotlib.pyplot as plt # Add if needed here
        if self.activation_fig is None or self.activation_ax is None:
            # Initialization might need to happen elsewhere or be passed in
            # self.initialize_activation_plot() # Requires matplotlib figure/axes
            print("Warning: Activation plot not initialized.")
            return

        activations = {
            label: np.intersect1d(class_area.winners, assembly).size
            for label, assembly in label_assemblies.items()
        }
        self.activation_ax.clear()
        self.activation_ax.bar(activations.keys(), activations.values())
        self.activation_ax.set_title("Assembly Activations in CLASS_AREA")
        self.activation_ax.set_xlabel("Class")
        self.activation_ax.set_ylabel("Activations")
        self.activation_ax.set_xticks(range(len(label_assemblies)))
        self.activation_ax.set_xticklabels(label_assemblies.keys(), rotation=45)
        self.activation_fig.canvas.draw()
        self.activation_fig.canvas.flush_events()

    def parse_receptive_fields(self, rf_vector):
        """
        Processes a precomputed receptive field activation vector.
        Directly sets LOW_LEVEL activation without internal k-WTA,
        then projects activity through the hierarchy.
        """
        low_level_area = self.brain.areas['LOW_LEVEL']
        if rf_vector.shape[0] != low_level_area.n:
            raise ValueError(f"Input receptive field vector size ({rf_vector.shape[0]}) "
                             f"does not match LOW_LEVEL area size ({low_level_area.n})")

        # Directly set LOW_LEVEL activation from the receptive field vector
        # No k-WTA applied here; the pattern represents the sensory input directly.
        low_level_area.a = rf_vector.copy()

        # For projection purposes, we need to know which neurons *could* potentially fire.
        # We set winners based on non-zero activation, as k-WTA is skipped.
        # The actual strength is carried by the activation value `a`.
        low_level_area.winners = np.where(low_level_area.a > 1e-6)[0] # Use a small threshold to avoid floating point issues
        low_level_area.w = low_level_area.winners # Update winner history for projection source tracking

        # Project activity through the hierarchy
        self.brain.project('LOW_LEVEL', 'MID_LEVEL') # MID_LEVEL applies k-WTA based on input
        self.brain.project('MID_LEVEL', 'HIGH_LEVEL')
        self.brain.project('HIGH_LEVEL', 'CLASS_AREA')

        # Return the final activations in the CLASS_AREA
        return self.brain.areas['CLASS_AREA'].a

    def set_image_input(self, image_features):
        # Find the correct area using self.brain.areas
        low_level_area = self.brain.areas['LOW_LEVEL']
        k = low_level_area.k

        # Ensure image_features is numpy array for np functions
        if not isinstance(image_features, np.ndarray):
            image_features = np.array(image_features)

        # Perform calculations
        ptp_val = np.ptp(image_features)
        min_val = np.min(image_features)
        norm = np.linalg.norm(image_features)

        normalized_features = (image_features - min_val) / (ptp_val + 1e-6)
        normalized_features = normalized_features / (norm + 1e-6)

        top_indices = np.argsort(-normalized_features)[:k]

        low_level_area.winners = top_indices
        low_level_area.w = k

        print(f"[DEBUG] Activated {config.LOW_LEVEL} with top {k} neurons: {top_indices[:10]}...")

        # Project after setting input (assuming this is the intended logic)
        self.brain.project({}, {config.LOW_LEVEL: [config.MID_LEVEL]})

    def _activate_area_from_projection(self, area_name):
        area = self.brain.areas[area_name]
        input_sums = np.zeros(area.n)

        for src_area_name in area.beta_by_area:
            if src_area_name not in self.brain.areas:
                print(f"Warning: Source area {src_area_name} not found during projection to {area_name}")
                continue
            src_area = self.brain.areas[src_area_name]
            if src_area.winners.size == 0:
                continue

            # Check connectome existence
            if src_area_name not in self.brain.connectomes or area_name not in self.brain.connectomes[src_area_name]:
                 print(f"Warning: Connectome from {src_area_name} to {area_name} not found.")
                 continue

            connectome = self.brain.connectomes[src_area_name][area_name]
            pre_neurons = src_area.winners
            # Ensure indices are within bounds
            if np.any(pre_neurons >= connectome.shape[0]):
                print(f"Warning: pre_neurons indices out of bounds for connectome {src_area_name}->{area_name}.")
                pre_neurons = pre_neurons[pre_neurons < connectome.shape[0]]
                if pre_neurons.size == 0:
                    continue

            input_sums += connectome[pre_neurons, :].sum(axis=0)

        print(f"[DEBUG] Input sums for {area_name}: mean={input_sums.mean()}, max={input_sums.max()}, min={input_sums.min()}")

        k = area.k
        # Ensure k is not larger than the number of neurons
        actual_k = min(k, area.n)
        if input_sums.size == 0:
             print(f"Warning: Input sums are empty for {area_name}. Cannot determine winners.")
             top_indices = np.array([], dtype=np.uint32)
        elif actual_k > 0:
            top_indices = np.argsort(-input_sums)[:actual_k]
        else:
            top_indices = np.array([], dtype=np.uint32)

        area.winners = top_indices
        area.w = area.winners.size
        print(f"[DEBUG] Activated {area_name} with top {area.w} neurons from projection: {top_indices[:10]}...")

    def _activate_area(self, area_name, features):
        area = self.brain.areas[area_name]
        if features is None:
             print(f"Warning: Cannot activate {area_name}: Feature vector is None.")
             area.winners = np.array([], dtype=np.uint32)
             area.w = 0
             return
        # Ensure features is a numpy array
        if not isinstance(features, np.ndarray):
            features = np.array(features)

        if features.size == 0:
            print(f"Warning: Cannot activate {area_name}: Feature vector is empty.")
            area.winners = np.array([], dtype=np.uint32)
            area.w = 0
            return

        # Check for NaN or Inf values
        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
             print(f"Warning: Features for {area_name} contain NaN or Inf. Setting winners to empty.")
             area.winners = np.array([], dtype=np.uint32)
             area.w = 0
             return

        ptp_val = np.ptp(features)
        min_val = np.min(features)

        # Avoid division by zero if features are constant
        if ptp_val > 1e-9:
             normalized_features = (features - min_val) / ptp_val
        else:
             normalized_features = np.zeros_like(features)

        k = area.k
        actual_k = min(k, area.n)

        if normalized_features.size == 0:
             print(f"Warning: Normalized features are empty for {area_name}.")
             top_indices = np.array([], dtype=np.uint32)
        elif actual_k > 0:
             # Ensure normalized_features is 1D for argsort
             if normalized_features.ndim > 1:
                  normalized_features = normalized_features.flatten()
             # Check if size matches area.n after flatten, handle mismatch if necessary
             if normalized_features.size != area.n:
                  print(f"Warning: Feature size ({normalized_features.size}) mismatch with area n ({area.n}) for {area_name}. Padding/truncating.")
                  # Example: Pad with zeros or truncate
                  if normalized_features.size < area.n:
                       padded_features = np.zeros(area.n)
                       padded_features[:normalized_features.size] = normalized_features
                       normalized_features = padded_features
                  else:
                       normalized_features = normalized_features[:area.n]

             top_indices = np.argsort(-normalized_features)[:actual_k]
        else:
             top_indices = np.array([], dtype=np.uint32)

        area.winners = top_indices
        area.w = area.winners.size
        print(f"[DEBUG] Activated {area_name} with top {len(top_indices)} neurons: {top_indices[:10]}...")

    def project_hierarchy(self, from_area, to_area):
        area = self.brain.areas[from_area]
        print(f"[DEBUG] {from_area} winners size: {area.winners.size}, w: {area.w}")

        if area.winners.size == 0 or area.w == 0:
            # Changed from ValueError to warning to allow processing continuation
            print(f"Warning: No active assemblies in {from_area}, cannot project to {to_area}.")
            # Ensure the target area also has empty winners if projection fails
            if to_area in self.brain.areas:
                 target_area = self.brain.areas[to_area]
                 target_area.winners = np.array([], dtype=np.uint32)
                 target_area.w = 0
            return # Stop the projection for this step

        print(f"Projecting from {from_area} to {to_area}.")
        print(f"[DEBUG] {from_area} has {area.w} winners before projection.")

        areas_by_stim = {}
        dst_areas_by_src_area = {from_area: [to_area]}
        self.brain.project(areas_by_stim, dst_areas_by_src_area)

    def predict_label(self):
        self.brain.disable_plasticity = True
        # Ensure projection happens correctly
        # self.brain.project({}, {config.HIGH_LEVEL: [config.CLASS_AREA]})
        # Instead of direct project, let's explicitly activate CLASS_AREA from HIGH_LEVEL winners
        self._activate_area_from_projection(config.CLASS_AREA)
        self.brain.disable_plasticity = False

        class_area = self.brain.areas[config.CLASS_AREA]
        object_winners = class_area.winners

        # Handle case where object_winners might be empty
        if object_winners is None or object_winners.size == 0:
             print("Warning: No winners in CLASS_AREA during prediction.")
             return None # Indicate prediction failure

        scores = {
            label: np.intersect1d(object_winners, assembly).size
            for label, assembly in self.label_assemblies.items()
        }
        print(f"[DEBUG] Prediction scores: {scores}")
        # Handle case where scores dict might be empty or all scores are 0
        if not scores or all(s == 0 for s in scores.values()):
             print("Warning: No overlapping activations found for prediction or all scores zero.")
             return None # Indicate prediction failure or ambiguity

        predicted_label_name = max(scores, key=scores.get)
        return predicted_label_name

    def print_connectome_stats(self, src_area_name, dst_area_name):
        # Check if connectome exists
        if src_area_name not in self.brain.connectomes or dst_area_name not in self.brain.connectomes[src_area_name]:
            print(f"[DEBUG] Connectome {src_area_name} -> {dst_area_name} does not exist.")
            return
        connectome = self.brain.connectomes[src_area_name][dst_area_name]
        if connectome.size == 0:
             print(f"[DEBUG] Connectome {src_area_name} -> {dst_area_name} is empty.")
             mean_weight, max_weight, min_weight = 0, 0, 0
        else:
             mean_weight = np.mean(connectome)
             max_weight = np.max(connectome)
             min_weight = np.min(connectome)
        print(f"[DEBUG] Connectome {src_area_name} -> {dst_area_name}: mean={mean_weight:.4f}, max={max_weight:.4f}, min={min_weight:.4f}, shape={connectome.shape}")

    def apply_decay(self, area_from_name, area_to_name, decay_rate):
        """Applies multiplicative weight decay to connections between two areas."""
        conn_key = f"{area_from_name}->{area_to_name}"
        if conn_key in self.brain.connectomes:
             # Ensure decay doesn't reverse sign or go below a small epsilon? For now, simple multiplication.
            self.brain.connectomes[conn_key] *= (1.0 - decay_rate)
            # Optional: Add a small value or clip at zero if needed
            # self.brain.W[conn_key] = np.maximum(0, self.brain.W[conn_key])
        # else: # Don't warn for every decay step if connections don't exist
            # print(f"Warning: Connection {conn_key} not found during decay.")

    def reinforce(self, true_label_index):
        """
        Applies Hebbian reinforcement based on the true label.
        Includes optional LTD and Normalization based on config.
        """
        if not (0 <= true_label_index < len(self.class_names)):
            raise ValueError(f"Invalid true_label_index: {true_label_index}")

        high_level_area = self.brain.areas['HIGH_LEVEL']
        class_area = self.brain.areas['CLASS_AREA']
        beta = config.LEARNING_RATE_BETA

        # Identify winners in the HIGH_LEVEL area (presynaptic neurons)
        high_winners = high_level_area.winners
        if high_winners is None or high_winners.size == 0:
            # print("Warning: No winners in HIGH_LEVEL during reinforcement.")
            return # Cannot reinforce if the previous layer didn't activate

        # --- Associative Reinforcement / LTD / Normalization --- 
        if config.ASSOCIATIVE_REINFORCEMENT:
            # Postsynaptic neurons: Correct neuron in CLASS_AREA
            true_assembly_neuron = true_label_index # neuron index = class index

            # LTP: Strengthen connections from HIGH winners to the correct CLASS neuron
            conn_key = 'HIGH_LEVEL->CLASS_AREA'
            if conn_key in self.brain.connectomes:
                self.brain.connectomes[conn_key][np.ix_(high_winners, [true_assembly_neuron])] *= (1 + beta)
                updated_neurons_postsynaptic = {true_assembly_neuron}

                if config.ENABLE_LTD:
                    # Identify incorrectly activated CLASS neurons (if any)
                    class_winners = class_area.winners
                    if class_winners is not None and class_winners.size > 0:
                        incorrect_winners = np.setdiff1d(class_winners, [true_assembly_neuron])
                        if incorrect_winners.size > 0:
                            ltd_beta = beta * config.LTD_FACTOR
                            # LTD: Weaken connections from HIGH winners to incorrect CLASS neurons
                            self.brain.connectomes[conn_key][np.ix_(high_winners, incorrect_winners)] *= (1 - ltd_beta)
                            updated_neurons_postsynaptic.update(incorrect_winners.tolist())

                if config.ENABLE_NORMALIZATION:
                    # L2 Normalize incoming weights for the affected postsynaptic neurons in CLASS_AREA
                    affected_cols = list(updated_neurons_postsynaptic)
                    if affected_cols:
                         W_subset = self.brain.connectomes[conn_key][:, affected_cols]
                         norms = np.linalg.norm(W_subset, axis=0, keepdims=True)
                         # Avoid division by zero for columns with all zero weights
                         norms[norms == 0] = 1.0
                         self.brain.connectomes[conn_key][:, affected_cols] /= norms
            else:
                print(f"Warning: Connection {conn_key} not found during reinforcement.")

        else: # Original simpler reinforcement (force CLASS_AREA activation)
            # Force activation of the correct neuron assembly in CLASS_AREA
            class_area.a = np.zeros(class_area.n)
            class_area.a[true_label_index] = 1.0
            class_area.winners = np.array([true_label_index])
            class_area.w = class_area.winners
            # Apply standard Hebbian update for HIGH_LEVEL -> CLASS_AREA
            self.brain.update_weights('HIGH_LEVEL', 'CLASS_AREA', beta)

        # --- Apply Decay (if enabled) ---
        if config.ENABLE_DECAY and config.SYNAPTIC_DECAY_RATE > 0:
            # Apply decay to connections into plastic areas that were involved
            self.apply_decay('LOW_LEVEL', 'MID_LEVEL', config.SYNAPTIC_DECAY_RATE)
            self.apply_decay('MID_LEVEL', 'HIGH_LEVEL', config.SYNAPTIC_DECAY_RATE)
            self.apply_decay('HIGH_LEVEL', 'CLASS_AREA', config.SYNAPTIC_DECAY_RATE)

        # Apply clipping last if it were enabled
        # if config.WEIGHT_CLIP_MAX > 0:
        #     self.brain.clip_weights(config.WEIGHT_CLIP_MAX)

    def get_state(self):
        """Returns the current activation state of all areas."""
        return {name: area.a.copy() for name, area in self.brain.areas.items()}

    def get_weights(self):
        """Returns a copy of the current weight matrices."""
        return {key: W.copy() for key, W in self.brain.connectomes.items()}

    def get_brain(self):
        """Returns the underlying Brain instance."""
        return self.brain 