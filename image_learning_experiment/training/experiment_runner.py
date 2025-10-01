import time
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

# Ensure modules in the parent directory (config, data, models, etc.) can be found
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
import config

# Import components from within the image_learning_experiment package
from data.data_loader import load_cifar10_data, preprocess_images
from models.cifar_brain import CIFAR10Brain
from evaluation.evaluator import CIFAR10Evaluator
# Import Animator from the visualization module (will be created later)
from visualization.animator import Animator
from brain_module import utils as brain_utils # Import utils from brain_module

def run_experiment():
    """Runs the training and evaluation experiment using raw CIFAR-10 data.
       Assumes CIFAR-10 data is downloaded in config.DATA_ROOT/cifar-10-batches-py
    """
    # --- 1. Load and Preprocess Data --- 
    print("Loading CIFAR-10 data...")
    cifar_data_path = os.path.join(config.DATA_ROOT, 'cifar-10-batches-py')
    if not os.path.exists(cifar_data_path):
         print(f"ERROR: CIFAR-10 data not found at {cifar_data_path}")
         print("Please download and extract the CIFAR-10 Python version dataset.")
         return

    X_train, y_train, X_test, y_test = load_cifar10_data(cifar_data_path)
    print(f"Loaded {len(X_train)} training samples and {len(X_test)} test samples.")

    # Preprocess images (flatten and normalize for direct pixel input)
    # Target size should match the 'n' of the LOW_LEVEL area
    target_input_size = config.AREA_CONFIG[config.LOW_LEVEL]['n']
    print(f"Preprocessing images to target size: {target_input_size}")
    X_train_processed = preprocess_images(X_train, target_size=target_input_size)
    X_test_processed = preprocess_images(X_test, target_size=target_input_size)

    # Ensure the preprocessed size matches the target (basic check)
    if X_train_processed.shape[1] != target_input_size:
         print(f"WARNING: Processed training image size ({X_train_processed.shape[1]}) doesn't match target ({target_input_size}). Check preprocessing.")
    if X_test_processed.shape[1] != target_input_size:
         print(f"WARNING: Processed test image size ({X_test_processed.shape[1]}) doesn't match target ({target_input_size}). Check preprocessing.")

    # --- 2. Initialize Model and Animator --- 
    print("Initializing brain model...")
    brain_model = CIFAR10Brain() # Uses config internally

    print("Initializing animator...")
    # Ensure Animator path is relative to workspace root if needed
    # The path in config is relative to workspace root by default
    animator_output_path = os.path.join(parent_dir, config.ANIMATION_OUTPUT_DIR)
    # Make sure the output directory exists
    os.makedirs(animator_output_path, exist_ok=True)
    animator = Animator(brain=brain_model.brain, output_dir=animator_output_path, fps=config.ANIMATION_FPS)
    animator._initialize_connectome_snapshots() # Prepare for capturing frames

    # --- 3. Initial State --- 
    print("Before training connectome stats:")
    brain_model.print_connectome_stats(config.LOW_LEVEL, config.MID_LEVEL)
    brain_model.print_connectome_stats(config.MID_LEVEL, config.HIGH_LEVEL)
    brain_model.print_connectome_stats(config.HIGH_LEVEL, config.CLASS_AREA)

    # --- 4. Training Loop --- 
    print(f"Training on {config.NUM_TRAIN_SAMPLES} samples...")
    start_train_time = time.time()
    # Use config for number of training samples
    num_train_samples = min(config.NUM_TRAIN_SAMPLES, len(X_train_processed))

    snapshot_data = [] # Store data for snapshots

    for i in tqdm(range(num_train_samples), desc="Training Progress"):
        image_vector = X_train_processed[i]
        true_label = y_train[i]

        # Process image through the brain
        # --- Use the new method --- 
        brain_model.parse_receptive_fields(image_vector)
        # --- End change ---

        # Apply reinforcement
        brain_model.reinforce(true_label)

        # Update animator (less frequently)
        if animator and (i % config.ANIMATION_SNAPSHOT_RATE == 0):
            current_state = brain_model.get_state()
            current_weights = brain_model.get_weights()
            snapshot_data.append((current_state, current_weights))

        # Progress indicator
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{num_train_samples} training samples.")

    end_train_time = time.time()
    print(f"Training finished in {end_train_time - start_train_time:.2f} seconds.")

    # --- 5. Post-Training State --- 
    print("\nAfter training connectome stats:")
    brain_model.print_connectome_stats(config.LOW_LEVEL, config.MID_LEVEL)
    brain_model.print_connectome_stats(config.MID_LEVEL, config.HIGH_LEVEL)
    brain_model.print_connectome_stats(config.HIGH_LEVEL, config.CLASS_AREA)

    # --- 6. Save Animations --- 
    print("Saving connectome animations...")
    animator.save_connectome_animations()
    print(f"Animations saved to: {animator_output_path}")

    # --- 7. Evaluation --- 
    print("Evaluating on test set...")
    evaluator = CIFAR10Evaluator(brain_model)

    # Create a simple data loader iterable from the numpy arrays for the evaluator
    # The evaluator expects batches, but we can feed it one large "batch"
    # Or, more simply, adapt the evaluator to take X_test, y_test directly if possible.
    # Let's modify the evaluator call slightly, assuming it can handle numpy arrays directly:
    # We need a way to pass the data. Let's simulate a dataloader yielding one batch.
    test_dataloader_sim = [(X_test_processed, y_test)]

    accuracy, report, avg_time = evaluator.evaluate(
        preprocess_fn=lambda x, size: x, # Preprocessing already done
        data_loader=test_dataloader_sim,
        max_samples=config.NUM_EVAL_SAMPLES
    )

    # --- 8. Results --- 
    print("\n--- Experiment Results ---")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Average prediction time: {avg_time:.4f}s/image")
    print("Classification Report:")
    print(report)

    # --- 9. Save Model & Generate Animations ---
    model_save_path = os.path.join(config.OUTPUT_DIR, config.EXPERIMENT_NAME, "cifar10_brain_final.pkl")
    print(f"Saving final brain state to {model_save_path}...")
    brain_utils.sim_save(brain_model.get_brain(), model_save_path)
    print("Model saved.")

    if animator and snapshot_data:
        print(f"Generating animation from {len(snapshot_data)} snapshots...")
        animator.create_animation_from_snapshots(snapshot_data, 'training_activation_weights.mp4')
        print("Animation saved.")

    if config.PLOT_CONNECTOME:
        print("Plotting final connectomes...")
        connectome_dir = os.path.join(config.OUTPUT_DIR, config.EXPERIMENT_NAME, "connectomes")
        os.makedirs(connectome_dir, exist_ok=True)
        final_weights = brain_model.get_weights()
        for name, W in final_weights.items():
            if W.ndim == 2 and W.size > 0: # Only plot 2D weight matrices
                 plt.figure(figsize=(8, 6))
                 plt.imshow(W, cmap='viridis', aspect='auto')
                 plt.colorbar()
                 plt.title(f'Final Connectome: {name}')
                 plt.xlabel("Postsynaptic Neurons")
                 plt.ylabel("Presynaptic Neurons")
                 plt.tight_layout()
                 save_path = os.path.join(connectome_dir, f"connectome_{name.replace('->','_')}_final.png")
                 plt.savefig(save_path)
                 plt.close()
        print(f"Connectome plots saved to: {connectome_dir}")

    print("Experiment finished.")
    return accuracy

if __name__ == '__main__':
    # Example of how to run the experiment directly
    # Ensure PYTHONPATH includes the project root if running this script directly
    # Or run using `python -m image_learning_experiment.training.experiment_runner`
    run_experiment() 