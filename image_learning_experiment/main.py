# Main entry point for the image learning experiment

import sys
import os

# Ensure the image_learning_experiment package and its parent dir are discoverable
# Add the directory containing image_learning_experiment to sys.path
package_parent_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, package_parent_dir) 
# Also add the workspace root (parent of image_learning_experiment dir)
workspace_root = os.path.abspath(os.path.join(package_parent_dir, '..'))
sys.path.insert(0, workspace_root)

# Import the main experiment function from the new location
from training.experiment_runner import run_experiment

if __name__ == "__main__":
    print("Starting image learning experiment (Refactored)...")
    run_experiment()
    print("Experiment finished.")
