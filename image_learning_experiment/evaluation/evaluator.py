import time
import numpy as np
# Removed: import torch
from sklearn.metrics import classification_report, accuracy_score

# Assuming config.py and models/ are accessible from the parent directory
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
import config
# Updated import path for the specific brain model
from models.cifar_brain import CIFAR10Brain

class CIFAR10Evaluator:
    """Evaluates the brain simulation model."""
    def __init__(self, brain_model: CIFAR10Brain):
        self.brain_model = brain_model
        self.y_true = []
        self.y_pred = []

    def evaluate(self, preprocess_fn, data_loader, max_samples=config.NUM_EVAL_SAMPLES):
        """Evaluates the brain model using preprocessed image inputs.
        Assumes data_loader yields tuples of (image_batch, label_batch).
        Assumes preprocess_fn takes image_batch and target_size, returns numpy array.
        """
        self.y_true = [] # Reset for each evaluation call
        self.y_pred = [] # Reset for each evaluation call
        num_samples_processed = 0
        total_time = 0.0

        # Determine the target size needed for preprocessing based on config
        # The evaluation might use a different size/format than training
        # We'll use the PREPROCESS_TRAIN_TARGET_SIZE for consistency with how
        # the brain model expects input to LOW_LEVEL, assuming direct pixel use.
        target_size_for_eval = config.AREA_CONFIG[config.LOW_LEVEL]['n']

        for image_batch, label_batch in data_loader:
            # Preprocessing is now assumed to be done *before* calling evaluate.
            # The data_loader should yield already processed image batches.
            # REMOVED: processed_images = preprocess_fn(image_batch, target_size=target_size_for_eval)
            processed_images = image_batch # Directly use the input batch

            start_time = time.time()
            for image, label in zip(processed_images, label_batch):
                # image should now be a preprocessed numpy array (e.g., flattened vector)

                # --- Simulate brain processing for one image --- 
                # Reset winners (assuming parse_image or similar doesn't reset automatically)
                for area in self.brain_model.brain.area_by_name.values():
                    area.winners = np.array([], dtype=np.uint32)
                    area.w = 0

                # Activate LOW_LEVEL with the input image (no training)
                self.brain_model._activate_area(config.LOW_LEVEL, image)

                # Project through hierarchy (no training)
                self.brain_model.brain.project({}, {config.LOW_LEVEL: [config.MID_LEVEL]})
                self.brain_model.brain.project({}, {config.MID_LEVEL: [config.HIGH_LEVEL]})
                # Prediction itself handles the final projection to CLASS_AREA
                # self.brain_model.brain.project({}, {config.HIGH_LEVEL: [config.CLASS_AREA]})

                # --- Prediction --- 
                predicted_label_name = self.brain_model.predict_label() # This calls _activate_area_from_projection for CLASS_AREA

                # Map prediction to label index
                if predicted_label_name is None:
                    predicted_label = -1 # Represent failed prediction
                else:
                    # Use config for CLASS_INDICES
                    predicted_label = config.CLASS_INDICES.get(predicted_label_name, -1) # Use .get for safety
                # ------------------------------------------------- 

                self.y_true.append(label.item() if hasattr(label, 'item') else label) # Handle potential tensor label
                self.y_pred.append(predicted_label)

                num_samples_processed += 1
                if num_samples_processed >= max_samples:
                    break
            # --- End Batch Loop --- 

            batch_time = time.time() - start_time
            total_time += batch_time
            # print(f"Processed batch in {batch_time:.2f}s") # Optional batch timing

            if num_samples_processed >= max_samples:
                break
        # --- End Dataloader Loop --- 

        if num_samples_processed == 0:
            print("Warning: No samples processed during evaluation.")
            return 0.0, "No samples processed.", 0.0

        avg_time = total_time / num_samples_processed
        accuracy = accuracy_score(self.y_true, self.y_pred)

        # Use config for class names
        labels = list(range(-1, len(config.CIFAR10_CLASSES)))
        target_names = ['unknown'] + config.CIFAR10_CLASSES

        report = classification_report(
            self.y_true,
            self.y_pred,
            labels=labels,
            target_names=target_names,
            zero_division=0,
        )

        return accuracy, report, avg_time 