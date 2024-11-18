# image_learner.py

from brain import Brain
import numpy as np

LOW_LEVEL = "LOW_LEVEL"
MID_LEVEL = "MID_LEVEL"
HIGH_LEVEL = "HIGH_LEVEL"
OBJECT = "OBJECT"

class CIFAR10Brain:
    def __init__(self):
        self.brain = Brain(p=0.01, seed=42)
        self._initialize_areas()

    def _initialize_areas(self):
        """Initialize brain areas for image processing."""
        self.brain.add_area(LOW_LEVEL, n=10000, k=512, beta=0.05, explicit=True)
        self.brain.add_area(MID_LEVEL, n=5000, k=256, beta=0.05, explicit=True)
        self.brain.add_area(HIGH_LEVEL, n=2000, k=128, beta=0.05, explicit=True)
        self.brain.add_area(OBJECT, n=1000, k=64, beta=0.05, explicit=True)

    def parse_image(self, low_features, mid_features, high_features):
        """Process an image through hierarchical brain areas."""
        print("[DEBUG] Parsing image features...")
        self._activate_area(LOW_LEVEL, low_features)
        self.project_hierarchy(LOW_LEVEL, MID_LEVEL)
        self._activate_area(MID_LEVEL, mid_features)
        self.project_hierarchy(MID_LEVEL, HIGH_LEVEL)
        self._activate_area(HIGH_LEVEL, high_features)
        self.project_hierarchy(HIGH_LEVEL, OBJECT)
        print("[DEBUG] Parsing complete.")

    def _activate_area(self, area_name, features):
        """Activates a brain area based on the provided feature vector."""
        if features is None or len(features) == 0:
            raise ValueError(f"Cannot activate {area_name}: Feature vector is empty.")

        # Normalize features between 0 and 1
        normalized_features = utils.normalize_features(features)

        # Select top-k neurons to activate
        area = self.brain.areas[area_name]
        top_indices = utils.select_top_k_indices(normalized_features, area.k)
        if len(top_indices) == 0:
            raise ValueError(f"No neurons activated in {area_name}: check feature extraction and normalization.")

        # Set the winners for the area
        area.winners = top_indices
        print(f"[DEBUG] Activated {area_name} with top {len(top_indices)} neurons: {top_indices[:10]}...")

    def project_hierarchy(self, from_area, to_area):
        """Project neural assemblies from one area to another, with validation."""
        area = self.brain.areas[from_area]
        if len(area.winners) == 0:
            raise ValueError(f"No active assemblies in {from_area}, cannot project to {to_area}.")

        print(f"Projecting from {from_area} to {to_area}.")
        print(f"[DEBUG] {from_area} has {area.w} winners before projection.")

        # Perform the projection
        self.brain.project({}, {from_area: [to_area]})

# The rest of your code for training and testing
