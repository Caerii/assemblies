import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np
import time
import re # regex for animation file names
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from animator import Animator

import brain

# Define Brain Areas
LOW_LEVEL = "LOW_LEVEL"
MID_LEVEL = "MID_LEVEL"
HIGH_LEVEL = "HIGH_LEVEL"
CLASS_AREA = "CLASS_AREA"

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

# CIFAR-10 Setup - indirect data loader
# def get_cifar10_dataloader(batch_size=100, augment=False):
#     """Creates a DataLoader for the CIFAR-10 dataset."""
#     transform_list = [transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
#     if augment:
#         transform_list.insert(0, transforms.RandomHorizontalFlip())
#     transform = transforms.Compose(transform_list)
#     dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
#     return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

def get_cifar10_dataloader(batch_size=100, augment=False):
    """Creates a DataLoader for the CIFAR-10 dataset."""
    transform_list = [transforms.Resize((32, 32)), transforms.ToTensor()]
    if augment:
        transform_list.insert(0, transforms.RandomHorizontalFlip())
    transform = transforms.Compose(transform_list)
    dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Feature Extraction
# class FeatureExtractor:
#     """Extracts hierarchical feature representations from images using a pretrained ResNet-18 model."""
#     def __init__(self):
#         self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
#         self.model_low = torch.nn.Sequential(*list(self.model.children())[:4]) # Early convolutional layers
#         self.model_mid = torch.nn.Sequential(*list(self.model.children())[4:6]) # Mid-layer blocks
#         self.model_high = torch.nn.Sequential(*list(self.model.children())[6:-2]) # High-level blocks
#         self.model_fc = torch.nn.Sequential(*list(self.model.children())[-2:])  # Final fully connected layers
#         self.model.eval()

#     def extract_features(self, images):
#         """Extracts features at low, mid, and high levels of the model."""
#         with torch.no_grad():
#             low_features = self.model_low(images)
#             mid_features = self.model_mid(low_features)
#             high_features = self.model_high(mid_features)
#             return (
#                 low_features.view(low_features.size(0), -1).cpu().numpy(),
#                 mid_features.view(mid_features.size(0), -1).cpu().numpy(),
#                 high_features.view(high_features.size(0), -1).cpu().numpy(),
#             )

# directly serve as input
def preprocess_images(images, target_size):
    """Preprocess images by resizing and normalizing."""
    if len(images.shape) == 2:  # Handle flat tensors
        images = images.view(-1, 3, int(images.size(-1) ** 0.5), int(images.size(-1) ** 0.5))  # Assuming CIFAR-10 images

    resized_images = torch.nn.functional.interpolate(
        images, size=target_size, mode="bilinear", align_corners=False
    )

    # Normalize the images
    resized_images = resized_images / 255.0  # Scale pixel values to [0, 1]
    resized_images = resized_images.flatten(start_dim=1)  # Flatten for further processing

    return resized_images
# def preprocess_images(images, target_size):
#     """
#     Preprocess images to match the number of neurons in the LOW_LEVEL area.
    
#     Args:
#         images (torch.Tensor): Batch of images.
#         target_size (int): Target size for the flattened image.
        
#     Returns:
#         np.ndarray: Processed images.
#     """
#     flattened_images = images.view(images.size(0), -1)  # Flatten each image
#     resized_images = torch.nn.functional.interpolate(
#         flattened_images.unsqueeze(1), size=target_size, mode='nearest'
#     )
#     return resized_images.squeeze(1).numpy()

# CIFAR-10 Labels Mapping
CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# Brain Simulation
class CIFAR10Brain:
    def __init__(self):
        self.brain = brain.Brain(0.01, save_winners=True)  # Example probability
        self._initialize_areas()
        self.label_assemblies = {}  # Mapping from labels to neuron assemblies in CLASS_AREA area

        self._initialize_assemblies()

    def _initialize_assemblies(self):
        """Initializes label assemblies for all classes."""
        class_area = self.brain.area_by_name[CLASS_AREA]
        for class_name, index in CLASS_INDICES.items():
            start = index * class_area.k
            assembly = np.arange(start, start + class_area.k)
            self.label_assemblies[class_name] = assembly

        # Validate the keys in label_assemblies
        assert set(self.label_assemblies.keys()) == set(CLASS_INDICES.keys()), \
            "[DEBUG] Mismatch in label assemblies initialization!"

        print("[DEBUG] Initialized assemblies for all classes.")
        print(f"[DEBUG] Class assemblies: {self.label_assemblies}")

    def _initialize_areas(self):
        """Initialize brain areas for image processing."""
        self.brain.add_area(LOW_LEVEL, n=1024, k=100, beta=0.05, explicit=True)
        self.brain.add_area(MID_LEVEL, n=512, k=50, beta=0.05, explicit=True)
        self.brain.add_area(HIGH_LEVEL, n=256, k=25, beta=0.05, explicit=True)
        number_of_classes = len(CLASS_INDICES)
        class_area_k = 10  # As defined
        class_area_n = class_area_k * number_of_classes
        self.brain.add_area(CLASS_AREA, n=class_area_n, k=class_area_k, beta=0.05, explicit=True)

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
        if self.activation_fig is None or self.activation_ax is None:
            self.initialize_activation_plot()

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

    def parse_image(self, low_features, _, __, train=False):
        """Process a flattened image directly through LOW_LEVEL and onward."""
        # Reset winners in all areas
        for area in self.brain.area_by_name.values():
            area.winners = np.array([], dtype=np.uint32)
            area.w = 0

        # Activate LOW_LEVEL with flattened image data
        self._activate_area(LOW_LEVEL, low_features)

        if train:
            self.brain.disable_plasticity = False
            print("[DEBUG] Plasticity enabled.")
        else:
            self.brain.disable_plasticity = True
            print("[DEBUG] Plasticity disabled.")

        # Project activations through the hierarchy
        self.brain.project({}, {LOW_LEVEL: [MID_LEVEL]})
        self.brain.project({}, {MID_LEVEL: [HIGH_LEVEL]})
        self.brain.project({}, {HIGH_LEVEL: [CLASS_AREA]})

        # Re-enable plasticity for future training
        self.brain.disable_plasticity = False


    # def parse_image(self, low_features, mid_features, high_features, train=False):
    #     """Process an image through hierarchical brain areas."""
    #     # Reset winners in all areas
    #     for area in self.brain.area_by_name.values():
    #         area.winners = np.array([], dtype=np.uint32)
    #         area.w = 0

    #     # Activate LOW_LEVEL with features
    #     self._activate_area(LOW_LEVEL, low_features)

    #     if train:
    #         # Enable plasticity during training
    #         self.brain.disable_plasticity = False
    #         print("[DEBUG] Plasticity enabled.")

    #     else:
    #         # Disable plasticity during testing
    #         self.brain.disable_plasticity = True
    #         print("[DEBUG] Plasticity disabled.")

    #     self._activate_area(LOW_LEVEL, low_features)
    #     self.brain.project({}, {LOW_LEVEL: [MID_LEVEL]})
    #     self._activate_area(MID_LEVEL, mid_features)
    #     self.brain.project({}, {MID_LEVEL: [HIGH_LEVEL]})
    #     self._activate_area(HIGH_LEVEL, high_features)
    #     self.brain.project({}, {HIGH_LEVEL: [CLASS_AREA]})
    #     self._activate_area_from_projection(CLASS_AREA)

    #     # Re-enable plasticity for future training
    #     self.brain.disable_plasticity = False

    def set_image_input(self, image_features):
        """
        Sets the image features as inputs to the LOW_LEVEL area.
        
        Parameters:
        - image_features (np.ndarray): Feature vector representing the image.
        """
        # Normalize the features between 0 and 1
        normalized_features = (image_features - np.min(image_features)) / (np.ptp(image_features) + 1e-6)
        normalized_features = normalized_features / (np.linalg.norm(normalized_features) + 1e-6)
        
        # Select top-k neurons to activate based on normalized features
        k = self.area_by_name[LOW_LEVEL].k
        top_indices = np.argsort(-normalized_features)[:k]
        
        # Set winners in LOW_LEVEL
        self.area_by_name[LOW_LEVEL].winners = top_indices
        self.area_by_name[LOW_LEVEL].w = k
        
        # Debug
        print(f"[DEBUG] Activated {LOW_LEVEL} with top {k} neurons: {top_indices[:10]}...")
        
        # Project activations to higher areas
        self.project({}, {LOW_LEVEL: [MID_LEVEL]})

    def _activate_area_from_projection(self, area_name):
        """Activate an area based on incoming projections."""
        area = self.brain.area_by_name[area_name]
        input_sums = np.zeros(area.n)
        
        # Iterate over all source areas that project to this area
        for src_area_name in area.beta_by_area:
            src_area = self.brain.area_by_name[src_area_name]
            if src_area.winners.size == 0:
                continue  # Skip if source area has no winners
            connectome = self.brain.connectomes[src_area_name][area_name]
            pre_neurons = src_area.winners
            # Sum the inputs from pre_neurons to all neurons in the target area
            input_sums += connectome[pre_neurons, :].sum(axis=0)

         # Debug: Print statistics of input_sums
        print(f"[DEBUG] Input sums for {area_name}: mean={input_sums.mean()}, max={input_sums.max()}, min={input_sums.min()}")
        
        # Select top-k neurons based on input sums
        k = area.k
        top_indices = np.argsort(-input_sums)[:k]
        area.winners = top_indices
        area.w = k
        print(f"[DEBUG] Activated {area_name} with top {k} neurons from projection: {top_indices[:10]}...")

    def _activate_area(self, area_name, features):
        """Activates a brain area based on the provided feature vector."""
        if features is None or len(features) == 0:
            raise ValueError(f"Cannot activate {area_name}: Feature vector is empty.")

        # Normalize features between 0 and 1
        normalized_features = (features - np.min(features)) / (np.ptp(features) + 1e-6)

        # Select top-k neurons to activate
        area = self.brain.area_by_name[area_name]
        k = area.k
        top_indices = np.argsort(-normalized_features)[:k]
        if len(top_indices) == 0:
            raise ValueError(f"No neurons activated in {area_name}: check feature extraction and normalization.")

        # Set the winners for the area
        area.winners = top_indices  # Use NumPy array directly
        area.w = area.winners.size
        # area.w = len(area.winners) # Should be equal to area.k
        print(f"[DEBUG] Activated {area_name} with top {len(top_indices)} neurons: {top_indices[:10]}...")

    def project_hierarchy(self, from_area, to_area):
        """Project neural assemblies from one area to another, with validation."""
        area = self.brain.area_by_name[from_area]
        print(f"[DEBUG] {from_area} winners size: {area.winners.size}, w: {area.w}")

        if area.winners.size == 0 or area.w == 0:
            raise ValueError(f"No active assemblies in {from_area}, cannot project to {to_area}.")

        print(f"Projecting from {from_area} to {to_area}.")
        print(f"[DEBUG] {from_area} has {area.w} winners before projection.")

        # Prepare the projection mapping
        areas_by_stim = {}  # No stimuli in this case
        dst_areas_by_src_area = {from_area: [to_area]}

        # Perform the projection
        self.brain.project(areas_by_stim, dst_areas_by_src_area)

    def predict_label(self):
        """Predicts the label based on activations in CLASS_AREA."""
        # Project from HIGH_LEVEL to CLASS_AREA without plasticity
        self.brain.disable_plasticity = True
        self.brain.project({}, {HIGH_LEVEL: [CLASS_AREA]})
        self.brain.disable_plasticity = False

        class_area = self.brain.area_by_name[CLASS_AREA]
        object_winners = class_area.winners

        # Ensure scores only use class names
        scores = {
            label: np.intersect1d(object_winners, assembly).size
            for label, assembly in self.label_assemblies.items()
        }

        # Debugging: Print scores
        print(f"[DEBUG] Prediction scores: {scores}")

        # Predict the label with the highest score
        predicted_label_name = max(scores, key=scores.get, default=None)  # This is a class name
        return predicted_label_name
    
    def print_connectome_stats(self, src_area_name, dst_area_name):
        """Prints statistics of the connectome between two areas."""
        connectome = self.brain.connectomes[src_area_name][dst_area_name]
        mean_weight = np.mean(connectome)
        max_weight = np.max(connectome)
        min_weight = np.min(connectome)
        print(f"[DEBUG] Connectome {src_area_name} -> {dst_area_name}: mean={mean_weight}, max={max_weight}, min={min_weight}")

    # def reinforce(self, predicted_label, true_label):
    #     """Reinforces synaptic connections based on the true label."""
    #     true_class_name = CIFAR10_CLASSES[true_label]
    #     assert true_class_name in self.label_assemblies, f"[DEBUG] Missing assembly for {true_class_name}!"

    #     class_index = CLASS_INDICES[true_class_name]
    #     class_area = self.brain.area_by_name[CLASS_AREA]
    #     high_level_area = self.brain.area_by_name[HIGH_LEVEL]
    #     mid_level_area = self.brain.area_by_name[MID_LEVEL]
    #     low_level_area = self.brain.area_by_name[LOW_LEVEL]

    #     # Activate the correct class assembly
    #     assembly_start = class_index * class_area.k
    #     class_area.winners = np.arange(assembly_start, assembly_start + class_area.k)
    #     class_area.w = class_area.k

    #     # Update or initialize the label assembly
    #     object_winners = class_area.winners.copy()
    #     if true_class_name not in self.label_assemblies:
    #         self.label_assemblies[true_class_name] = object_winners
    #     else:
    #         existing_assembly = self.label_assemblies[true_class_name]
    #         updated_assembly = np.union1d(existing_assembly, object_winners)
    #         self.label_assemblies[true_class_name] = updated_assembly

    #     # Reinforce connections from HIGH_LEVEL to CLASS_AREA
    #     connectome_high_class = self.brain.connectomes[HIGH_LEVEL][CLASS_AREA]
    #     beta_high_class = class_area.beta_by_area[HIGH_LEVEL]
    #     pre_neurons_high = high_level_area.winners
    #     post_neurons_class = class_area.winners
    #     connectome_high_class[np.ix_(pre_neurons_high, post_neurons_class)] *= 1 + beta_high_class

    #     # Optionally, weaken connections to other labels to encourage competition
    #     for label, assembly in self.label_assemblies.items():
    #         if label != true_class_name:
    #             beta_factor = 1 - beta_high_class / 2  # Half-strength weakening
    #             connectome_high_class[np.ix_(pre_neurons_high, assembly)] *= beta_factor

    #     # Reinforce between MID_LEVEL and HIGH_LEVEL
    #     connectome_mid_high = self.brain.connectomes[MID_LEVEL][HIGH_LEVEL]
    #     beta_mid_high = high_level_area.beta_by_area[MID_LEVEL]
    #     pre_neurons_mid = mid_level_area.winners
    #     post_neurons_high = high_level_area.winners
    #     connectome_mid_high[np.ix_(pre_neurons_mid, post_neurons_high)] *= 1 + beta_mid_high

    #     # Reinforce between LOW_LEVEL and MID_LEVEL
    #     connectome_low_mid = self.brain.connectomes[LOW_LEVEL][MID_LEVEL]
    #     beta_low_mid = mid_level_area.beta_by_area[LOW_LEVEL]
    #     pre_neurons_low = low_level_area.winners
    #     post_neurons_mid = mid_level_area.winners
    #     connectome_low_mid[np.ix_(pre_neurons_low, post_neurons_mid)] *= 1 + beta_low_mid

    #     # Debug: Print reinforcement statistics
    #     print("[DEBUG] Reinforcement stats:")
    #     print(f"  True label: {true_class_name}, Predicted label: {predicted_label}")
    #     print(f"  Reinforced CLASS_AREA assembly for class {true_class_name}.")
    #     print(f"  Reinforced connections from HIGH_LEVEL to CLASS_AREA.")
    #     print(f"  Reinforced connections from MID_LEVEL to HIGH_LEVEL.")
    #     print(f"  Reinforced connections from LOW_LEVEL to MID_LEVEL.")

    def apply_decay(self, connectome, decay_rate=0.005):
        """
        Applies decay to a connectome matrix to simulate forgetting.
        """
        if isinstance(connectome, np.ndarray):
            connectome *= (1 - decay_rate)
        else:
            raise TypeError(f"Expected connectome as numpy array, got {type(connectome)}")

    def normalize_connectome(self, connectome):
        """
        Normalizes the given connectome weights to be within a defined range or with a specific norm.
        Args:
            connectome (np.ndarray): The connectome matrix to normalize.
        """
        # Normalize the connectome weights to have a max of 1
        max_value = np.max(connectome)
        if max_value > 0:
            connectome /= max_value
        print(f"[DEBUG] Normalized connectome. Max value after normalization: {np.max(connectome)}")

    def reinforce(self, predicted_label, true_label):
        """Reinforces synaptic connections based on the true label."""
        true_class_name = CIFAR10_CLASSES[true_label]
        assert true_class_name in self.label_assemblies, f"[DEBUG] Missing assembly for {true_class_name}!"

        class_index = CLASS_INDICES[true_class_name]
        class_area = self.brain.area_by_name[CLASS_AREA]
        high_level_area = self.brain.area_by_name[HIGH_LEVEL]
        mid_level_area = self.brain.area_by_name[MID_LEVEL]
        low_level_area = self.brain.area_by_name[LOW_LEVEL]

        # Activate the correct class assembly
        assembly_start = class_index * class_area.k
        class_area.winners = np.arange(assembly_start, assembly_start + class_area.k)
        class_area.w = class_area.k

        # Update or initialize the label assembly
        object_winners = class_area.winners.copy()
        if true_class_name not in self.label_assemblies:
            self.label_assemblies[true_class_name] = object_winners
        else:
            existing_assembly = self.label_assemblies[true_class_name]
            updated_assembly = np.union1d(existing_assembly, object_winners)
            self.label_assemblies[true_class_name] = updated_assembly

        # Apply decay
        self.apply_decay(self.brain.connectomes[HIGH_LEVEL][CLASS_AREA])
        self.apply_decay(self.brain.connectomes[MID_LEVEL][HIGH_LEVEL])
        self.apply_decay(self.brain.connectomes[LOW_LEVEL][MID_LEVEL])

        # Reinforce connections from HIGH_LEVEL to CLASS_AREA
        connectome_high_class = self.brain.connectomes[HIGH_LEVEL][CLASS_AREA]
        beta_high_class = class_area.beta_by_area[HIGH_LEVEL]
        pre_neurons_high = high_level_area.winners
        post_neurons_class = class_area.winners
        connectome_high_class[np.ix_(pre_neurons_high, post_neurons_class)] *= 1 + beta_high_class

        # Penalize incorrect predictions
        if predicted_label != true_label:
            # Determine the wrong assembly based on the predicted label
            if isinstance(predicted_label, int):  # If predicted_label is an index
                predicted_class_name = CIFAR10_CLASSES[predicted_label]
                wrong_assembly = self.label_assemblies[predicted_class_name]
            elif isinstance(predicted_label, str):  # If predicted_label is a class name
                wrong_assembly = self.label_assemblies[predicted_label]
            else:
                raise TypeError(f"Unsupported predicted_label type: {type(predicted_label)}")

            connectome_high_class[np.ix_(pre_neurons_high, wrong_assembly)] *= (1 - beta_high_class / 2)

        # Reinforce between MID_LEVEL and HIGH_LEVEL
        connectome_mid_high = self.brain.connectomes[MID_LEVEL][HIGH_LEVEL]
        beta_mid_high = high_level_area.beta_by_area[MID_LEVEL]
        pre_neurons_mid = mid_level_area.winners
        post_neurons_high = high_level_area.winners
        connectome_mid_high[np.ix_(pre_neurons_mid, post_neurons_high)] *= 1 + beta_mid_high

        # Reinforce between LOW_LEVEL and MID_LEVEL
        connectome_low_mid = self.brain.connectomes[LOW_LEVEL][MID_LEVEL]
        beta_low_mid = mid_level_area.beta_by_area[LOW_LEVEL]
        pre_neurons_low = low_level_area.winners
        post_neurons_mid = mid_level_area.winners
        connectome_low_mid[np.ix_(pre_neurons_low, post_neurons_mid)] *= 1 + beta_low_mid

        # Normalize connectomes
        self.normalize_connectome(connectome_high_class)
        self.normalize_connectome(connectome_mid_high)
        self.normalize_connectome(connectome_low_mid)

        # Debugging stats
        print("[DEBUG] Reinforcement stats:")
        print(f"  True label: {true_class_name}, Predicted label: {predicted_label}")
        print(f"  Reinforced CLASS_AREA assembly for class {true_class_name}.")
        print(f"  Reinforced connections from HIGH_LEVEL to CLASS_AREA.")
        print(f"  Reinforced connections from MID_LEVEL to HIGH_LEVEL.")
        print(f"  Reinforced connections from LOW_LEVEL to MID_LEVEL.")


# Evaluator
class CIFAR10Evaluator:
    """Evaluates the brain simulation model."""
    def __init__(self, brain):
        self.brain_model : CIFAR10Brain = brain
        self.y_true = []
        self.y_pred = []

    def evaluate(self, preprocess_fn, dataloader, max_samples=100):
        """Evaluates the brain model using raw image inputs."""
        num_samples_processed = 0
        total_time = 0.0

        for images, labels in dataloader:
            # Flatten images to ensure compatibility with brain input
            processed_images = preprocess_fn(images, target_size=(32,32))  # Convert to numpy array

            start_time = time.time()
            for image, label in zip(processed_images, labels):
                # Ensure image is normalized and processed
                normalized_image = (image - torch.min(image)) / (torch.max(image) - torch.min(image) + 1e-6)

                # Activate brain areas using the normalized image
                self.brain_model.brain.activate_with_image(LOW_LEVEL, normalized_image)
                self.brain_model.brain.activate_with_image(MID_LEVEL, normalized_image)
                self.brain_model.brain.activate_with_image(HIGH_LEVEL, normalized_image)

                # Perform projections between areas
                self.brain_model.brain.project_with_image(
                    image=normalized_image,
                    areas_by_stim={},  # Specify stimuli if needed
                    dst_areas_by_src_area={
                        LOW_LEVEL: [MID_LEVEL],
                        MID_LEVEL: [HIGH_LEVEL],
                        HIGH_LEVEL: [CLASS_AREA],
                    },
                    input_area=LOW_LEVEL,
                )

                # Predict the label based on CLASS_AREA activations
                predicted_label_name = self.brain_model.predict_label()
                if predicted_label_name is None:
                    predicted_label = -1  # Default for failed predictions
                else:
                    predicted_label = CLASS_INDICES[predicted_label_name]  # Map to numeric index

                # Record ground truth and prediction
                self.y_true.append(label.item())  # Ground truth as integer
                self.y_pred.append(predicted_label)  # Predicted as integer

                num_samples_processed += 1
                if num_samples_processed >= max_samples:
                    break
            total_time += time.time() - start_time
            if num_samples_processed >= max_samples:
                break

        # Compute evaluation metrics
        avg_time = total_time / num_samples_processed if num_samples_processed > 0 else 0
        accuracy = accuracy_score(self.y_true, self.y_pred)

        # Prepare labels and target_names for classification_report
        labels = list(range(-1, 10))  # Include -1 for unknown predictions
        target_names = ['unknown'] + CIFAR10_CLASSES

        report = classification_report(
            self.y_true,
            self.y_pred,
            labels=labels,
            target_names=target_names,
            zero_division=0,
        )

        return accuracy, report, avg_time

    # def evaluate(self, dataloader, feature_extractor, max_samples=100):
    #     """Evaluates the brain model on the given dataloader."""
    #     num_samples_processed = 0
    #     total_time = 0.0

    #     for images, labels in dataloader:
    #         low_features, mid_features, high_features = feature_extractor.extract_features(images)
    #         start_time = time.time()
    #         for low, mid, high, label in zip(low_features, mid_features, high_features, labels):
    #             self.brain.parse_image(low, mid, high, train=False)

    #             predicted_label_name = self.brain.predict_label()
    #             if predicted_label_name is None:
    #                 predicted_label = -1  # Default for failed predictions
    #             else:
    #                 predicted_label = CLASS_INDICES[predicted_label_name]  # Map to numeric index

    #             self.y_true.append(label.item())  # Ground truth as integer
    #             self.y_pred.append(predicted_label)  # Predicted as integer
    #             num_samples_processed += 1
    #             if num_samples_processed >= max_samples:
    #                 break
    #         total_time += time.time() - start_time
    #         if num_samples_processed >= max_samples:
    #             break

    #     avg_time = total_time / num_samples_processed if num_samples_processed > 0 else 0
    #     accuracy = accuracy_score(self.y_true, self.y_pred)

    #     # Prepare labels and target_names for classification_report
    #     labels = list(range(-1, 10))  # Include -1 for unknown predictions
    #     target_names = ['unknown'] + CIFAR10_CLASSES

    #     report = classification_report(
    #         self.y_true,
    #         self.y_pred,
    #         labels=labels,
    #         target_names=target_names,
    #         zero_division=0
    #     )
    #     return accuracy, report, avg_time

# Training and Testing
# def train_and_test():
#     """Trains and evaluates the CIFAR-10 Brain model."""
#     dataloader = get_cifar10_dataloader(batch_size=64, augment=True)
#     feature_extractor = FeatureExtractor()
#     brain_model = CIFAR10Brain()

#     # Initialize connectome snapshots
#     animator = Animator(brain=brain_model, output_dir="animations", fps=10)
#     animator._initialize_connectome_snapshots()

#     # Before training
#     print("Before training connectome stats:")
#     brain_model.print_connectome_stats(LOW_LEVEL, MID_LEVEL)
#     brain_model.print_connectome_stats(MID_LEVEL, HIGH_LEVEL)
#     brain_model.print_connectome_stats(HIGH_LEVEL, CLASS_AREA)

#     print("Training...")
#     num_train_samples = 100  # Increase the number of training samples
#     num_samples_processed = 0
#     for images, labels in dataloader:
#         low_features, mid_features, high_features = feature_extractor.extract_features(images)

#         for low, mid, high, label in zip(low_features, mid_features, high_features, labels):
#             # Train with projections and plasticity enabled
#             brain_model.parse_image(low, mid, high, train=True)
#             predicted_label = brain_model.predict_label()

#             brain_model.reinforce(predicted_label, label.item())

#             # Update only the required connectomes
#             animator.update_connectome(LOW_LEVEL, MID_LEVEL, brain_model.brain.connectomes[LOW_LEVEL][MID_LEVEL])
#             animator.add_snapshot(LOW_LEVEL, MID_LEVEL, brain_model.brain.connectomes[LOW_LEVEL][MID_LEVEL])
#             animator.update_connectome(MID_LEVEL, HIGH_LEVEL, brain_model.brain.connectomes[MID_LEVEL][HIGH_LEVEL])
#             animator.add_snapshot(MID_LEVEL, HIGH_LEVEL, brain_model.brain.connectomes[MID_LEVEL][HIGH_LEVEL])
#             animator.update_connectome(HIGH_LEVEL, CLASS_AREA, brain_model.brain.connectomes[HIGH_LEVEL][CLASS_AREA])
#             animator.add_snapshot(HIGH_LEVEL, CLASS_AREA, brain_model.brain.connectomes[HIGH_LEVEL][CLASS_AREA])

#             num_samples_processed += 1
#             if num_samples_processed >= num_train_samples:
#                 break
#         if num_samples_processed >= num_train_samples:
#             break

#     # After training
#     print("\n[DEBUG] After training:")
#     brain_model.print_connectome_stats(LOW_LEVEL, MID_LEVEL)
#     brain_model.print_connectome_stats(MID_LEVEL, HIGH_LEVEL)
#     brain_model.print_connectome_stats(HIGH_LEVEL, CLASS_AREA)

#     # Visualize connectome
#     animator.save_connectome_animations()

#     brain_model.brain.disable_plasticity = True  # Disable plasticity during evaluation
#     print("Evaluating...")
#     evaluator = CIFAR10Evaluator(brain_model)
#     accuracy, report, avg_time = evaluator.evaluate(dataloader, feature_extractor, max_samples=100)

#     # Visualize activations dynamically
#     # animator.update_activations(brain_model.brain.area_by_name[CLASS_AREA], brain_model.label_assemblies)

#     # Confusion matrix visualization
#     # animator.plot_confusion_matrix(evaluator.y_true, evaluator.y_pred, CIFAR10_CLASSES)

#     print("\n--- Results ---")
#     print(f"Accuracy: {accuracy * 100:.2f}%")
#     print(f"Avg Time: {avg_time:.4f}s/image")
#     print(report)

def train_and_test():
    """Trains and evaluates the CIFAR-10 Brain model."""
    dataloader = get_cifar10_dataloader(batch_size=64, augment=True)
    brain_model = CIFAR10Brain()

    # Initialize connectome snapshots
    animator = Animator(brain=brain_model, output_dir="animations", fps=10)
    animator._initialize_connectome_snapshots()

    # Before training
    print("Before training connectome stats:")
    brain_model.print_connectome_stats(LOW_LEVEL, MID_LEVEL)
    brain_model.print_connectome_stats(MID_LEVEL, HIGH_LEVEL)
    brain_model.print_connectome_stats(HIGH_LEVEL, CLASS_AREA)

    print("Training...")
    num_train_samples = 500
    num_samples_processed = 0

    for images, labels in dataloader:
        processed_images = preprocess_images(images, target_size=1024)  # Use raw pixel data

        for image, label in zip(processed_images, labels):
            # Parse the image directly
            # brain_model.parse_image(image, None, None, train=True)
            # predicted_label = brain_model.predict_label()

            # Ensure image is flattened for processing
            image_np = image.flatten() if isinstance(image, np.ndarray) else image.numpy().flatten()

            brain_model.brain.activate_with_image(
                area_name=LOW_LEVEL,
                image=image_np
            )

            brain_model.brain.activate_with_image(
                area_name=MID_LEVEL,
                image=image_np
            )

            brain_model.brain.activate_with_image(
                area_name=HIGH_LEVEL,
                image=image_np
            )

            brain_model.brain.project_with_image(
                image=image_np,
                areas_by_stim={},  # Specify stimuli if needed
                dst_areas_by_src_area={
                    LOW_LEVEL: [MID_LEVEL],
                    MID_LEVEL: [HIGH_LEVEL],
                    HIGH_LEVEL: [CLASS_AREA]
                },
                input_area=LOW_LEVEL
            )
            predicted_label = brain_model.predict_label()

            brain_model.reinforce(predicted_label, label.item())

            # Update connectome snapshots
            animator.update_connectome(LOW_LEVEL, MID_LEVEL, brain_model.brain.connectomes[LOW_LEVEL][MID_LEVEL])
            animator.add_snapshot(LOW_LEVEL, MID_LEVEL, brain_model.brain.connectomes[LOW_LEVEL][MID_LEVEL])

            animator.update_connectome(MID_LEVEL, HIGH_LEVEL, brain_model.brain.connectomes[MID_LEVEL][HIGH_LEVEL])
            animator.add_snapshot(MID_LEVEL, HIGH_LEVEL, brain_model.brain.connectomes[MID_LEVEL][HIGH_LEVEL])

            animator.update_connectome(HIGH_LEVEL, CLASS_AREA, brain_model.brain.connectomes[HIGH_LEVEL][CLASS_AREA])
            animator.add_snapshot(HIGH_LEVEL, CLASS_AREA, brain_model.brain.connectomes[HIGH_LEVEL][CLASS_AREA])

            num_samples_processed += 1
            if num_samples_processed >= num_train_samples:
                break
        if num_samples_processed >= num_train_samples:
            break

    # After training
    print("\n[DEBUG] After training:")
    brain_model.print_connectome_stats(LOW_LEVEL, MID_LEVEL)
    brain_model.print_connectome_stats(MID_LEVEL, HIGH_LEVEL)
    brain_model.print_connectome_stats(HIGH_LEVEL, CLASS_AREA)

    # Save animations
    animator.save_connectome_animations()

    print("Evaluating...")
    evaluator = CIFAR10Evaluator(brain_model)
    accuracy, report, avg_time = evaluator.evaluate(preprocess_fn=preprocess_images, dataloader=dataloader, max_samples=500)

    print("\n--- Results ---")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Avg Time: {avg_time:.4f}s/image")
    print(report)


if __name__ == "__main__":
    train_and_test()
