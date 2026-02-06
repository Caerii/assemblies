import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.metrics import ConfusionMatrixDisplay

LOW_LEVEL = "LOW_LEVEL"
MID_LEVEL = "MID_LEVEL"
HIGH_LEVEL = "HIGH_LEVEL"
CLASS_AREA = "CLASS_AREA"

class Animator:
    def __init__(self, brain, output_dir="animations", fps=10):
        self.output_dir = output_dir
        self.fps = fps
        self.connectome_snapshots = {
            f"{LOW_LEVEL} to {MID_LEVEL}": [],
            f"{MID_LEVEL} to {HIGH_LEVEL}": [],
            f"{HIGH_LEVEL} to {CLASS_AREA}": []
        }  # Snapshots only for the required connectomes
        self.connectome_figures = {}  # To track figures for each connectome
        self.feature_map_fig = None
        self.feature_map_axes = None
        self.activation_fig = None
        self.activation_ax = None
        os.makedirs(self.output_dir, exist_ok=True)

    # ---------- Feature Map Visualizations ----------
    def initialize_feature_map_plot(self, title, grid_size):
        """Initialize a reusable figure for feature maps."""
        self.feature_map_fig, self.feature_map_axes = plt.subplots(
            grid_size, grid_size, figsize=(10, 10)
        )
        plt.suptitle(title)

    def update_feature_maps(self, features, title):
        """Update the feature map plot in real-time."""
        if self.feature_map_fig is None or self.feature_map_axes is None:
            grid_size = int(np.ceil(np.sqrt(features.shape[0])))
            self.initialize_feature_map_plot(title, grid_size)

        for i, ax in enumerate(self.feature_map_axes.flat):
            if i < features.shape[0]:
                ax.clear()
                ax.imshow(features[i], cmap="viridis")
                ax.axis("off")
        self.feature_map_fig.suptitle(title)
        self.feature_map_fig.canvas.draw()
        self.feature_map_fig.canvas.flush_events()

    # ---------- Connectome Visualizations ----------

    def _initialize_connectome_snapshots(self):
        """Initialize empty lists for storing connectome snapshots for specified connections."""
        for src_area, dst_area in [
            (LOW_LEVEL, MID_LEVEL),
            (MID_LEVEL, HIGH_LEVEL),
            (HIGH_LEVEL, CLASS_AREA),
        ]:
            title = f"{src_area} to {dst_area}"
            self.connectome_snapshots[title] = []

    def add_snapshot(self, src_area, dst_area, connectome):
        """Add a snapshot of the connectome for the given source and destination areas."""
        title = f"{src_area} to {dst_area}"
        if title in self.connectome_snapshots:
            self.connectome_snapshots[title].append(connectome.copy())
            print(f"[DEBUG] Added snapshot for {title}. Total snapshots: {len(self.connectome_snapshots[title])}")
        else:
            print(f"[DEBUG] Skipping snapshot for {title}. Not in required list.")

    def update_connectome(self, src_area, dst_area, connectome):
        """Updates the connectome visualization dynamically for specific connectomes."""
        title = f"{src_area} to {dst_area}"
        if title not in self.connectome_snapshots:
            print(f"[DEBUG] Skipping update for {title}. Not in required list.")
            return

        if title not in self.connectome_figures:
            # Create a new figure for this connectome
            fig, ax = plt.subplots(figsize=(10, 8))
            cax = ax.imshow(connectome, cmap="hot", aspect="auto")
            colorbar = fig.colorbar(cax, ax=ax)
            ax.set_title(title)
            ax.set_xlabel("Post-synaptic Neurons")
            ax.set_ylabel("Pre-synaptic Neurons")
            self.connectome_figures[title] = {"fig": fig, "ax": ax, "cax": cax, "colorbar": colorbar}
        else:
            # Update existing visualization
            fig = self.connectome_figures[title]["fig"]
            cax = self.connectome_figures[title]["cax"]
            cax.set_data(connectome)
            cax.autoscale()  # Adjust color scale to match new data
            fig.canvas.draw_idle()
            plt.pause(0.001)  # Allow updates to render

        # Add snapshot
        self.add_snapshot(src_area, dst_area, connectome)

    def save_connectome_animations(self):
        """Generate animations only for the specified connectomes."""
        writer = PillowWriter(fps=self.fps)
        for title, snapshots in self.connectome_snapshots.items():
            if not snapshots:
                print(f"[DEBUG] No snapshots for {title}, skipping animation.")
                continue

            sanitized_title = re.sub(r'[<>:"/\\|?*]', '_', title)
            output_file = os.path.join(self.output_dir, f"{sanitized_title}.gif")

            # Create a figure for the animation
            fig, ax = plt.subplots(figsize=(10, 8))
            cax = ax.imshow(snapshots[0], cmap="hot", aspect="auto")
            colorbar = fig.colorbar(cax, ax=ax)
            ax.set_title(title)
            ax.set_xlabel("Post-synaptic Neurons")
            ax.set_ylabel("Pre-synaptic Neurons")

            def update(frame):
                cax.set_data(snapshots[frame])
                return cax,

            anim = FuncAnimation(fig, update, frames=len(snapshots), blit=True)
            anim.save(output_file, writer=writer)
            plt.close(fig)
            print(f"[DEBUG] Animation saved: {output_file}")

    # ---------- Activation Visualizations ----------
    def initialize_activation_plot(self):
        """Initialize a reusable figure for activation bar chart."""
        self.activation_fig, self.activation_ax = plt.subplots(figsize=(8, 6))
        self.activation_ax.set_xlabel("Class")
        self.activation_ax.set_ylabel("Activations")
        self.activation_ax.set_title("Assembly Activations in CLASS_AREA")

    def update_activations(self):
        """Update the activation bar chart in real-time using brain data."""
        class_area = self.brain.area_by_name[CLASS_AREA]
        activations = {
            label: np.intersect1d(class_area.winners, assembly).size
            for label, assembly in self.brain.label_assemblies.items()
        }
        if self.activation_fig is None or self.activation_ax is None:
            self.initialize_activation_plot()

        self.activation_ax.clear()
        self.activation_ax.bar(activations.keys(), activations.values())
        self.activation_ax.set_title("Assembly Activations in CLASS_AREA")
        self.activation_ax.set_xlabel("Class")
        self.activation_ax.set_ylabel("Activations")
        self.activation_ax.set_xticks(range(len(self.brain.label_assemblies)))
        self.activation_ax.set_xticklabels(self.brain.label_assemblies.keys(), rotation=45)
        self.activation_fig.canvas.draw()
        self.activation_fig.canvas.flush_events()


    # ---------- Helper for Updating All Connectomes ----------
    def update_all_connectomes(self):
        """Visualizes and updates specified connectomes in the brain model."""
        for src_area, dst_area in [
            (LOW_LEVEL, MID_LEVEL),
            (MID_LEVEL, HIGH_LEVEL),
            (HIGH_LEVEL, CLASS_AREA),
        ]:
            if (
                src_area in self.brain.connectomes
                and dst_area in self.brain.connectomes[src_area]
            ):
                connectome = self.brain.connectomes[src_area][dst_area]
                self.update_connectome(src_area, dst_area, connectome)
                self.add_snapshot(src_area, dst_area, connectome)
            else:
                print(f"[DEBUG] Missing connectome: {src_area} -> {dst_area}")

    # Confusion matrix
    def plot_confusion_matrix(self, y_true, y_pred, labels):
        """Plots a confusion matrix."""
        ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=labels, cmap="Blues")
        plt.title("Confusion Matrix")
        plt.show()
