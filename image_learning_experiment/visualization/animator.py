import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Assuming config.py is in the parent directory
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
import config

# Import base Brain class for type hinting (optional but good practice)
from brain_module.core import Brain

class Animator:
    def __init__(self, brain: Brain, output_dir="animations", fps=10):
        """Initializes the Animator.

        Args:
            brain: The base Brain object (from brain_module.core).
            output_dir: Directory to save animations.
            fps: Frames per second for the animation.
        """
        self.brain = brain
        self.output_dir = output_dir
        self.fps = fps
        # Define which connectomes to track based on config area names
        self.tracked_connectomes = [
            (config.LOW_LEVEL, config.MID_LEVEL),
            (config.MID_LEVEL, config.HIGH_LEVEL),
            (config.HIGH_LEVEL, config.CLASS_AREA),
        ]
        self.connectome_snapshots = { 
            f"{src} to {dst}": [] for src, dst in self.tracked_connectomes
        }
        self.connectome_figures = {}  # For potential dynamic updates (optional)
        os.makedirs(self.output_dir, exist_ok=True)

    def _initialize_connectome_snapshots(self):
        """Initialize empty lists for storing connectome snapshots for tracked connections."""
        for src_area, dst_area in self.tracked_connectomes:
            title = f"{src_area} to {dst_area}"
            self.connectome_snapshots[title] = []
            print(f"[DEBUG] Initialized snapshot tracking for {title}")

    def add_snapshot(self, src_area, dst_area, connectome):
        """Add a snapshot of the connectome if it's being tracked."""
        title = f"{src_area} to {dst_area}"
        if title in self.connectome_snapshots:
            # Ensure connectome is a numpy array before copying
            if isinstance(connectome, np.ndarray):
                self.connectome_snapshots[title].append(connectome.copy())
                # print(f"[DEBUG] Added snapshot for {title}. Total: {len(self.connectome_snapshots[title])}") # Verbose
            else:
                print(f"[WARNING] Snapshot skipped for {title}: Connectome is not a numpy array (type: {type(connectome)})." )
        # else: # No need to print if not tracked, it's expected
            # print(f"[DEBUG] Skipping snapshot for {title}. Not in tracked list.")

    def update_connectome(self, src_area, dst_area, connectome):
        """(Optional) Updates a dynamic connectome visualization if figure exists.
           Also adds a snapshot for animation generation.
        """
        title = f"{src_area} to {dst_area}"
        
        # Add snapshot regardless of dynamic plot update
        self.add_snapshot(src_area, dst_area, connectome)
        
        # Dynamic update part (can be computationally expensive)
        if title in self.connectome_figures:
            # Update existing visualization
            fig = self.connectome_figures[title]["fig"]
            cax = self.connectome_figures[title]["cax"]
            
            if isinstance(connectome, np.ndarray):
                cax.set_data(connectome)
                cax.autoscale()  # Adjust color scale to match new data
                fig.canvas.draw_idle()
                plt.pause(0.001)  # Allow updates to render
            else:
                 print(f"[WARNING] Cannot update plot for {title}: Connectome is not a numpy array.")
        # else: # Only create figure if explicitly needed for dynamic view
            # print(f"[DEBUG] No dynamic figure for {title}, snapshot added only.")
            pass 

    def _create_or_get_dynamic_figure(self, title, initial_connectome):
         """Creates a figure for dynamic updating if it doesn't exist."""
         if title not in self.connectome_figures:
            if not isinstance(initial_connectome, np.ndarray) or initial_connectome.ndim != 2:
                 print(f"[WARNING] Cannot create figure for {title}: Initial connectome is invalid.")
                 return None, None, None, None
            
            fig, ax = plt.subplots(figsize=(10, 8))
            try:
                cax = ax.imshow(initial_connectome, cmap="hot", aspect="auto")
                colorbar = fig.colorbar(cax, ax=ax)
            except Exception as e:
                 print(f"[ERROR] Failed to create imshow for {title}: {e}")
                 plt.close(fig)
                 return None, None, None, None
                 
            ax.set_title(title)
            ax.set_xlabel("Post-synaptic Neurons")
            ax.set_ylabel("Pre-synaptic Neurons")
            self.connectome_figures[title] = {"fig": fig, "ax": ax, "cax": cax, "colorbar": colorbar}
            print(f"[DEBUG] Created dynamic figure for {title}")
            return fig, ax, cax, colorbar
         else:
            return (
                self.connectome_figures[title]["fig"],
                self.connectome_figures[title]["ax"],
                self.connectome_figures[title]["cax"],
                self.connectome_figures[title]["colorbar"]
            )

    def save_connectome_animations(self):
        """Generate animations for all tracked connectomes that have snapshots."""
        # Ensure matplotlib backend is suitable for saving animations without display
        # plt.switch_backend('Agg') # Uncomment if running in headless environment
        
        writer = PillowWriter(fps=self.fps)
        saved_animations = []

        for title, snapshots in self.connectome_snapshots.items():
            if not snapshots:
                print(f"[INFO] No snapshots for {title}, skipping animation.")
                continue
                
            # Validate snapshots are numpy arrays
            if not all(isinstance(snap, np.ndarray) for snap in snapshots):
                print(f"[WARNING] Invalid snapshot data for {title}, skipping animation.")
                continue
            # Validate snapshots are 2D
            if not all(snap.ndim == 2 for snap in snapshots):
                print(f"[WARNING] Snapshots for {title} are not all 2D, skipping animation.")
                continue

            sanitized_title = re.sub(r'[<>:"/\\|?*\']', '_', title)  # More aggressive sanitization
            output_file = os.path.join(self.output_dir, f"{sanitized_title}.gif")

            print(f"Generating animation for {title} ({len(snapshots)} frames)...")
            
            # Create a figure specifically for saving the animation
            fig_anim, ax_anim = plt.subplots(figsize=(10, 8))
            
            # Use the first valid snapshot for initial setup
            try:
                cax_anim = ax_anim.imshow(snapshots[0], cmap="hot", aspect="auto")
                colorbar_anim = fig_anim.colorbar(cax_anim, ax=ax_anim)
            except Exception as e:
                print(f"[ERROR] Failed imshow for animation {title}: {e}")
                plt.close(fig_anim)
                continue # Skip this animation
                
            ax_anim.set_title(title)
            ax_anim.set_xlabel("Post-synaptic Neurons")
            ax_anim.set_ylabel("Pre-synaptic Neurons")

            # Animation update function
            def update_anim(frame):
                cax_anim.set_data(snapshots[frame])
                # Optional: Update colorbar limits if needed per frame
                # cax_anim.autoscale() 
                # colorbar_anim.update_normal(cax_anim)
                return cax_anim,

            try:
                anim = FuncAnimation(fig_anim, update_anim, frames=len(snapshots), blit=True, interval=1000/self.fps)
                anim.save(output_file, writer=writer)
                saved_animations.append(output_file)
                print(f"  Animation saved: {output_file}")
            except Exception as e:
                print(f"[ERROR] Failed to save animation for {title}: {e}")
            finally:
                 # Close the figure used for animation to free memory
                 plt.close(fig_anim)

        print(f"Finished saving animations. {len(saved_animations)} files created.")

    # Removed update_activations as it depends on CIFAR10Brain structure (label_assemblies)
    # Removed update_all_connectomes (specific helper)
    # Removed plot_confusion_matrix (belongs elsewhere, e.g., evaluation or plotting module) 