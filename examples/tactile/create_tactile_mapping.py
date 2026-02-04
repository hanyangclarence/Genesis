"""
Interactive Tool for Creating Tactile Point to Image Mapping

This tool allows you to:
1. View all tactile points flattened to a 2D map
2. View a 24x32 pixel grid (tactile image)
3. Select regions on both maps by clicking points
4. Create mappings between selected tactile points and pixels

Usage:
    python create_tactile_mapping.py --tactile-grid merged_tactile_grid.json

Controls:
    Left panel (Tactile Points):
        - Left click/drag: Add points to selection
        - Right click/drag: Remove points from selection

    Right panel (Pixel Grid):
        - Left click/drag: Add pixels to selection
        - Right click/drag: Remove pixels from selection

    Keyboard:
        - 'c': Clear current selection (both panels)
        - 'm': Create mapping from current selection
        - 's': Save all mappings to file
        - 'u': Undo last mapping
        - 'q': Quit
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors

# YZ offsets for visualization layout (from tactile_field_hand.py)
LINK_YZ_OFFSETS = {
    "palm_link": (0.0, 0.0),
    "finger1_link1": (0.05, 0.02),
    "finger1_link2": (0.05, 0.05),
    "finger1_link3": (0.05, 0.08),
    "finger1_link4": (0.05, 0.11),
    "finger1_tip_link": (0.05, 0.14),
    "finger2_link1": (0.02, 0.06),
    "finger2_link2": (0.02, 0.10),
    "finger2_link3": (0.02, 0.14),
    "finger2_link4": (0.02, 0.18),
    "finger2_tip_link": (0.02, 0.21),
    "finger3_link1": (0.0, 0.06),
    "finger3_link2": (0.0, 0.10),
    "finger3_link3": (0.0, 0.14),
    "finger3_link4": (0.0, 0.18),
    "finger3_tip_link": (0.0, 0.21),
    "finger4_link1": (-0.02, 0.06),
    "finger4_link2": (-0.02, 0.10),
    "finger4_link3": (-0.02, 0.14),
    "finger4_link4": (-0.02, 0.18),
    "finger4_tip_link": (-0.02, 0.21),
    "finger5_link1": (-0.04, 0.04),
    "finger5_link2": (-0.04, 0.08),
    "finger5_link3": (-0.04, 0.12),
    "finger5_link4": (-0.04, 0.16),
    "finger5_tip_link": (-0.04, 0.19),
}

# Colors for different mapping groups
COLORS = list(mcolors.TABLEAU_COLORS.values())


class TactileMappingTool:
    def __init__(self, tactile_data, image_shape=(24, 32)):
        self.tactile_data = tactile_data
        self.image_shape = image_shape

        # Parse tactile points
        self.tactile_points = []  # List of (link_name, point_idx, local_pos, offset_pos_2d)
        self.parse_tactile_points()

        # Selection state
        self.selected_tactile_indices = set()  # Indices into self.tactile_points
        self.selected_pixels = set()  # (row, col) tuples

        # Mappings: list of (tactile_indices, pixel_coords, color)
        self.mappings = []

        # Drag state
        self.dragging = False
        self.drag_button = None  # 1 for left (add), 3 for right (remove)
        self.drag_panel = None  # 'tactile' or 'pixel'

        # Setup figure
        self.setup_figure()

    def parse_tactile_points(self):
        """Parse tactile points and compute 2D offset positions."""
        links_data = self.tactile_data.get('links', {})

        for link_name, link_data in links_data.items():
            points = link_data.get('points', [])
            yz_offset = LINK_YZ_OFFSETS.get(link_name, (0.0, 0.0))

            for point_idx, point in enumerate(points):
                local_pos = np.array(point['local'] if isinstance(point, dict) else point)

                # Project to 2D with offset (same logic as tactile_field_hand.py)
                if link_name == "finger1_link2":
                    # Project to XZ plane for thumb
                    offset_pos_2d = np.array([local_pos[0] + yz_offset[0],
                                               local_pos[2] + yz_offset[1]])
                else:
                    # Project to YZ plane
                    offset_pos_2d = np.array([local_pos[1] + yz_offset[0],
                                               local_pos[2] + yz_offset[1]])

                self.tactile_points.append({
                    'link_name': link_name,
                    'point_idx': point_idx,
                    'local_pos': local_pos,
                    'offset_pos_2d': offset_pos_2d,
                })

        print(f"Loaded {len(self.tactile_points)} tactile points from {len(links_data)} links")

    def setup_figure(self):
        """Setup the matplotlib figure with two panels."""
        self.fig, (self.ax_tactile, self.ax_pixel) = plt.subplots(1, 2, figsize=(16, 8))

        # Left panel: Tactile points
        self.ax_tactile.set_title("Tactile Points (Left drag: select, Right drag: deselect)")
        self.ax_tactile.set_xlabel("Y (m)")
        self.ax_tactile.set_ylabel("Z (m)")
        self.ax_tactile.set_aspect('equal')

        # Plot all tactile points
        positions = np.array([p['offset_pos_2d'] for p in self.tactile_points])
        self.tactile_scatter = self.ax_tactile.scatter(
            positions[:, 0], positions[:, 1],
            c='lightgray', s=30, edgecolors='black', linewidths=0.5,
            picker=True, pickradius=5
        )

        # Right panel: Pixel grid
        self.ax_pixel.set_title("24x32 Tactile Image (Left drag: select, Right drag: deselect)")
        self.ax_pixel.set_xlabel("Column")
        self.ax_pixel.set_ylabel("Row")

        # Draw pixel grid
        self.pixel_colors = np.ones((self.image_shape[0], self.image_shape[1], 3)) * 0.9  # Light gray
        self.pixel_image = self.ax_pixel.imshow(
            self.pixel_colors, origin='upper', aspect='equal',
            extent=[-0.5, self.image_shape[1]-0.5, self.image_shape[0]-0.5, -0.5]
        )

        # Draw grid lines
        for i in range(self.image_shape[0] + 1):
            self.ax_pixel.axhline(i - 0.5, color='gray', linewidth=0.5, alpha=0.5)
        for j in range(self.image_shape[1] + 1):
            self.ax_pixel.axvline(j - 0.5, color='gray', linewidth=0.5, alpha=0.5)

        self.ax_pixel.set_xlim(-0.5, self.image_shape[1] - 0.5)
        self.ax_pixel.set_ylim(self.image_shape[0] - 0.5, -0.5)

        # Status text
        self.status_text = self.fig.text(
            0.5, 0.02,
            "Keys: 'c'=clear, 'm'=create mapping, 's'=save, 'u'=undo, 'q'=quit",
            ha='center', fontsize=10, color='blue'
        )

        # Connect events
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.08)

    def on_press(self, event):
        """Handle mouse button press - start drag."""
        if event.inaxes is None:
            return
        if event.button not in [1, 3]:
            return

        self.dragging = True
        self.drag_button = event.button

        if event.inaxes == self.ax_tactile:
            self.drag_panel = 'tactile'
            self.handle_tactile_select(event)
        elif event.inaxes == self.ax_pixel:
            self.drag_panel = 'pixel'
            self.handle_pixel_select(event)

        self.update_display()

    def on_release(self, event):
        """Handle mouse button release - end drag."""
        self.dragging = False
        self.drag_button = None
        self.drag_panel = None

    def on_motion(self, event):
        """Handle mouse motion - continue selection while dragging."""
        if not self.dragging or event.inaxes is None:
            return

        if self.drag_panel == 'tactile' and event.inaxes == self.ax_tactile:
            self.handle_tactile_select(event)
            self.update_display()
        elif self.drag_panel == 'pixel' and event.inaxes == self.ax_pixel:
            self.handle_pixel_select(event)
            self.update_display()

    def handle_tactile_select(self, event):
        """Handle selection on tactile points panel."""
        if event.xdata is None or event.ydata is None:
            return

        # Find closest point
        positions = np.array([p['offset_pos_2d'] for p in self.tactile_points])
        distances = np.sqrt((positions[:, 0] - event.xdata)**2 +
                           (positions[:, 1] - event.ydata)**2)
        closest_idx = np.argmin(distances)

        # Only select if close enough
        if distances[closest_idx] < 0.01:  # Threshold in data coordinates
            if self.drag_button == 1:  # Left - add
                if closest_idx not in self.selected_tactile_indices:
                    self.selected_tactile_indices.add(closest_idx)
                    point = self.tactile_points[closest_idx]
                    print(f"Selected: {point['link_name']}[{point['point_idx']}]")
            elif self.drag_button == 3:  # Right - remove
                self.selected_tactile_indices.discard(closest_idx)

    def handle_pixel_select(self, event):
        """Handle selection on pixel grid panel."""
        if event.xdata is None or event.ydata is None:
            return

        col = int(np.round(event.xdata))
        row = int(np.round(event.ydata))

        if 0 <= row < self.image_shape[0] and 0 <= col < self.image_shape[1]:
            if self.drag_button == 1:  # Left - add
                if (row, col) not in self.selected_pixels:
                    self.selected_pixels.add((row, col))
                    print(f"Selected pixel: ({row}, {col})")
            elif self.drag_button == 3:  # Right - remove
                self.selected_pixels.discard((row, col))

    def on_key(self, event):
        """Handle keyboard events."""
        if event.key == 'c':
            self.clear_selection()
        elif event.key == 'm':
            self.create_mapping()
        elif event.key == 's':
            self.save_mappings()
        elif event.key == 'u':
            self.undo_mapping()
        elif event.key == 'q':
            plt.close(self.fig)
        self.update_display()

    def clear_selection(self):
        """Clear current selection."""
        self.selected_tactile_indices.clear()
        self.selected_pixels.clear()
        print("Selection cleared")

    def create_mapping(self):
        """Create a mapping from current selection."""
        if not self.selected_tactile_indices:
            print("No tactile points selected!")
            return
        if not self.selected_pixels:
            print("No pixels selected!")
            return

        # Get next color
        color_idx = len(self.mappings) % len(COLORS)
        color = COLORS[color_idx]

        # Store mapping
        mapping = {
            'tactile_indices': list(self.selected_tactile_indices),
            'pixels': list(self.selected_pixels),
            'color': color,
        }
        self.mappings.append(mapping)

        # Get link info for display
        links_in_selection = set()
        for idx in self.selected_tactile_indices:
            links_in_selection.add(self.tactile_points[idx]['link_name'])

        print(f"\nMapping #{len(self.mappings)} created:")
        print(f"  Tactile points: {len(self.selected_tactile_indices)} from links: {links_in_selection}")
        print(f"  Pixels: {len(self.selected_pixels)}")
        print(f"  Color: {color}")

        # Clear selection for next mapping
        self.clear_selection()

    def undo_mapping(self):
        """Undo the last mapping."""
        if self.mappings:
            self.mappings.pop()
            print(f"Undid last mapping. {len(self.mappings)} mappings remaining.")
        else:
            print("No mappings to undo.")

    def save_mappings(self, output_path=None):
        """Save mappings to JSON file."""
        if not self.mappings:
            print("No mappings to save!")
            return

        if output_path is None:
            output_path = "tactile_to_image_mapping.json"

        # Build output structure
        output = {
            'image_shape': list(self.image_shape),
            'num_tactile_points': len(self.tactile_points),
            'num_mappings': len(self.mappings),
            'mappings': [],
            'point_to_pixel': {},  # tactile_point_global_idx -> (row, col)
            'pixel_to_points': {},  # "row,col" -> [tactile_point_global_idx, ...]
        }

        for mapping_idx, mapping in enumerate(self.mappings):
            tactile_indices = mapping['tactile_indices']
            pixels = mapping['pixels']

            # Get detailed info for each tactile point
            tactile_info = []
            for idx in tactile_indices:
                point = self.tactile_points[idx]
                tactile_info.append({
                    'global_idx': int(idx),
                    'link_name': point['link_name'],
                    'point_idx': int(point['point_idx']),
                    'local_pos': [float(x) for x in point['local_pos']],
                })

            # Convert pixels to native Python types
            pixels_native = [[int(row), int(col)] for row, col in pixels]

            mapping_data = {
                'mapping_idx': mapping_idx,
                'tactile_points': tactile_info,
                'pixels': pixels_native,
            }
            output['mappings'].append(mapping_data)

            # Build point_to_pixel and pixel_to_points using optimal assignment
            self._compute_optimal_assignment(tactile_indices, pixels, output)

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\nMappings saved to: {output_path}")
        print(f"  Total mappings: {len(self.mappings)}")
        print(f"  Total tactile points mapped: {len(output['point_to_pixel'])}")
        print(f"  Total pixels used: {len(output['pixel_to_points'])}")

    def _compute_optimal_assignment(self, tactile_indices, pixels, output):
        """Compute optimal assignment between tactile points and pixels."""
        from scipy.optimize import linear_sum_assignment

        # Convert to Python native types
        tactile_indices = [int(x) for x in tactile_indices]
        pixels = [(int(row), int(col)) for row, col in pixels]

        n_tactile = len(tactile_indices)
        n_pixels = len(pixels)

        if n_tactile == 0 or n_pixels == 0:
            return

        # Get 2D positions for tactile points (normalized)
        tactile_2d = np.array([self.tactile_points[idx]['offset_pos_2d'] for idx in tactile_indices])
        tactile_2d_norm = (tactile_2d - tactile_2d.min(axis=0)) / (tactile_2d.max(axis=0) - tactile_2d.min(axis=0) + 1e-10)

        # Get 2D positions for pixels (normalized)
        pixels_arr = np.array(pixels)
        pixels_norm = pixels_arr.astype(float)
        pixels_norm[:, 0] = pixels_norm[:, 0] / (self.image_shape[0] - 1)  # row
        pixels_norm[:, 1] = pixels_norm[:, 1] / (self.image_shape[1] - 1)  # col
        # Swap to match tactile_2d format (x, y) vs (row, col)
        pixels_norm = pixels_norm[:, ::-1]  # (col_norm, row_norm) -> similar to (y, z)

        # Compute cost matrix
        cost_matrix = np.zeros((n_tactile, n_pixels))
        for i in range(n_tactile):
            for j in range(n_pixels):
                cost_matrix[i, j] = np.linalg.norm(tactile_2d_norm[i] - pixels_norm[j])

        # Solve assignment (handle n_tactile != n_pixels)
        if n_tactile <= n_pixels:
            # More pixels than tactile points: each tactile point gets one pixel
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for i, j in zip(row_ind, col_ind):
                tactile_idx = int(tactile_indices[i])
                pixel = [int(pixels[j][0]), int(pixels[j][1])]
                output['point_to_pixel'][str(tactile_idx)] = pixel
                pixel_key = f"{pixel[0]},{pixel[1]}"
                if pixel_key not in output['pixel_to_points']:
                    output['pixel_to_points'][pixel_key] = []
                output['pixel_to_points'][pixel_key].append(tactile_idx)
        else:
            # More tactile points than pixels: each pixel gets one or more tactile points
            # Use transposed cost matrix
            row_ind, col_ind = linear_sum_assignment(cost_matrix.T)
            for j, i in zip(row_ind, col_ind):
                tactile_idx = int(tactile_indices[i])
                pixel = [int(pixels[j][0]), int(pixels[j][1])]
                output['point_to_pixel'][str(tactile_idx)] = pixel
                pixel_key = f"{pixel[0]},{pixel[1]}"
                if pixel_key not in output['pixel_to_points']:
                    output['pixel_to_points'][pixel_key] = []
                output['pixel_to_points'][pixel_key].append(tactile_idx)

            # Assign remaining tactile points to nearest pixel
            assigned = set(col_ind)
            for i in range(n_tactile):
                if i not in assigned:
                    # Find nearest pixel
                    distances = [np.linalg.norm(tactile_2d_norm[i] - pixels_norm[j]) for j in range(n_pixels)]
                    nearest_j = int(np.argmin(distances))
                    tactile_idx = int(tactile_indices[i])
                    pixel = [int(pixels[nearest_j][0]), int(pixels[nearest_j][1])]
                    output['point_to_pixel'][str(tactile_idx)] = pixel
                    pixel_key = f"{pixel[0]},{pixel[1]}"
                    if pixel_key not in output['pixel_to_points']:
                        output['pixel_to_points'][pixel_key] = []
                    output['pixel_to_points'][pixel_key].append(tactile_idx)

    def update_display(self):
        """Update the display with current selections and mappings."""
        # Update tactile scatter colors
        colors = []
        for i in range(len(self.tactile_points)):
            # Check if part of existing mapping
            mapping_color = None
            for mapping in self.mappings:
                if i in mapping['tactile_indices']:
                    mapping_color = mapping['color']
                    break

            if i in self.selected_tactile_indices:
                colors.append('red')  # Currently selected
            elif mapping_color:
                colors.append(mapping_color)  # Part of mapping
            else:
                colors.append('lightgray')  # Unselected

        self.tactile_scatter.set_facecolors(colors)

        # Update pixel grid colors
        self.pixel_colors = np.ones((self.image_shape[0], self.image_shape[1], 3)) * 0.9

        # Color pixels from existing mappings
        for mapping in self.mappings:
            color_rgb = mcolors.to_rgb(mapping['color'])
            for row, col in mapping['pixels']:
                self.pixel_colors[row, col] = color_rgb

        # Color currently selected pixels (red)
        for row, col in self.selected_pixels:
            self.pixel_colors[row, col] = [1.0, 0.0, 0.0]

        self.pixel_image.set_data(self.pixel_colors)

        # Update status
        self.status_text.set_text(
            f"Selected: {len(self.selected_tactile_indices)} tactile points, {len(self.selected_pixels)} pixels | "
            f"Mappings: {len(self.mappings)} | "
            f"Keys: 'c'=clear, 'm'=create mapping, 's'=save, 'u'=undo, 'q'=quit"
        )

        self.fig.canvas.draw_idle()

    def run(self):
        """Run the interactive tool."""
        self.update_display()
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Interactive Tactile Mapping Tool")
    parser.add_argument("--tactile-grid", type=str,
                        default="examples/tactile/merged_tactile_grid.json",
                        help="Path to tactile grid JSON file")
    parser.add_argument("--image-rows", type=int, default=24,
                        help="Number of rows in tactile image")
    parser.add_argument("--image-cols", type=int, default=32,
                        help="Number of columns in tactile image")
    parser.add_argument("--output", type=str, default="tactile_to_image_mapping.json",
                        help="Output path for mapping file")
    args = parser.parse_args()

    # Load tactile grid
    tactile_grid_path = Path(args.tactile_grid)
    if not tactile_grid_path.exists():
        print(f"Error: Tactile grid file not found: {tactile_grid_path}")
        return

    print(f"Loading tactile grid from: {tactile_grid_path}")
    with open(tactile_grid_path, 'r') as f:
        tactile_data = json.load(f)

    # Create and run tool
    tool = TactileMappingTool(
        tactile_data,
        image_shape=(args.image_rows, args.image_cols)
    )
    tool.run()


if __name__ == "__main__":
    main()
