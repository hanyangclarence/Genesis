"""
Interactive Tool for Creating Tactile Point to Image Mapping

This tool allows you to:
1. View all tactile points flattened to a 2D map (the layout in tactile_layout.py)
2. View a 24x32 pixel grid (tactile image)
3. Select regions on both maps by clicking or dragging
4. Create mappings between selected tactile points and pixels

It saves the raw mapping: groups of tactile points and the pixels each group should fill.
compute_tactile_mapping.py turns it into the final point-to-pixel mapping.

Usage:
    python examples/tactile/create_tactile_mapping.py \
        --tactile-grid full_hand_tactile.json \
        --output tactile_to_image_mapping.json

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
import matplotlib.colors as mcolors

from tactile_layout import parse_tactile_points

# Colors for different mapping groups
COLORS = list(mcolors.TABLEAU_COLORS.values())


class TactileMappingTool:
    def __init__(self, tactile_data, output_path, image_shape=(24, 32)):
        self.image_shape = image_shape
        self.output_path = output_path

        # List of dicts with link_name, point_idx, local_pos, offset_pos_2d, collapse_axis
        self.tactile_points = parse_tactile_points(tactile_data)
        print(f"Loaded {len(self.tactile_points)} tactile points from {len(tactile_data.get('links', {}))} links")

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

    def setup_figure(self):
        """Setup the matplotlib figure with two panels."""
        self.fig, (self.ax_tactile, self.ax_pixel) = plt.subplots(1, 2, figsize=(16, 8))

        # Left panel: Tactile points (per-link local 2D + layout offsets)
        self.ax_tactile.set_title("Tactile Points (Left drag: select, Right drag: deselect)")
        self.ax_tactile.set_xlabel("Horizontal (m)")
        self.ax_tactile.set_ylabel("Vertical (m)")
        self.ax_tactile.set_aspect('equal')

        # Plot all tactile points
        positions = np.array([p['offset_pos_2d'] for p in self.tactile_points])
        self.tactile_scatter = self.ax_tactile.scatter(
            positions[:, 0], positions[:, 1],
            c='lightgray', s=30, edgecolors='black', linewidths=0.5,
            picker=True, pickradius=5
        )
        # Flip so Y+ points left and Z+ points down (hand points down visually)
        self.ax_tactile.invert_xaxis()
        self.ax_tactile.invert_yaxis()

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

    def save_mappings(self):
        """Save mappings to JSON file."""
        if not self.mappings:
            print("No mappings to save!")
            return

        # Build output structure
        output = {
            'image_shape': list(self.image_shape),
            'num_tactile_points': len(self.tactile_points),
            'num_mappings': len(self.mappings),
            'mappings': [],
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

        with open(self.output_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\nMappings saved to: {self.output_path}")
        print(f"  Total mappings: {len(self.mappings)}")

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
                        default="examples/tactile/full_hand_tactile_left_v5.json",
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
        output_path=args.output,
        image_shape=(args.image_rows, args.image_cols),
    )
    tool.run()


if __name__ == "__main__":
    main()
