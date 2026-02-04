"""
Visualize Tactile Point to Pixel Mapping

This script visualizes the computed mapping between tactile points and pixels:
1. Left panel: Tactile points colored by their assigned pixel region
2. Right panel: 24x32 pixel grid colored by link assignment
3. Interactive: Click on a tactile point to highlight its pixel, or vice versa

Usage:
    python visualize_tactile_mapping.py --mapping tactile_pixel_mapping.json

Interactive Controls:
    - Left click on tactile point: Highlight corresponding pixel(s)
    - Left click on pixel: Highlight corresponding tactile point(s)
    - Right click: Clear highlight
    - 'c' key: Clear all highlights
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
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

# Color scheme for different fingers
FINGER_COLORS = {
    "finger1": "#e41a1c",  # Red - Thumb
    "finger2": "#377eb8",  # Blue - Index
    "finger3": "#4daf4a",  # Green - Middle
    "finger4": "#984ea3",  # Purple - Ring
    "finger5": "#ff7f00",  # Orange - Pinky
    "palm": "#a65628",     # Brown - Palm
}


def get_link_color(link_name):
    """Get color for a link based on which finger it belongs to."""
    for finger, color in FINGER_COLORS.items():
        if finger in link_name:
            return color
    return "#999999"  # Gray for unknown


class InteractiveMappingVisualizer:
    """Interactive visualizer for tactile-to-pixel mapping."""

    def __init__(self, mapping_data, tactile_data, save_path=None):
        self.mapping_data = mapping_data
        self.tactile_data = tactile_data
        self.save_path = save_path

        # Parse data
        self.image_shape = tuple(mapping_data['image_shape'])
        self.point_to_pixel = mapping_data['point_to_pixel']
        self.pixel_to_points = mapping_data.get('pixel_to_points', {})

        # Build tactile points list
        self.tactile_points = []
        self.parse_tactile_points()

        # Highlight state
        self.highlighted_tactile = set()
        self.highlighted_pixels = set()

        # Setup figure
        self.setup_figure()

    def parse_tactile_points(self):
        """Parse tactile points and compute 2D positions."""
        links_data = self.tactile_data.get('links', {})

        for link_name, link_data in links_data.items():
            points = link_data.get('points', [])
            yz_offset = LINK_YZ_OFFSETS.get(link_name, (0.0, 0.0))

            for point_idx, point in enumerate(points):
                local_pos = np.array(point['local'] if isinstance(point, dict) else point)

                if link_name == "finger1_link2":
                    offset_pos_2d = np.array([local_pos[0] + yz_offset[0],
                                               local_pos[2] + yz_offset[1]])
                else:
                    offset_pos_2d = np.array([local_pos[1] + yz_offset[0],
                                               local_pos[2] + yz_offset[1]])

                self.tactile_points.append({
                    'link_name': link_name,
                    'point_idx': point_idx,
                    'offset_pos_2d': offset_pos_2d,
                })

    def setup_figure(self):
        """Setup the matplotlib figure."""
        self.fig, (self.ax_tactile, self.ax_pixel) = plt.subplots(1, 2, figsize=(18, 10))

        # Draw initial state
        self.draw_tactile_panel()
        self.draw_pixel_panel()

        # Statistics
        num_mapped = sum(1 for i in range(len(self.tactile_points)) if str(i) in self.point_to_pixel)
        num_unmapped = len(self.tactile_points) - num_mapped

        stats_text = f"Tactile: {num_mapped} mapped | {num_unmapped} unmapped | "
        stats_text += f"Pixels: {len(self.pixel_to_points)}/{self.image_shape[0]*self.image_shape[1]} used\n"
        stats_text += "Click tactile point or pixel to highlight correspondence. Right-click or 'c' to clear."

        self.fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10,
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Connect events
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.08)

    def draw_tactile_panel(self):
        """Draw the tactile points panel."""
        self.ax_tactile.clear()
        self.ax_tactile.set_title("Tactile Points (click to highlight)", fontsize=14)
        self.ax_tactile.set_xlabel("Y (m)")
        self.ax_tactile.set_ylabel("Z (m)")
        self.ax_tactile.set_aspect('equal')

        # Separate points by state
        for global_idx, point in enumerate(self.tactile_points):
            link_name = point['link_name']
            pos = point['offset_pos_2d']
            base_color = get_link_color(link_name)
            is_mapped = str(global_idx) in self.point_to_pixel
            is_highlighted = global_idx in self.highlighted_tactile

            if is_highlighted:
                # Highlighted: large yellow marker with black edge
                self.ax_tactile.scatter(pos[0], pos[1], c='yellow', s=150,
                                        edgecolors='black', linewidths=2, zorder=10)
                # Also show the point index
                self.ax_tactile.annotate(str(global_idx), (pos[0], pos[1]),
                                         fontsize=8, ha='center', va='bottom',
                                         xytext=(0, 8), textcoords='offset points', zorder=11)
            elif is_mapped:
                # Mapped: filled circle
                self.ax_tactile.scatter(pos[0], pos[1], c=base_color, s=40,
                                        edgecolors='black', linewidths=0.5, zorder=2)
            else:
                # Unmapped: X marker
                self.ax_tactile.scatter(pos[0], pos[1], c='white', s=60,
                                        edgecolors=base_color, linewidths=2, marker='o', zorder=1)
                self.ax_tactile.scatter(pos[0], pos[1], c=base_color, s=30,
                                        marker='x', linewidths=1.5, zorder=3)

        # Legend
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        legend_elements = []
        for finger, color in FINGER_COLORS.items():
            label = {"finger1": "Thumb", "finger2": "Index", "finger3": "Middle",
                     "finger4": "Ring", "finger5": "Pinky", "palm": "Palm"}.get(finger, finger)
            legend_elements.append(Patch(facecolor=color, edgecolor='black', label=label))
        legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow',
                                       markersize=10, markeredgecolor='black', markeredgewidth=2,
                                       label='Highlighted'))
        self.ax_tactile.legend(handles=legend_elements, loc='upper left', fontsize=8)

    def draw_pixel_panel(self):
        """Draw the pixel grid panel."""
        self.ax_pixel.clear()
        self.ax_pixel.set_title("24x32 Tactile Image (click to highlight)", fontsize=14)
        self.ax_pixel.set_xlabel("Column")
        self.ax_pixel.set_ylabel("Row")

        # Create pixel color array
        pixel_colors = np.ones((self.image_shape[0], self.image_shape[1], 3)) * 0.95

        # Color by link
        for pixel_key, point_indices in self.pixel_to_points.items():
            row, col = map(int, pixel_key.split(','))
            if point_indices:
                link_name = self.tactile_points[point_indices[0]]['link_name']
                color_rgb = mcolors.to_rgb(get_link_color(link_name))
                pixel_colors[row, col] = color_rgb

        # Highlight selected pixels
        for row, col in self.highlighted_pixels:
            pixel_colors[row, col] = [1.0, 1.0, 0.0]  # Yellow

        self.ax_pixel.imshow(pixel_colors, origin='upper', aspect='equal',
                             extent=[-0.5, self.image_shape[1]-0.5, self.image_shape[0]-0.5, -0.5])

        # Grid lines
        for i in range(self.image_shape[0] + 1):
            self.ax_pixel.axhline(i - 0.5, color='gray', linewidth=0.3, alpha=0.5)
        for j in range(self.image_shape[1] + 1):
            self.ax_pixel.axvline(j - 0.5, color='gray', linewidth=0.3, alpha=0.5)

        # Annotate highlighted pixels with point indices
        for row, col in self.highlighted_pixels:
            pixel_key = f"{row},{col}"
            if pixel_key in self.pixel_to_points:
                indices = self.pixel_to_points[pixel_key]
                label = ','.join(str(i) for i in indices[:3])  # Show up to 3
                if len(indices) > 3:
                    label += '...'
                self.ax_pixel.annotate(label, (col, row), fontsize=7, ha='center', va='center',
                                       color='black', fontweight='bold')

        self.ax_pixel.set_xlim(-0.5, self.image_shape[1] - 0.5)
        self.ax_pixel.set_ylim(self.image_shape[0] - 0.5, -0.5)
        self.ax_pixel.set_xticks(range(0, self.image_shape[1], 4))
        self.ax_pixel.set_yticks(range(0, self.image_shape[0], 4))

    def on_click(self, event):
        """Handle mouse click."""
        if event.button == 3:  # Right click - clear
            self.clear_highlights()
            return

        if event.button != 1:
            return

        if event.inaxes == self.ax_tactile:
            self.handle_tactile_click(event)
        elif event.inaxes == self.ax_pixel:
            self.handle_pixel_click(event)

    def handle_tactile_click(self, event):
        """Handle click on tactile panel."""
        if event.xdata is None or event.ydata is None:
            return

        # Find closest tactile point
        positions = np.array([p['offset_pos_2d'] for p in self.tactile_points])
        distances = np.sqrt((positions[:, 0] - event.xdata)**2 +
                           (positions[:, 1] - event.ydata)**2)
        closest_idx = int(np.argmin(distances))

        if distances[closest_idx] < 0.015:  # Threshold
            self.highlight_tactile_point(closest_idx)

    def handle_pixel_click(self, event):
        """Handle click on pixel panel."""
        if event.xdata is None or event.ydata is None:
            return

        col = int(np.round(event.xdata))
        row = int(np.round(event.ydata))

        if 0 <= row < self.image_shape[0] and 0 <= col < self.image_shape[1]:
            self.highlight_pixel(row, col)

    def highlight_tactile_point(self, idx):
        """Highlight a tactile point and its corresponding pixel(s)."""
        self.highlighted_tactile.clear()
        self.highlighted_pixels.clear()

        self.highlighted_tactile.add(idx)

        # Find corresponding pixel(s)
        if str(idx) in self.point_to_pixel:
            pixel = self.point_to_pixel[str(idx)]
            self.highlighted_pixels.add((pixel[0], pixel[1]))

        point = self.tactile_points[idx]
        print(f"Tactile point {idx} ({point['link_name']}[{point['point_idx']}]) -> Pixel {list(self.highlighted_pixels)}")

        self.redraw()

    def highlight_pixel(self, row, col):
        """Highlight a pixel and its corresponding tactile point(s)."""
        self.highlighted_tactile.clear()
        self.highlighted_pixels.clear()

        self.highlighted_pixels.add((row, col))

        # Find corresponding tactile points
        pixel_key = f"{row},{col}"
        if pixel_key in self.pixel_to_points:
            for idx in self.pixel_to_points[pixel_key]:
                self.highlighted_tactile.add(idx)

        print(f"Pixel ({row}, {col}) -> Tactile points {list(self.highlighted_tactile)}")

        self.redraw()

    def clear_highlights(self):
        """Clear all highlights."""
        self.highlighted_tactile.clear()
        self.highlighted_pixels.clear()
        print("Highlights cleared")
        self.redraw()

    def on_key(self, event):
        """Handle key press."""
        if event.key == 'c':
            self.clear_highlights()

    def redraw(self):
        """Redraw both panels."""
        self.draw_tactile_panel()
        self.draw_pixel_panel()
        self.fig.canvas.draw_idle()

    def run(self):
        """Run the visualizer."""
        if self.save_path:
            plt.savefig(self.save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to: {self.save_path}")
        else:
            plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize Tactile Mapping")
    parser.add_argument("--mapping", type=str,
                        default="tactile_pixel_mapping.json",
                        help="Path to computed mapping JSON")
    parser.add_argument("--tactile-grid", type=str,
                        default="examples/tactile/full_hand_tactile.json",
                        help="Path to original tactile grid JSON (for positions)")
    parser.add_argument("--save", type=str, default=None,
                        help="Save figure to file instead of displaying")
    args = parser.parse_args()

    # Load mapping
    mapping_path = Path(args.mapping)
    if not mapping_path.exists():
        print(f"Error: Mapping file not found: {mapping_path}")
        return

    print(f"Loading mapping from: {mapping_path}")
    with open(mapping_path, 'r') as f:
        mapping_data = json.load(f)

    # Load tactile grid for positions
    tactile_grid_path = Path(args.tactile_grid)
    if not tactile_grid_path.exists():
        print(f"Error: Tactile grid file not found: {tactile_grid_path}")
        return

    print(f"Loading tactile grid from: {tactile_grid_path}")
    with open(tactile_grid_path, 'r') as f:
        tactile_data = json.load(f)

    print(f"Image shape: {mapping_data['image_shape']}")
    print(f"Mapped points: {len(mapping_data['point_to_pixel'])}")

    # Create and run interactive visualizer
    visualizer = InteractiveMappingVisualizer(mapping_data, tactile_data, args.save)
    visualizer.run()

    # Print detailed statistics
    image_shape = tuple(mapping_data['image_shape'])
    point_to_pixel = mapping_data['point_to_pixel']
    pixel_to_points = mapping_data.get('pixel_to_points', {})
    per_link_mapping = mapping_data.get('per_link_mapping', {})

    print(f"\n{'='*70}")
    print("MAPPING STATISTICS")
    print(f"{'='*70}")
    print(f"Image shape: {image_shape[0]} x {image_shape[1]} = {image_shape[0]*image_shape[1]} pixels")
    print(f"Total tactile points: {mapping_data.get('num_tactile_points', 'N/A')}")
    print(f"Mapped points: {len(point_to_pixel)}")
    print(f"Pixels used: {len(pixel_to_points)}")
    print(f"Average points per pixel: {len(point_to_pixel) / max(1, len(pixel_to_points)):.2f}")

    print(f"\nPer-link breakdown:")
    for link_name in sorted(per_link_mapping.keys()):
        count = len(per_link_mapping[link_name])
        print(f"  {link_name:25s}: {count:3d} points")

    print(f"\nPixel coverage: {len(pixel_to_points)}/{image_shape[0]*image_shape[1]} ({100*len(pixel_to_points)/(image_shape[0]*image_shape[1]):.1f}%)")


if __name__ == "__main__":
    main()
