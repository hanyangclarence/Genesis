"""Interactive 3D tactile-point ↔ pixel-map visualizer.

Loads any tactile grid JSON + pixel mapping JSON + URDF, transforms each
tactile point from its link-local frame to world coordinates at the hand's
built-in rest configuration, and shows two linked panels:

    Left panel:  3D scatter of tactile points in world frame.
    Right panel: 2D pixel grid of the tactile image.

Controls
--------
    Left click on a tactile point      -> highlight the pixel it maps to
    Left click on a pixel              -> highlight the tactile point(s)
                                          that map to it
    Right click (either panel) or 'c'  -> clear highlights
    'q'                                -> quit

Because the point positions come from the URDF's own forward kinematics,
this script works uniformly for the right and left hand with no hard-coded
per-link offsets.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import genesis as gs
from genesis.utils import geom as gu


FINGER_COLORS = {
    "finger1": "#e41a1c",  # Thumb
    "finger2": "#377eb8",  # Index
    "finger3": "#4daf4a",  # Middle
    "finger4": "#984ea3",  # Ring
    "finger5": "#ff7f00",  # Pinky
    "palm":    "#a65628",  # Palm
}


def link_color(link_name):
    for key, c in FINGER_COLORS.items():
        if key in link_name:
            return c
    return "#999999"


def compute_world_points(urdf_path, tactile_data):
    """Return (points_world, meta) where points_world is (N, 3) and meta is
    a list of {link_name, point_idx} dicts in the same global-index order
    as the tactile grid JSON."""
    gs.init(backend=gs.cpu, logging_level="warning")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    hand = scene.add_entity(
        gs.morphs.URDF(
            file=urdf_path,
            merge_fixed_links=False,
            fixed=True,
            pos=(0.0, 0.0, 0.0),
            euler=(0.0, 0.0, 0.0),
        ),
    )
    scene.build()

    points_world = []
    meta = []
    for link_name, link_data in tactile_data["links"].items():
        points = link_data.get("points", [])
        if not points:
            continue
        link = hand.get_link(link_name)
        link_pos = torch.tensor(link.get_pos().cpu().numpy(), dtype=gs.tc_float)
        link_quat = torch.tensor(link.get_quat().cpu().numpy(), dtype=gs.tc_float)
        local = torch.tensor(np.asarray(points, dtype=np.float32), dtype=gs.tc_float)
        world = gu.transform_by_trans_quat(local, link_pos, link_quat).cpu().numpy()
        for i, w in enumerate(world):
            points_world.append(w)
            meta.append({"link_name": link_name, "point_idx": i})

    return np.asarray(points_world, dtype=np.float64), meta


class Interactive3DMappingViewer:
    def __init__(self, points_world, meta, mapping):
        self.points_world = points_world
        self.meta = meta
        self.image_shape = tuple(mapping["image_shape"])
        self.point_to_pixel = {int(k): tuple(v) for k, v in mapping["point_to_pixel"].items()}
        self.pixel_to_points = {
            tuple(int(x) for x in k.split(",")): list(v)
            for k, v in mapping.get("pixel_to_points", {}).items()
        }

        self.highlighted_points = set()
        self.highlighted_pixels = set()

        # Figure layout: 3D on the left, 2D pixel map on the right.
        self.fig = plt.figure(figsize=(16, 8))
        self.ax3d = self.fig.add_subplot(1, 2, 1, projection="3d")
        self.ax2d = self.fig.add_subplot(1, 2, 2)

        self._draw_3d()
        self._draw_2d()

        num_mapped = sum(1 for i in range(len(self.meta)) if i in self.point_to_pixel)
        stats = (
            f"{len(self.meta)} points ({num_mapped} mapped) | "
            f"Image {self.image_shape[0]}x{self.image_shape[1]} | "
            f"Click a 3D point or pixel to highlight correspondences. "
            f"Right-click or 'c' to clear."
        )
        self.fig.text(0.5, 0.02, stats, ha="center", fontsize=9,
                      bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        self.fig.canvas.mpl_connect("pick_event", self._on_pick)
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.08)

    def _draw_3d(self):
        ax = self.ax3d
        ax.clear()
        colors = np.array([mcolors.to_rgb(link_color(m["link_name"])) for m in self.meta])

        base_sizes = np.full(len(self.meta), 15.0)
        edge_colors = np.array([[0, 0, 0, 0.25]] * len(self.meta))

        plot_colors = colors.copy()
        for i in self.highlighted_points:
            plot_colors[i] = (1.0, 1.0, 0.0)
            base_sizes[i] = 80.0
            edge_colors[i] = (0, 0, 0, 1)

        self._scatter = ax.scatter(
            self.points_world[:, 0],
            self.points_world[:, 1],
            self.points_world[:, 2],
            c=plot_colors, s=base_sizes, edgecolors=edge_colors, linewidths=0.6,
            picker=5,
            depthshade=False,
        )

        # Equal aspect via bounding cube
        mins = self.points_world.min(axis=0)
        maxs = self.points_world.max(axis=0)
        center = 0.5 * (mins + maxs)
        radius = 0.5 * (maxs - mins).max() * 1.1
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title("Tactile points (world frame, rest pose)")

        legend_elements = [
            Patch(facecolor=c, edgecolor="black",
                  label={"finger1": "Thumb", "finger2": "Index",
                         "finger3": "Middle", "finger4": "Ring",
                         "finger5": "Pinky", "palm": "Palm"}[k])
            for k, c in FINGER_COLORS.items()
        ]
        legend_elements.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor="yellow",
                   markersize=10, markeredgecolor="black", label="Highlighted")
        )
        ax.legend(handles=legend_elements, loc="upper left", fontsize=8)

    def _draw_2d(self):
        ax = self.ax2d
        ax.clear()
        H, W = self.image_shape
        pixel_colors = np.ones((H, W, 3)) * 0.95

        # Color by link of the first point assigned to each pixel
        for (row, col), pts in self.pixel_to_points.items():
            if pts:
                c = mcolors.to_rgb(link_color(self.meta[pts[0]]["link_name"]))
                pixel_colors[row, col] = c

        # Highlights override
        for row, col in self.highlighted_pixels:
            pixel_colors[row, col] = (1.0, 1.0, 0.0)

        ax.imshow(pixel_colors, origin="upper", aspect="equal",
                  extent=[-0.5, W - 0.5, H - 0.5, -0.5])
        for i in range(H + 1):
            ax.axhline(i - 0.5, color="gray", linewidth=0.3, alpha=0.5)
        for j in range(W + 1):
            ax.axvline(j - 0.5, color="gray", linewidth=0.3, alpha=0.5)

        for row, col in self.highlighted_pixels:
            pts = self.pixel_to_points.get((row, col), [])
            if pts:
                label = ",".join(str(p) for p in pts[:3])
                if len(pts) > 3:
                    label += "…"
                ax.annotate(label, (col, row), ha="center", va="center",
                            fontsize=7, fontweight="bold")

        ax.set_xlim(-0.5, W - 0.5)
        ax.set_ylim(H - 0.5, -0.5)
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        ax.set_title(f"Tactile image ({H}x{W})")
        ax.set_xticks(range(0, W, 4))
        ax.set_yticks(range(0, H, 4))

    def _on_pick(self, event):
        if event.artist is not self._scatter:
            return
        if len(event.ind) == 0:
            return
        # Closest picked index (matplotlib may return several)
        idx = int(event.ind[0])
        self.highlighted_points = {idx}
        if idx in self.point_to_pixel:
            self.highlighted_pixels = {self.point_to_pixel[idx]}
        else:
            self.highlighted_pixels = set()
        m = self.meta[idx]
        print(f"Point {idx} ({m['link_name']}[{m['point_idx']}]) -> "
              f"pixel {list(self.highlighted_pixels)}")
        self._redraw()

    def _on_click(self, event):
        if event.button == 3:  # right click clears
            self._clear()
            return
        if event.button != 1:
            return
        if event.inaxes is self.ax2d:
            if event.xdata is None or event.ydata is None:
                return
            col = int(round(event.xdata))
            row = int(round(event.ydata))
            H, W = self.image_shape
            if not (0 <= row < H and 0 <= col < W):
                return
            self.highlighted_pixels = {(row, col)}
            self.highlighted_points = set(self.pixel_to_points.get((row, col), []))
            print(f"Pixel ({row}, {col}) -> points {sorted(self.highlighted_points)}")
            self._redraw()
        # Clicks on ax3d are handled by the picker above.

    def _on_key(self, event):
        if event.key == "c":
            self._clear()
        elif event.key == "q":
            plt.close(self.fig)

    def _clear(self):
        self.highlighted_points.clear()
        self.highlighted_pixels.clear()
        print("Cleared highlights")
        self._redraw()

    def _redraw(self):
        # Preserve the 3D camera view when redrawing.
        elev, azim = self.ax3d.elev, self.ax3d.azim
        xlim = self.ax3d.get_xlim()
        ylim = self.ax3d.get_ylim()
        zlim = self.ax3d.get_zlim()
        self._draw_3d()
        self.ax3d.view_init(elev=elev, azim=azim)
        self.ax3d.set_xlim(xlim)
        self.ax3d.set_ylim(ylim)
        self.ax3d.set_zlim(zlim)
        self._draw_2d()
        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Interactive 3D tactile-point / pixel-map viewer")
    parser.add_argument("--tactile-grid", type=str, required=True,
                        help="Path to tactile grid JSON (e.g. full_hand_tactile_left.json)")
    parser.add_argument("--mapping", type=str, required=True,
                        help="Path to pixel mapping JSON (e.g. tactile_pixel_mapping_left.json)")
    parser.add_argument("--urdf", type=str, required=True,
                        help="Path to the URDF that the tactile grid was generated for")
    args = parser.parse_args()

    tactile_grid_path = Path(args.tactile_grid)
    mapping_path = Path(args.mapping)
    urdf_path = Path(args.urdf)
    for p in (tactile_grid_path, mapping_path, urdf_path):
        if not p.exists():
            raise FileNotFoundError(p)

    with open(tactile_grid_path) as f:
        tactile_data = json.load(f)
    with open(mapping_path) as f:
        mapping = json.load(f)

    print(f"URDF         : {urdf_path}")
    print(f"Tactile grid : {tactile_grid_path}")
    print(f"Mapping      : {mapping_path}")
    print("Computing world positions via Genesis forward kinematics...")
    points_world, meta = compute_world_points(str(urdf_path), tactile_data)
    print(f"Got {len(points_world)} world-space tactile points")

    viewer = Interactive3DMappingViewer(points_world, meta, mapping)
    viewer.show()


if __name__ == "__main__":
    main()
