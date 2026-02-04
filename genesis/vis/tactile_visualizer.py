"""
Tactile Visualizer for converting tactile sensor readouts to 2D tactile maps.

This module provides functionality to:
1. Load tactile point configuration and pixel mapping
2. Convert tactile sensor readouts (B, N) to 2D tactile maps (B, H, W)
3. Optionally display and update the tactile map in a viewer

Usage:
    from genesis.vis import TactileVisualizer

    # Create visualizer for batched operation
    vis = TactileVisualizer(
        tactile_grid_path="path/to/tactile_grid.json",
        pixel_mapping_path="path/to/pixel_mapping.json",
        num_envs=16,
        show_viewer=True,
    )

    # Convert tactile readout to 2D map (batched)
    tactile_map = vis.to_map(force_magnitudes)  # (B, N) -> (B, H, W)

    # Update viewer (if show_viewer=True, displays env 0)
    vis.update(force_magnitudes)
"""

import json
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch


class TactileVisualizer:
    """
    Converts tactile sensor readouts to 2D tactile maps with batch support.

    The visualizer loads precomputed mappings between tactile points and image pixels,
    then uses efficient tensor operations to aggregate force values into 2D images.

    Supports batched operations for multiple parallel environments.

    Attributes:
        image_shape: Tuple of (height, width) for the output tactile map.
        num_tactile_points: Total number of tactile points.
        num_envs: Number of parallel environments (batch size).
        device: PyTorch device for tensor operations.
    """

    def __init__(
        self,
        tactile_grid_path: Union[str, Path],
        pixel_mapping_path: Union[str, Path],
        num_envs: int = 1,
        show_viewer: bool = False,
        device: Optional[torch.device] = None,
        cmap: str = "hot",
        vmax: Optional[float] = None,
    ):
        """
        Initialize the tactile visualizer.

        Args:
            tactile_grid_path: Path to tactile grid JSON file.
            pixel_mapping_path: Path to pixel mapping JSON file.
            num_envs: Number of parallel environments (batch size).
            show_viewer: Whether to show interactive matplotlib viewer.
            device: PyTorch device. Defaults to CUDA if available.
            cmap: Colormap for visualization (default: "hot").
            vmax: Maximum value for colormap scaling. If None, auto-scales.
        """
        self._tactile_grid_path = Path(tactile_grid_path)
        self._pixel_mapping_path = Path(pixel_mapping_path)
        self._num_envs = num_envs
        self._show_viewer = show_viewer
        self._cmap = cmap
        self._vmax = vmax

        # Set device
        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = device

        # Load configurations
        self._load_tactile_grid()
        self._load_pixel_mapping()
        self._build_aggregation_tensors()

        # Initialize viewer if requested
        self._fig = None
        self._ax = None
        self._img_plot = None
        if show_viewer:
            self._init_viewer()

    def _load_tactile_grid(self):
        """Load tactile grid configuration."""
        if not self._tactile_grid_path.exists():
            raise FileNotFoundError(f"Tactile grid file not found: {self._tactile_grid_path}")

        with open(self._tactile_grid_path, "r") as f:
            self._tactile_data = json.load(f)

        self._links_data = self._tactile_data.get("links", {})

        # Compute global index offset for each link (based on JSON order)
        self._link_global_offset = {}
        offset = 0
        for link_name in self._links_data.keys():
            self._link_global_offset[link_name] = offset
            offset += self._links_data[link_name]["num_points"]

        self._num_tactile_points = offset

    def _load_pixel_mapping(self):
        """Load pixel mapping configuration."""
        if not self._pixel_mapping_path.exists():
            raise FileNotFoundError(f"Pixel mapping file not found: {self._pixel_mapping_path}")

        with open(self._pixel_mapping_path, "r") as f:
            self._pixel_mapping = json.load(f)

        self._image_shape = tuple(self._pixel_mapping["image_shape"])
        self._pixel_to_points = self._pixel_mapping["pixel_to_points"]

        # Validate consistency
        mapping_num_points = self._pixel_mapping.get("num_tactile_points", self._num_tactile_points)
        if mapping_num_points != self._num_tactile_points:
            raise ValueError(
                f"Tactile point count mismatch: grid has {self._num_tactile_points}, "
                f"mapping expects {mapping_num_points}"
            )

    def _build_aggregation_tensors(self):
        """Build tensors for efficient pixel aggregation using scatter_add."""
        pixel_row_list = []
        pixel_col_list = []
        point_idx_list = []

        for pixel_key, point_indices in self._pixel_to_points.items():
            row, col = map(int, pixel_key.split(","))
            for global_idx in point_indices:
                pixel_row_list.append(row)
                pixel_col_list.append(col)
                point_idx_list.append(global_idx)

        # Convert to tensors
        self._pixel_rows = torch.tensor(pixel_row_list, dtype=torch.long, device=self._device)
        self._pixel_cols = torch.tensor(pixel_col_list, dtype=torch.long, device=self._device)
        self._point_indices = torch.tensor(point_idx_list, dtype=torch.long, device=self._device)

        # Compute flat pixel indices for scatter_add (row * num_cols + col)
        self._pixel_flat_indices = self._pixel_rows * self._image_shape[1] + self._pixel_cols

        # Count how many points map to each pixel (for averaging)
        self._pixel_counts = torch.zeros(
            self._image_shape[0] * self._image_shape[1], dtype=torch.float32, device=self._device
        )
        self._pixel_counts.scatter_add_(
            0, self._pixel_flat_indices, torch.ones_like(self._pixel_flat_indices, dtype=torch.float32)
        )

        # Store count before clamping for info
        self._num_pixels_with_points = int((self._pixel_counts > 0).sum().item())

        # Clamp to avoid division by zero
        self._pixel_counts = self._pixel_counts.clamp(min=1.0)

    def _init_viewer(self):
        """Initialize matplotlib viewer for interactive display."""
        import matplotlib.pyplot as plt

        plt.ion()
        self._fig, self._ax = plt.subplots(1, 1, figsize=(10, 8))

        self._ax.set_title("Tactile Map (env 0)")
        self._ax.set_xlabel("Column")
        self._ax.set_ylabel("Row")

        # Initialize with zeros
        initial_image = np.zeros(self._image_shape)
        vmax = self._vmax if self._vmax is not None else 10.0
        self._img_plot = self._ax.imshow(
            initial_image, cmap=self._cmap, vmin=0, vmax=vmax, aspect="equal", origin="upper"
        )

        self._colorbar = plt.colorbar(self._img_plot, ax=self._ax, label="Force (N)")
        plt.tight_layout()

    def to_map(self, force_magnitudes: torch.Tensor) -> torch.Tensor:
        """
        Convert tactile force magnitudes to 2D tactile maps (batched).

        Args:
            force_magnitudes: Tensor of shape (B, N) containing force magnitudes
                for each tactile point, where B = num_envs and N = num_tactile_points.

        Returns:
            Tensor of shape (B, H, W) containing the aggregated tactile maps.
        """
        if force_magnitudes.dim() != 2:
            raise ValueError(
                f"Expected 2D tensor (B, N), got {force_magnitudes.dim()}D tensor"
            )

        batch_size, num_points = force_magnitudes.shape

        if batch_size != self._num_envs:
            raise ValueError(
                f"Expected batch size {self._num_envs}, got {batch_size}"
            )

        if num_points != self._num_tactile_points:
            raise ValueError(
                f"Expected {self._num_tactile_points} force values, got {num_points}"
            )

        # Ensure tensor is on correct device
        if force_magnitudes.device != self._device:
            force_magnitudes = force_magnitudes.to(self._device)

        # Gather forces at mapped points: (B, N) -> (B, M) where M = num_mappings
        mapped_forces = force_magnitudes[:, self._point_indices]  # (B, M)

        # Expand pixel indices for batched scatter_add
        num_pixels = self._image_shape[0] * self._image_shape[1]
        pixel_flat_indices_expanded = self._pixel_flat_indices.unsqueeze(0).expand(batch_size, -1)  # (B, M)

        # Sum forces into pixels using scatter_add along dim 1
        pixel_force_sum = torch.zeros(batch_size, num_pixels, dtype=torch.float32, device=self._device)
        pixel_force_sum.scatter_add_(1, pixel_flat_indices_expanded, mapped_forces.float())  # (B, H*W)

        # Average by count (broadcast pixel_counts to (1, H*W))
        pixel_force_avg = pixel_force_sum / self._pixel_counts.unsqueeze(0)  # (B, H*W)

        # Reshape to (B, H, W)
        return pixel_force_avg.reshape(batch_size, self._image_shape[0], self._image_shape[1])

    def update(self, force_magnitudes: torch.Tensor) -> torch.Tensor:
        """
        Convert forces to tactile maps and update viewer if enabled.

        Args:
            force_magnitudes: Tensor of shape (B, N) containing force magnitudes.

        Returns:
            Tensor of shape (B, H, W) containing the tactile maps.
        """
        tactile_maps = self.to_map(force_magnitudes)

        if self._show_viewer and self._img_plot is not None:
            import matplotlib.pyplot as plt

            # Display env 0
            tactile_map_np = tactile_maps[0].cpu().numpy()
            self._img_plot.set_data(tactile_map_np)

            # Auto-scale if vmax not set
            if self._vmax is None:
                max_val = max(tactile_map_np.max(), 1.0)
                self._img_plot.set_clim(vmin=0, vmax=max_val)

            plt.pause(0.001)

        return tactile_maps

    def close(self):
        """Close the viewer."""
        if self._fig is not None:
            import matplotlib.pyplot as plt

            plt.close(self._fig)
            self._fig = None
            self._ax = None
            self._img_plot = None

    def show(self):
        """Show the viewer (blocking)."""
        if self._show_viewer:
            import matplotlib.pyplot as plt

            plt.ioff()
            plt.show()

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def image_shape(self) -> Tuple[int, int]:
        """Shape of the output tactile map (height, width)."""
        return self._image_shape

    @property
    def num_tactile_points(self) -> int:
        """Total number of tactile points."""
        return self._num_tactile_points

    @property
    def num_envs(self) -> int:
        """Number of parallel environments (batch size)."""
        return self._num_envs

    @property
    def num_pixels_with_points(self) -> int:
        """Number of pixels that have at least one tactile point mapped."""
        return self._num_pixels_with_points

    @property
    def device(self) -> torch.device:
        """PyTorch device used for tensor operations."""
        return self._device

    @property
    def links_data(self) -> dict:
        """Dictionary of link data from tactile grid."""
        return self._links_data

    @property
    def link_global_offset(self) -> dict:
        """Dictionary mapping link names to their global index offsets."""
        return self._link_global_offset
