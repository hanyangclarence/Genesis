"""
TactileFieldSensor: Dense tactile force field sensor for Genesis
Inspired by TacSL implementation from IsaacGymEnvs

Uses Genesis's precomputed SDF for fast penetration depth and normal queries.

Migrated from Genesis v0.3.3 to v0.3.10.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Type

import gstaichi as ti
import numpy as np
import torch

import genesis as gs
from genesis.utils.geom import transform_by_quat, inv_transform_by_quat
from genesis.utils.misc import concat_with_tensor, make_tensor_field

from .base_sensor import (
    Sensor,
    SharedSensorMetadata,
)
from .sensor_manager import register_sensor

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity.rigid_link import RigidLink
    from genesis.engine.solvers import RigidSolver
    from genesis.utils.ring_buffer import TensorRingBuffer
    from genesis.vis.rasterizer_context import RasterizerContext

    from .sensor_manager import SensorManager


# ==================== Sensor Options ====================
# Import the options class from the central options module
from genesis.options.sensors import TactileField as TactileFieldSensorOptions


# ==================== Sensor Metadata ====================

@dataclass
class TactileFieldSensorMetadata(SharedSensorMetadata):
    """
    Shared metadata for all tactile field sensors.
    """
    # Solver reference
    solver: "RigidSolver | None" = None

    # Sensor link indices (global)
    links_idx: list[int] = field(default_factory=list)

    # Tactile point positions in local frame
    tactile_points_local: torch.Tensor = make_tensor_field((0, 3))

    # Number of tactile points per sensor
    n_tactile_points: list[int] = field(default_factory=list)

    # Force field parameters per sensor
    kn: torch.Tensor = make_tensor_field((0, 1))
    kt: torch.Tensor = make_tensor_field((0, 1))
    mu: torch.Tensor = make_tensor_field((0, 1))
    damping: torch.Tensor = make_tensor_field((0, 1))

    # Multi-indenter support: lists of indenter data (shared by all sensors)
    # Each indenter has: global link index, geometry, and bounding box
    indenter_links_idx: list[int] = field(default_factory=list)  # Global link index per indenter
    indenter_geoms: list[object] = field(default_factory=list)  # Geometry per indenter
    indenter_mesh_bbox_lowers: list[torch.Tensor] = field(default_factory=list)  # Bbox lower per indenter
    indenter_mesh_bbox_uppers: list[torch.Tensor] = field(default_factory=list)  # Bbox upper per indenter

    # Precomputed mappings for efficient batched processing (cached at build time)
    precomputed_sensor_link_indices: torch.Tensor = None  # (total_points,) - sensor link idx for each point
    precomputed_kn_per_point: torch.Tensor = None  # (total_points,) - kn value for each point


# ==================== Sensor Class ====================

@register_sensor(TactileFieldSensorOptions, TactileFieldSensorMetadata, tuple)
@ti.data_oriented
class TactileFieldSensor(Sensor[TactileFieldSensorMetadata]):
    """
    Dense tactile force field sensor using SDF-based penetration depth computation.

    Follows TacSL's approach:
    1. Generate tactile point grid on elastomer surface
    2. Build SDF of indenter mesh
    3. Query penetration depth at each tactile point
    4. Compute forces using penalty method: F = kn * depth
    """

    def __init__(
        self,
        sensor_options: TactileFieldSensorOptions,
        sensor_idx: int,
        data_cls: Type[tuple],
        sensor_manager: "SensorManager",
    ):
        # Calculate number of tactile points before calling super().__init__
        # IMPORTANT: This must be set correctly before super().__init__() because
        # _get_return_format() is called during parent initialization to set _cache_size
        if sensor_options.tactile_points_local is not None:
            self._n_tactile_points = len(sensor_options.tactile_points_local)
        else:
            self._n_tactile_points = sensor_options.num_rows * sensor_options.num_cols
        self._tactile_points_local = None
        self._link: "RigidLink | None" = None

        super().__init__(sensor_options, sensor_idx, data_cls, sensor_manager)

    def build(self):
        super().build()

        if self._shared_metadata.solver is None:
            self._shared_metadata.solver = self._manager._sim.rigid_solver

        # Generate tactile point grid
        self._generate_tactile_points()

        # Store sensor link index
        entity = self._shared_metadata.solver.entities[self._options.entity_idx]
        sensor_link_idx = self._options.link_idx_local + entity.link_start
        self._shared_metadata.links_idx.append(sensor_link_idx)
        self._link = entity.links[self._options.link_idx_local]

        # Store tactile points in shared metadata
        self._shared_metadata.tactile_points_local = concat_with_tensor(
            self._shared_metadata.tactile_points_local,
            self._tactile_points_local,
            dim=0,
        )
        self._shared_metadata.n_tactile_points.append(self._n_tactile_points)

        # Store force parameters
        self._shared_metadata.kn = concat_with_tensor(
            self._shared_metadata.kn,
            torch.tensor([[self._options.kn]], dtype=gs.tc_float, device=gs.device),
            dim=0,
        )
        self._shared_metadata.kt = concat_with_tensor(
            self._shared_metadata.kt,
            torch.tensor([[self._options.kt]], dtype=gs.tc_float, device=gs.device),
            dim=0,
        )
        self._shared_metadata.mu = concat_with_tensor(
            self._shared_metadata.mu,
            torch.tensor([[self._options.mu]], dtype=gs.tc_float, device=gs.device),
            dim=0,
        )
        self._shared_metadata.damping = concat_with_tensor(
            self._shared_metadata.damping,
            torch.tensor([[self._options.damping]], dtype=gs.tc_float, device=gs.device),
            dim=0,
        )

        # Register indenter geometries (only on first sensor, all sensors share the same indenters)
        if len(self._shared_metadata.indenter_geoms) == 0:
            self._register_indenters()

        # Precompute mappings after this sensor is built
        self._precompute_mappings()

    def _register_indenters(self):
        """
        Register all indenter geometries from the options.
        This is called only once (on first sensor) since all sensors share the same indenters.
        """
        # Normalize to lists
        ent_indices = (
            self._options.indenter_entity_idx
            if isinstance(self._options.indenter_entity_idx, list)
            else [self._options.indenter_entity_idx]
        )
        link_indices = (
            self._options.indenter_link_idx_local
            if isinstance(self._options.indenter_link_idx_local, list)
            else [self._options.indenter_link_idx_local]
        )

        for ent_idx, link_idx_local in zip(ent_indices, link_indices):
            indenter_entity = self._shared_metadata.solver.entities[ent_idx]
            indenter_link_idx = link_idx_local + indenter_entity.link_start
            indenter_link = indenter_entity.links[link_idx_local]

            if len(indenter_link.geoms) == 0:
                gs.raise_exception(f"Indenter link (entity={ent_idx}, link={link_idx_local}) has no geometries")

            # Loop through ALL geoms on this link (not just geoms[0])
            for geom_idx, indenter_geom in enumerate(indenter_link.geoms):
                # Compute bounding box from mesh vertices
                mesh_verts = indenter_geom._sdf_verts
                bbox_lower = torch.from_numpy(mesh_verts.min(axis=0)).to(device=gs.device, dtype=gs.tc_float)
                bbox_upper = torch.from_numpy(mesh_verts.max(axis=0)).to(device=gs.device, dtype=gs.tc_float)

                # Add small margin for safety
                safety_margin = 0.005  # 5mm margin
                bbox_lower = bbox_lower - safety_margin
                bbox_upper = bbox_upper + safety_margin

                # Store in metadata (same link_idx for all geoms on this link)
                self._shared_metadata.indenter_links_idx.append(indenter_link_idx)
                self._shared_metadata.indenter_geoms.append(indenter_geom)
                self._shared_metadata.indenter_mesh_bbox_lowers.append(bbox_lower)
                self._shared_metadata.indenter_mesh_bbox_uppers.append(bbox_upper)

                gs.logger.info(
                    f"[TactileFieldSensor] Registered indenter {len(self._shared_metadata.indenter_geoms)-1}: "
                    f"entity={ent_idx}, link={link_idx_local}, geom={geom_idx}/{len(indenter_link.geoms)}, "
                    f"global link idx={indenter_link_idx}, geom_idx={indenter_geom.idx}, "
                    f"sdf_res={indenter_geom.sdf_res}"
                )

    def _precompute_mappings(self):
        """
        Precompute mapping tensors for efficient batched processing.
        This avoids recomputing the same tensors in every simulation step.

        These mappings tell us, for each tactile point:
        - Which sensor link it's attached to
        - What force parameters (kn) to use
        """
        n_sensors = len(self._shared_metadata.links_idx)

        # Convert lists to tensors for indexing
        n_tactile_points_tensor = torch.tensor(self._shared_metadata.n_tactile_points, dtype=gs.tc_int, device=gs.device)
        links_idx_tensor = torch.tensor(self._shared_metadata.links_idx, dtype=gs.tc_int, device=gs.device)

        # Create mapping: point index -> sensor index
        point_to_sensor = torch.repeat_interleave(
            torch.arange(n_sensors, device=gs.device, dtype=gs.tc_int),
            n_tactile_points_tensor
        )  # (total_points,)

        # Get sensor link indices for each point
        sensor_link_indices = links_idx_tensor[point_to_sensor]  # (total_points,)

        # Get kn for each point
        kn_per_point = self._shared_metadata.kn[point_to_sensor, 0]  # (total_points,)

        # Store precomputed mappings
        self._shared_metadata.precomputed_sensor_link_indices = sensor_link_indices
        self._shared_metadata.precomputed_kn_per_point = kn_per_point

        gs.logger.debug(
            f"[TactileFieldSensor] Precomputed mappings for {n_sensors} sensors, "
            f"{len(sensor_link_indices)} total tactile points, "
            f"{len(self._shared_metadata.indenter_geoms)} indenters"
        )

    def _generate_tactile_points(self):
        """
        Generate tactile points on the sensor surface.
        If custom points are provided, use them directly.
        Otherwise, generate a uniform grid of tactile points (similar to TacSL).
        """
        # Use custom tactile points if provided
        if self._options.tactile_points_local is not None:
            points = self._options.tactile_points_local
            self._n_tactile_points = len(points)
            self._tactile_points_local = torch.tensor(points, dtype=gs.tc_float, device=gs.device)

            gs.logger.info(
                f"[TactileFieldSensor] Using {self._n_tactile_points} custom tactile points"
            )
            return

        # Generate uniform grid
        num_rows = self._options.num_rows
        num_cols = self._options.num_cols
        width, height = self._options.surface_size

        # Create uniform grid
        x = np.linspace(-width/2, width/2, num_cols)
        y = np.linspace(-height/2, height/2, num_rows)
        xv, yv = np.meshgrid(x, y)

        # Z is at the top of the sensor (base + elastomer thickness)
        # In the URDF: sensor_base is 0.01m thick, elastomer_pad is 0.005m above that
        # So top surface is at z = 0.005 (base center) + 0.005 (to base top) + 0.0025 (half elastomer) + 0.0025 (other half) = 0.015
        z = np.ones_like(xv) * 0.015  # Top of elastomer layer

        # Stack into (N, 3) array
        points = np.stack([xv.flatten(), yv.flatten(), z.flatten()], axis=-1)

        self._n_tactile_points = len(points)
        self._tactile_points_local = torch.tensor(points, dtype=gs.tc_float, device=gs.device)

        gs.logger.info(
            f"[TactileFieldSensor] Generated {self._n_tactile_points} tactile points "
            f"({num_rows}x{num_cols}) on surface ({width:.3f}m x {height:.3f}m)"
        )

    def _get_return_format(self) -> tuple[int, ...]:
        # Return 3 force components per tactile point
        return (self._n_tactile_points * 3,)

    @classmethod
    def _get_cache_dtype(cls) -> torch.dtype:
        return gs.tc_float

    @classmethod
    def _update_shared_ground_truth_cache(
        cls, shared_metadata: TactileFieldSensorMetadata, shared_ground_truth_cache: torch.Tensor
    ):
        """
        Compute tactile force field using SDF-based penetration depth.

        Multi-indenter version: Loop through each indenter and accumulate forces.

        Following TacSL's approach (tacsl_sensors.py:825-887):
        1. Transform tactile points to world frame (all sensors at once)
        2. For each indenter: transform to indenter frame, query SDF, compute forces
        3. Sum forces from all indenters
        """
        assert shared_metadata.solver is not None

        n_sensors = len(shared_metadata.links_idx)
        n_envs = shared_metadata.solver._scene._B
        if n_envs == 0:
            n_envs = 1  # Handle non-batched case

        # Initialize force field to zero
        shared_ground_truth_cache.fill_(0.0)

        if n_sensors == 0:
            return

        # Get all link poses at once
        links_pos = shared_metadata.solver.get_links_pos()  # (B, L, 3) or (L, 3)
        links_quat = shared_metadata.solver.get_links_quat()  # (B, L, 4) or (L, 4)

        # Use precomputed mappings (cached at build time)
        sensor_link_indices = shared_metadata.precomputed_sensor_link_indices  # (total_points,)
        kn_per_point = shared_metadata.precomputed_kn_per_point  # (total_points,)

        # All tactile points in local frame (already stored contiguously)
        all_tactile_points_local = shared_metadata.tactile_points_local  # (total_points, 3)
        total_points = all_tactile_points_local.shape[0]

        # Transform ALL tactile points to world frame in one operation
        if n_envs == 1 and links_pos.dim() == 2:
            # Non-batched case
            point_link_pos = links_pos[sensor_link_indices, :]  # (total_points, 3)
            point_link_quat = links_quat[sensor_link_indices, :]  # (total_points, 4)

            # Add batch dimension
            point_link_pos = point_link_pos.unsqueeze(0)  # (1, total_points, 3)
            point_link_quat = point_link_quat.unsqueeze(0)  # (1, total_points, 4)
            all_tactile_points_local_batched = all_tactile_points_local.unsqueeze(0)  # (1, total_points, 3)
        else:
            # Batched case
            point_link_pos = links_pos[:, sensor_link_indices, :]  # (B, total_points, 3)
            point_link_quat = links_quat[:, sensor_link_indices, :]  # (B, total_points, 4)
            all_tactile_points_local_batched = all_tactile_points_local.unsqueeze(0).expand(n_envs, -1, -1)  # (B, total_points, 3)

        # Single batched transform for ALL points: p_world = link_pos + quat_rotate(link_quat, p_local)
        all_tactile_points_world = point_link_pos + transform_by_quat(all_tactile_points_local_batched, point_link_quat)  # (B, total_points, 3)

        # Accumulate forces from all indenters
        all_forces = torch.zeros((n_envs, total_points, 3), dtype=gs.tc_float, device=gs.device)

        # Loop through each indenter and compute forces
        n_indenters = len(shared_metadata.indenter_geoms)
        for i in range(n_indenters):
            indenter_link_idx = shared_metadata.indenter_links_idx[i]
            geom = shared_metadata.indenter_geoms[i]
            bbox_lower = shared_metadata.indenter_mesh_bbox_lowers[i]
            bbox_upper = shared_metadata.indenter_mesh_bbox_uppers[i]

            # Compute forces for this indenter
            indenter_forces = cls._compute_sdf_based_forces_for_indenter(
                shared_metadata.solver,
                all_tactile_points_world,
                point_link_quat,
                indenter_link_idx,
                geom,
                bbox_lower,
                bbox_upper,
                kn_per_point,
                n_envs
            )  # (B, total_points, 3)

            # For heterogeneous morphs, only apply forces for envs where this geom is active
            if geom.active_envs_mask is not None:
                indenter_forces = indenter_forces * geom.active_envs_mask.view(n_envs, 1, 1).float()

            # Sum forces from all indenters
            all_forces = all_forces + indenter_forces

        # Flatten and store in cache
        shared_ground_truth_cache[:, :] = all_forces.reshape(n_envs, -1)

    @classmethod
    def _query_genesis_sdf_gpu(cls, geom, points_mesh_frame_torch):
        """
        Query Genesis's precomputed SDF using optimized GPU trilinear interpolation.

        Uses torch.nn.functional.grid_sample for 1.5-2x faster interpolation.
        Benchmarking showed this is faster than Taichi due to no CPU/GPU transfer overhead.

        Args:
            geom: RigidGeom object with precomputed SDF
            points_mesh_frame_torch: torch.Tensor of shape (N, 3) in mesh coordinate frame (on GPU)

        Returns:
            sdf_values: torch.Tensor of shape (N,) with signed distances (on GPU)
            sdf_grads: torch.Tensor of shape (N, 3) with gradients/normals (on GPU)
        """
        import torch.nn.functional as F

        N = points_mesh_frame_torch.shape[0]
        device = points_mesh_frame_torch.device

        # Convert SDF data to torch tensors on GPU (cached for efficiency)
        # Also prepare grid_sample compatible format (cached)
        if not hasattr(geom, '_sdf_val_torch'):
            geom._sdf_val_torch = torch.from_numpy(geom.sdf_val).to(device=device, dtype=gs.tc_float)
            geom._sdf_grad_torch = torch.from_numpy(geom.sdf_grad).to(device=device, dtype=gs.tc_float)
            geom._T_mesh_to_sdf_torch = torch.from_numpy(geom.T_mesh_to_sdf).to(device=device, dtype=gs.tc_float)

            # Prepare grid_sample format: (1, C, D, H, W)
            # SDF val: (1, 1, D, H, W)
            geom._sdf_val_grid = geom._sdf_val_torch.unsqueeze(0).unsqueeze(0)
            # SDF grad: (1, 3, D, H, W) - channel dimension is the gradient components
            geom._sdf_grad_grid = geom._sdf_grad_torch.permute(3, 0, 1, 2).unsqueeze(0)

        # Transform to SDF grid coordinates (vectorized)
        T = geom._T_mesh_to_sdf_torch
        points_homo = torch.cat([points_mesh_frame_torch, torch.ones((N, 1), device=device, dtype=gs.tc_float)], dim=1)
        points_sdf = (T @ points_homo.T).T[:, :3]  # (N, 3)

        res = torch.tensor(geom.sdf_res, device=device, dtype=gs.tc_float)
        cell_size = geom.sdf_cell_size

        # Identify points outside grid
        outside_mask = (points_sdf >= res - 1).any(dim=1) | (points_sdf < 0).any(dim=1)
        inside_mask = ~outside_mask

        # Initialize outputs
        sdf_values = torch.zeros(N, device=device, dtype=gs.tc_float)
        sdf_grads = torch.zeros((N, 3), device=device, dtype=gs.tc_float)

        # Handle outside points (proxy distance)
        if outside_mask.any():
            center = (res - 1) / 2.0
            points_outside = points_sdf[outside_mask]
            diff = points_outside - center
            dist_to_center = torch.norm(diff, dim=1)
            sdf_values[outside_mask] = dist_to_center / cell_size + geom.sdf_max
            sdf_grads[outside_mask] = diff / (dist_to_center[:, None] + 1e-9)

        # Handle inside points using optimized grid_sample
        if inside_mask.any():
            points_inside = points_sdf[inside_mask]  # (M, 3)

            # Normalize coordinates to [-1, 1] range for grid_sample
            # grid_sample expects (x, y, z) in range [-1, 1] where:
            #   -1 maps to coordinate 0
            #   +1 maps to coordinate (size - 1)
            grid_coords_normalized = (points_inside / (res - 1)) * 2.0 - 1.0  # (M, 3)

            # grid_sample expects coordinates in (z, y, x) order (reversed)
            grid_coords_normalized = grid_coords_normalized.flip(-1)  # (M, 3) - now (z, y, x)

            # Reshape for grid_sample: (1, 1, 1, M, 3)
            grid_coords_5d = grid_coords_normalized.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, M, 3)

            # Interpolate SDF values using grid_sample (highly optimized CUDA kernel)
            # mode='bilinear' gives trilinear interpolation for 3D (confusing naming)
            # align_corners=True ensures correct mapping of grid coordinates
            sdf_vals_sampled = F.grid_sample(
                geom._sdf_val_grid,  # (1, 1, D, H, W)
                grid_coords_5d,      # (1, 1, 1, M, 3)
                mode='bilinear',     # trilinear for 3D
                padding_mode='border',
                align_corners=True
            )  # (1, 1, 1, 1, M)

            # Interpolate SDF gradients
            sdf_grads_sampled = F.grid_sample(
                geom._sdf_grad_grid,  # (1, 3, D, H, W)
                grid_coords_5d,       # (1, 1, 1, M, 3)
                mode='bilinear',
                padding_mode='border',
                align_corners=True
            )  # (1, 3, 1, 1, M)

            # Reshape outputs: (1, 1, 1, 1, M) -> (M,)
            sdf_values[inside_mask] = sdf_vals_sampled.squeeze()
            # Reshape: (1, 3, 1, 1, M) -> (M, 3)
            sdf_grads[inside_mask] = sdf_grads_sampled.squeeze(0).squeeze(1).squeeze(1).permute(1, 0)

        return sdf_values, sdf_grads

    @classmethod
    def _transform_points_to_world(cls, solver, link_idx, points_local, n_envs):
        """
        Transform tactile points from link local frame to world frame.
        """
        # Get link pose
        links_pos = solver.get_links_pos()
        links_quat = solver.get_links_quat()

        if n_envs == 1 and links_pos.dim() == 2:
            # Non-batched: add batch dimension
            link_pos = links_pos[link_idx, :].unsqueeze(0)  # (1, 3)
            link_quat = links_quat[link_idx, :].unsqueeze(0)  # (1, 4)
        else:
            link_pos = links_pos[:, link_idx, :]  # (B, 3)
            link_quat = links_quat[:, link_idx, :]  # (B, 4)

        # Expand for all tactile points
        n_points = points_local.shape[0]
        link_pos_expanded = link_pos.unsqueeze(1).expand(n_envs, n_points, 3)  # (B, N, 3)
        link_quat_expanded = link_quat.unsqueeze(1).expand(n_envs, n_points, 4)  # (B, N, 4)
        points_local_expanded = points_local.unsqueeze(0).expand(n_envs, n_points, 3)  # (B, N, 3)

        # Transform: p_world = link_pos + quat_rotate(link_quat, p_local)
        points_world = link_pos_expanded + transform_by_quat(points_local_expanded, link_quat_expanded)

        return points_world  # (B, N, 3)

    @classmethod
    def _compute_sdf_based_forces_for_indenter(cls, solver, tactile_points_world,
                                                sensor_link_quat, indenter_link_idx,
                                                geom, bbox_lower, bbox_upper,
                                                kn_per_point, n_envs):
        """
        Compute forces using Genesis's precomputed SDF for a single indenter.

        Args:
            solver: RigidSolver instance
            tactile_points_world: (B, total_points, 3) - ALL tactile points in world frame
            sensor_link_quat: (B, total_points, 4) - Sensor link quaternions for each point
            indenter_link_idx: int - Global link index of the indenter
            geom: RigidGeom - Indenter geometry with precomputed SDF
            bbox_lower: (3,) - Lower bound of indenter mesh bounding box
            bbox_upper: (3,) - Upper bound of indenter mesh bounding box
            kn_per_point: (total_points,) - Normal stiffness for each point
            n_envs: int - Number of environments

        Returns:
            forces: (B, total_points, 3) - Force vectors at each tactile point
        """
        total_points = tactile_points_world.shape[1]
        forces = torch.zeros((n_envs, total_points, 3), dtype=gs.tc_float, device=gs.device)

        # Get all link poses
        links_pos = solver.get_links_pos()  # (B, L, 3) or (L, 3)
        links_quat = solver.get_links_quat()  # (B, L, 4) or (L, 4)

        # Get indenter pose (same for all points since it's a single indenter)
        if n_envs == 1 and links_pos.dim() == 2:
            # Non-batched case
            indenter_pos = links_pos[indenter_link_idx, :].unsqueeze(0).unsqueeze(1)  # (1, 1, 3)
            indenter_quat = links_quat[indenter_link_idx, :].unsqueeze(0).unsqueeze(1)  # (1, 1, 4)
        else:
            # Batched case
            indenter_pos = links_pos[:, indenter_link_idx, :].unsqueeze(1)  # (B, 1, 3)
            indenter_quat = links_quat[:, indenter_link_idx, :].unsqueeze(1)  # (B, 1, 4)

        # Expand to all points
        indenter_pos = indenter_pos.expand(-1, total_points, -1)  # (B, total_points, 3)
        indenter_quat = indenter_quat.expand(-1, total_points, -1)  # (B, total_points, 4)

        # Transform ALL points to indenter frame
        points_relative = tactile_points_world - indenter_pos  # (B, total_points, 3)
        points_indenter_frame = inv_transform_by_quat(points_relative, indenter_quat)  # (B, total_points, 3)

        # Bounding box pre-filtering using TIGHT mesh bounds
        in_bbox = ((points_indenter_frame >= bbox_lower) & (points_indenter_frame <= bbox_upper)).all(dim=2)  # (B, total_points)

        if not in_bbox.any():
            return forces  # No points near the indenter

        # Query SDF for points in bbox
        points_to_query = points_indenter_frame[in_bbox]  # (M, 3) where M = sum of all in_bbox

        if points_to_query.shape[0] > 0:
            # Query SDF for all points at once
            signed_distances_flat, sdf_gradients_flat = cls._query_genesis_sdf_gpu(geom, points_to_query)  # (M,), (M, 3)

            # Scatter results back to original shape
            signed_distances = torch.zeros((n_envs, total_points), device=gs.device, dtype=gs.tc_float)
            sdf_gradients = torch.zeros((n_envs, total_points, 3), device=gs.device, dtype=gs.tc_float)
            signed_distances[in_bbox] = signed_distances_flat
            sdf_gradients[in_bbox] = sdf_gradients_flat
        else:
            signed_distances = torch.zeros((n_envs, total_points), device=gs.device, dtype=gs.tc_float)
            sdf_gradients = torch.zeros((n_envs, total_points, 3), device=gs.device, dtype=gs.tc_float)

        # Compute penetration and normals
        penetration_depth = -signed_distances  # (B, total_points)
        penetration_mask = penetration_depth > 0  # (B, total_points)
        sdf_gradients = -sdf_gradients  # Flip gradients to point outward

        if penetration_mask.any():
            # Normalize gradients to get normals
            normals_indenter = sdf_gradients.clone()
            norms = torch.norm(normals_indenter, dim=2, keepdim=True)
            normals_indenter = normals_indenter / (norms + 1e-9)
        else:
            normals_indenter = sdf_gradients

        # Transform normals to world frame
        normals_world = transform_by_quat(normals_indenter, indenter_quat)  # (B, total_points, 3)

        # Compute normal forces (penalty method)
        kn_per_point_batched = kn_per_point.unsqueeze(0).expand(n_envs, -1)  # (B, total_points)
        fc_norm = kn_per_point_batched * penetration_depth  # (B, total_points)

        # Apply forces in normal direction
        forces_world = fc_norm.unsqueeze(-1) * normals_world  # (B, total_points, 3)

        # Transform forces to sensor link frame
        forces_local = inv_transform_by_quat(forces_world, sensor_link_quat)  # (B, total_points, 3)

        # Zero out forces where there's no penetration
        forces_local = forces_local * penetration_mask.unsqueeze(-1).float()

        return forces_local

    @classmethod
    def _update_shared_cache(
        cls,
        shared_metadata: TactileFieldSensorMetadata,
        shared_ground_truth_cache: torch.Tensor,
        shared_cache: torch.Tensor,
        buffered_data: "TensorRingBuffer",
    ):
        """
        Update the shared sensor cache for all tactile field sensors.

        For now, we simply apply the delay and copy ground truth to cache.
        Noise models can be added later if needed.
        """
        buffered_data.set(shared_ground_truth_cache)
        cls._apply_delay_to_shared_cache(shared_metadata, shared_cache, buffered_data)
