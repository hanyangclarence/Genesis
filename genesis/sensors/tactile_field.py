"""
TactileFieldSensor: Dense tactile force field sensor for Genesis
Inspired by TacSL implementation from IsaacGymEnvs

Uses Genesis's precomputed SDF for fast penetration depth and normal queries.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Type

import numpy as np
import torch

import genesis as gs
from genesis.utils.geom import transform_by_quat
from genesis.utils.misc import concat_with_tensor, make_tensor_field

from .base_sensor import (
    NoisySensorMetadataMixin,
    NoisySensorMixin,
    NoisySensorOptionsMixin,
    RigidSensorMetadataMixin,
    RigidSensorMixin,
    RigidSensorOptionsMixin,
    Sensor,
    SensorOptions,
    SharedSensorMetadata,
)
from .sensor_manager import register_sensor

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity.rigid_link import RigidLink
    from genesis.utils.ring_buffer import TensorRingBuffer
    from genesis.vis.rasterizer_context import RasterizerContext
    from .sensor_manager import SensorManager


class TactileFieldSensorOptions(RigidSensorOptionsMixin, NoisySensorOptionsMixin, SensorOptions):
    """
    Sensor that returns dense tactile force field on a surface using SDF-based penetration.

    Parameters
    ----------
    num_rows : int
        Number of tactile points in the first dimension (default: 10)
    num_cols : int
        Number of tactile points in the second dimension (default: 10)
    elastomer_thickness : float
        Thickness of the elastomer layer in meters (default: 0.005)
    surface_size : tuple[float, float]
        Width and height of the tactile surface (default: (0.08, 0.08))
    indenter_entity_idx : int
        Entity index of the indenter object (the object making contact)
    indenter_link_idx_local : int
        Local link index of the indenter within its entity (default: 0)
    kn : float
        Normal stiffness coefficient (default: 1000.0)
    kt : float
        Tangential stiffness coefficient (default: 100.0)
    mu : float
        Friction coefficient (default: 0.5)
    damping : float
        Contact damping coefficient (default: 0.003)
    """

    num_rows: int = 10
    num_cols: int = 10
    elastomer_thickness: float = 0.005
    surface_size: tuple[float, float] = (0.08, 0.08)
    indenter_entity_idx: int = -1
    indenter_link_idx_local: int = 0
    kn: float = 1000.0
    kt: float = 100.0
    mu: float = 0.5
    damping: float = 0.003


@dataclass
class TactileFieldSensorMetadata(RigidSensorMetadataMixin, NoisySensorMetadataMixin, SharedSensorMetadata):
    """
    Shared metadata for all tactile field sensors.
    """
    # Tactile point positions in local frame
    tactile_points_local: torch.Tensor = make_tensor_field((0, 3))
    # Tactile point orientations in local frame (for force transformation)
    tactile_points_quat_local: torch.Tensor = make_tensor_field((0, 4))
    # Number of tactile points per sensor
    n_tactile_points: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    # Indenter link indices (global)
    indenter_links_idx: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)
    # Force field parameters per sensor
    kn: torch.Tensor = make_tensor_field((0, 1))
    kt: torch.Tensor = make_tensor_field((0, 1))
    mu: torch.Tensor = make_tensor_field((0, 1))
    damping: torch.Tensor = make_tensor_field((0, 1))
    # Indenter geometries (for Genesis precomputed SDF access)
    indenter_geoms: list = None


@register_sensor(TactileFieldSensorOptions, TactileFieldSensorMetadata, tuple)
class TactileFieldSensor(
    RigidSensorMixin[TactileFieldSensorMetadata],
    NoisySensorMixin[TactileFieldSensorMetadata],
    Sensor[TactileFieldSensorMetadata],
):
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
        self._n_tactile_points = sensor_options.num_rows * sensor_options.num_cols
        self._tactile_points_local = None
        self._tactile_points_quat_local = None
        self._sdf = None
        super().__init__(sensor_options, sensor_idx, data_cls, sensor_manager)

    def build(self):
        super().build()

        if self._shared_metadata.solver is None:
            self._shared_metadata.solver = self._manager._sim.rigid_solver

        # Generate tactile point grid
        self._generate_tactile_points()

        # Get indenter geometry (use Genesis's precomputed SDF)
        self._get_indenter_geom()

        # Store tactile points in shared metadata
        self._shared_metadata.tactile_points_local = concat_with_tensor(
            self._shared_metadata.tactile_points_local,
            self._tactile_points_local,
            expand=(self._n_tactile_points, 3),
        )
        self._shared_metadata.tactile_points_quat_local = concat_with_tensor(
            self._shared_metadata.tactile_points_quat_local,
            self._tactile_points_quat_local,
            expand=(self._n_tactile_points, 4),
        )
        self._shared_metadata.n_tactile_points = concat_with_tensor(
            self._shared_metadata.n_tactile_points,
            torch.tensor([self._n_tactile_points], dtype=gs.tc_int, device=gs.device),
            expand=(1,),
        )

        # Store indenter link index
        entity = self._shared_metadata.solver.entities[self._options.indenter_entity_idx]
        indenter_link_idx = self._options.indenter_link_idx_local + entity.link_start
        self._shared_metadata.indenter_links_idx = concat_with_tensor(
            self._shared_metadata.indenter_links_idx,
            torch.tensor([indenter_link_idx], dtype=gs.tc_int, device=gs.device),
            expand=(1,),
        )

        # Store force parameters
        self._shared_metadata.kn = concat_with_tensor(
            self._shared_metadata.kn, torch.tensor([[self._options.kn]], device=gs.device), expand=(1, 1)
        )
        self._shared_metadata.kt = concat_with_tensor(
            self._shared_metadata.kt, torch.tensor([[self._options.kt]], device=gs.device), expand=(1, 1)
        )
        self._shared_metadata.mu = concat_with_tensor(
            self._shared_metadata.mu, torch.tensor([[self._options.mu]], device=gs.device), expand=(1, 1)
        )
        self._shared_metadata.damping = concat_with_tensor(
            self._shared_metadata.damping, torch.tensor([[self._options.damping]], device=gs.device), expand=(1, 1)
        )

        # Store indenter geometry (for Genesis SDF)
        if self._shared_metadata.indenter_geoms is None:
            self._shared_metadata.indenter_geoms = []
        self._shared_metadata.indenter_geoms.append(self._indenter_geom)

    def _generate_tactile_points(self):
        """
        Generate a uniform grid of tactile points on the sensor surface.
        Similar to TacSL's generate_tactile_points() method.
        """
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
        # z = np.zeros_like(xv)  # temporary, set to 0 for now

        # Stack into (N, 3) array
        points = np.stack([xv.flatten(), yv.flatten(), z.flatten()], axis=-1)

        self._n_tactile_points = len(points)
        self._tactile_points_local = torch.tensor(points, dtype=gs.tc_float, device=gs.device)

        # All tactile points have same orientation (pointing up in local frame: +Z)
        # Normal direction: (0, 0, 1) in local frame
        # Quaternion for identity rotation (w, x, y, z)
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        self._tactile_points_quat_local = torch.tensor(
            np.tile(quat, (self._n_tactile_points, 1)),
            dtype=gs.tc_float,
            device=gs.device
        )

        gs.logger.info(
            f"[TactileFieldSensor] Generated {self._n_tactile_points} tactile points "
            f"({num_rows}x{num_cols}) on surface ({width:.3f}m x {height:.3f}m)"
        )

    def _get_indenter_geom(self):
        """
        Get the indenter geometry from Genesis's rigid solver.
        This geometry already has a precomputed GPU-accelerated SDF.
        """
        entity = self._shared_metadata.solver.entities[self._options.indenter_entity_idx]
        indenter_link = entity.links[self._options.indenter_link_idx_local]

        if len(indenter_link.geoms) == 0:
            gs.raise_exception("Indenter link has no geometries")

        # Get the first geometry (assuming single-geometry indenter)
        self._indenter_geom = indenter_link.geoms[0]

        gs.logger.info(
            f"[TactileFieldSensor] Using Genesis precomputed SDF for indenter geometry "
            f"(idx={self._indenter_geom.idx}, res={self._indenter_geom.sdf_res}, "
            f"cell_size={self._indenter_geom.sdf_cell_size:.6f}m)"
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

        Following TacSL's approach (tacsl_sensors.py:825-887):
        1. Transform tactile points to world frame
        2. Transform tactile points to indenter frame
        3. Query SDF for penetration depth and normal
        4. Compute forces using penalty method
        """
        assert shared_metadata.solver is not None

        n_sensors = len(shared_metadata.links_idx)
        n_envs = shared_metadata.solver._scene._B
        if n_envs == 0:
            n_envs = 1  # Handle non-batched case

        # Initialize force field to zero
        shared_ground_truth_cache.fill_(0.0)

        # Process each sensor
        offset = 0
        for i_sensor in range(n_sensors):
            sensor_link_idx = shared_metadata.links_idx[i_sensor]
            indenter_link_idx = shared_metadata.indenter_links_idx[i_sensor]
            n_points = shared_metadata.n_tactile_points[i_sensor].item()

            # Get tactile points for this sensor
            tactile_points_local = shared_metadata.tactile_points_local[offset:offset+n_points]

            # Get force parameters
            kn = shared_metadata.kn[i_sensor, 0].item()
            kt = shared_metadata.kt[i_sensor, 0].item()
            mu = shared_metadata.mu[i_sensor, 0].item()
            damping = shared_metadata.damping[i_sensor, 0].item()

            # Transform tactile points to world frame
            tactile_points_world = cls._transform_points_to_world(
                shared_metadata.solver,
                sensor_link_idx,
                tactile_points_local,
                n_envs
            )

            # Compute forces at each tactile point using Genesis SDF
            forces = cls._compute_sdf_based_forces(
                shared_metadata,
                i_sensor,
                tactile_points_world,
                sensor_link_idx,
                indenter_link_idx,
                kn, kt, mu, damping,
                n_envs
            )

            # Store in cache (flatten to 1D: B x n_points x 3 -> B x (n_points*3))
            start_idx = offset * 3
            end_idx = (offset + n_points) * 3
            shared_ground_truth_cache[:, start_idx:end_idx] = forces.reshape(n_envs, -1)

            offset += n_points

    @classmethod
    def _query_genesis_sdf_cpu(cls, geom, points_mesh_frame):
        """
        Query Genesis's precomputed SDF using CPU-based trilinear interpolation.

        Args:
            geom: RigidGeom object with precomputed SDF
            points_mesh_frame: numpy array of shape (N, 3) in mesh coordinate frame

        Returns:
            sdf_values: numpy array of shape (N,) with signed distances
            sdf_grads: numpy array of shape (N, 3) with gradients (normals)
        """
        N = points_mesh_frame.shape[0]
        sdf_values = np.zeros(N, dtype=np.float32)
        sdf_grads = np.zeros((N, 3), dtype=np.float32)

        # Transform to SDF grid coordinates
        T = geom.T_mesh_to_sdf
        points_homo = np.concatenate([points_mesh_frame, np.ones((N, 1))], axis=1)
        points_sdf = (T @ points_homo.T).T[:, :3]  # (N, 3)

        res = geom.sdf_res
        cell_size = geom.sdf_cell_size

        for i in range(N):
            point = points_sdf[i]

            # Check if outside grid (use proxy distance)
            if (point >= res - 1).any() or (point < 0).any():
                # Outside grid: use distance to grid center as proxy
                center = (res - 1) / 2.0
                dist_to_center = np.linalg.norm(point - center)
                sdf_values[i] = dist_to_center / cell_size + geom.sdf_max
                sdf_grads[i] = (point - center) / (dist_to_center + 1e-9)
                continue

            # Inside grid: use trilinear interpolation
            base = np.floor(point).astype(int)
            base = np.clip(base, 0, res - 2)

            sdf_val = 0.0
            grad = np.zeros(3, dtype=np.float32)

            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        offset = np.array([di, dj, dk])
                        idx = tuple(base + offset)
                        weight = np.prod(1.0 - np.abs(point - (base + offset)))

                        sdf_val += weight * geom.sdf_val[idx]
                        grad += weight * geom.sdf_grad[idx]

            sdf_values[i] = sdf_val
            sdf_grads[i] = grad

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
    def _compute_sdf_based_forces(cls, shared_metadata, i_sensor, tactile_points_world,
                                    sensor_link_idx, indenter_link_idx,
                                    kn, kt, mu, damping, n_envs):
        """
        Compute forces using Genesis's precomputed SDF.

        Uses Genesis's precomputed SDF voxel grids with trilinear interpolation
        for fast penetration depth and normal queries.
        """
        import time

        # Track timing for each step
        timings = {}
        t_start = time.perf_counter()

        solver = shared_metadata.solver
        n_points = tactile_points_world.shape[1]
        forces = torch.zeros((n_envs, n_points, 3), dtype=gs.tc_float, device=gs.device)

        # STEP 1: Check contacts
        t1 = time.perf_counter()
        all_contacts = solver.collider.get_contacts(as_tensor=True, to_torch=True)
        timings['1_get_contacts'] = time.perf_counter() - t1

        if all_contacts["link_a"].numel() == 0:
            timings['total'] = time.perf_counter() - t_start
            cls._print_timings(timings)
            return forces  # No contacts in scene, return zero forces

        # STEP 2: Check sensor involvement
        t2 = time.perf_counter()
        link_a = all_contacts["link_a"]
        link_b = all_contacts["link_b"]

        sensor_has_contact = False
        if n_envs > 0:
            sensor_has_contact = ((link_a == sensor_link_idx) | (link_b == sensor_link_idx)).any()
        else:
            sensor_has_contact = ((link_a == sensor_link_idx) | (link_b == sensor_link_idx)).any()
        timings['2_check_sensor_contact'] = time.perf_counter() - t2

        if not sensor_has_contact:
            timings['total'] = time.perf_counter() - t_start
            cls._print_timings(timings)
            return forces  # Sensor not in contact, return zero forces

        # STEP 3: Get indenter pose
        t3 = time.perf_counter()
        links_pos = solver.get_links_pos()
        links_quat = solver.get_links_quat()

        if n_envs == 1 and links_pos.dim() == 2:
            indenter_pos = links_pos[indenter_link_idx, :].unsqueeze(0)  # (1, 3)
            indenter_quat = links_quat[indenter_link_idx, :].unsqueeze(0)  # (1, 4)
        else:
            indenter_pos = links_pos[:, indenter_link_idx, :]  # (B, 3)
            indenter_quat = links_quat[:, indenter_link_idx, :]  # (B, 4)
        timings['3_get_indenter_pose'] = time.perf_counter() - t3

        # STEP 4-10: Process each environment
        timings['4_gpu_to_cpu'] = 0.0
        timings['5_coord_transform'] = 0.0
        timings['6_bbox_prefilter'] = 0.0
        timings['7_sdf_query'] = 0.0
        timings['8_get_normals'] = 0.0
        timings['9_compute_forces'] = 0.0
        timings['10_cpu_to_gpu'] = 0.0

        # Cache scipy rotation objects to avoid repeated imports
        from scipy.spatial.transform import Rotation as R

        for i_env in range(n_envs):
            # Get tactile points for this environment
            points_world_env = tactile_points_world[i_env]  # (N, 3) on GPU

            # Transform points to indenter local frame
            indenter_pos_env = indenter_pos[i_env]
            indenter_quat_env = indenter_quat[i_env]

            # STEP 4: GPU to CPU transfer (batched)
            t4 = time.perf_counter()
            points_world_np = points_world_env.cpu().numpy()
            indenter_pos_np = indenter_pos_env.cpu().numpy()
            indenter_quat_np = indenter_quat_env.cpu().numpy()
            timings['4_gpu_to_cpu'] += time.perf_counter() - t4

            # STEP 5: Coordinate transformation
            t5 = time.perf_counter()
            points_relative_np = points_world_np - indenter_pos_np
            rot = R.from_quat([indenter_quat_np[1], indenter_quat_np[2],
                              indenter_quat_np[3], indenter_quat_np[0]])  # x,y,z,w format
            points_indenter_frame = rot.inv().apply(points_relative_np)
            timings['5_coord_transform'] += time.perf_counter() - t5

            # STEP 6: Get indenter geometry
            t6 = time.perf_counter()
            geom = shared_metadata.indenter_geoms[i_sensor]

            # Bounding box pre-filtering using Genesis SDF grid bounds
            # Convert SDF grid bounds to mesh frame
            sdf_res = geom.sdf_res
            cell_size = geom.sdf_cell_size
            # SDF grid bounds in mesh frame (approximate)
            max_extent = (sdf_res[0] - 1) * cell_size / 2.0
            margin = 0.01  # 1cm margin for near-surface points

            # Filter points outside approximate bounds
            in_bbox = np.all((np.abs(points_indenter_frame) <= max_extent + margin), axis=1)
            timings['6_bbox_prefilter'] += time.perf_counter() - t6

            if not in_bbox.any():
                continue  # No points near the indenter, skip this environment

            # STEP 7: Query Genesis's precomputed SDF (only for points in bbox)
            t7 = time.perf_counter()
            # Query all points, but SDF function will handle out-of-bounds efficiently
            signed_distances, sdf_gradients = cls._query_genesis_sdf_cpu(geom, points_indenter_frame[in_bbox])

            # Expand to full grid
            signed_distances_full = np.zeros(n_points, dtype=np.float32)
            sdf_gradients_full = np.zeros((n_points, 3), dtype=np.float32)
            signed_distances_full[in_bbox] = signed_distances
            sdf_gradients_full[in_bbox] = sdf_gradients

            signed_distances = signed_distances_full
            sdf_gradients = sdf_gradients_full
            timings['7_sdf_query'] += time.perf_counter() - t7

            # Penetration depth: negative signed_distance means point is inside (penetrating)
            penetration_depth = - signed_distances
            penetration_mask_local = penetration_depth > 0
            sdf_gradients = -sdf_gradients  # Flip gradients to point outward

            if penetration_mask_local.any():
                # STEP 8: Get normals from precomputed gradients
                t8 = time.perf_counter()
                normals_indenter = sdf_gradients[penetration_mask_local]
                norms = np.linalg.norm(normals_indenter, axis=1, keepdims=True)
                normals_indenter = normals_indenter / (norms + 1e-9)
                timings['8_get_normals'] += time.perf_counter() - t8

                # STEP 9: Compute forces
                t9 = time.perf_counter()
                # Transform normals back to world frame
                normals_world = rot.apply(normals_indenter)

                # Compute normal forces (penalty method)
                fc_norm = kn * penetration_depth[penetration_mask_local]

                # Apply forces in normal direction
                forces_world = fc_norm[:, None] * normals_world
                timings['9_compute_forces'] += time.perf_counter() - t9

                # STEP 10: CPU to GPU transfer
                t10 = time.perf_counter()
                # Map back to full tactile grid
                penetrated_indices = torch.tensor(penetration_mask_local, device=gs.device)

                forces[i_env, penetrated_indices, :] = torch.tensor(
                    forces_world, dtype=gs.tc_float, device=gs.device
                )
                timings['10_cpu_to_gpu'] += time.perf_counter() - t10

        timings['total'] = time.perf_counter() - t_start
        cls._print_timings(timings)

        return forces  # (B, N, 3)

    @classmethod
    def _print_timings(cls, timings):
        """Print timing breakdown for profiling."""
        total = timings['total']
        gs.logger.info(f"\n{'='*70}")
        gs.logger.info(f"TACTILE FIELD SENSOR TIMING BREAKDOWN (Total: {total*1000:.2f}ms)")
        gs.logger.info(f"{'='*70}")

        # Sort by time (descending)
        sorted_timings = sorted([(k, v) for k, v in timings.items() if k != 'total'],
                               key=lambda x: x[1], reverse=True)

        for step_name, step_time in sorted_timings:
            pct = (step_time / total * 100) if total > 0 else 0
            gs.logger.info(f"  {step_name:30s}: {step_time*1000:7.2f}ms ({pct:5.1f}%)")

        gs.logger.info(f"{'='*70}\n")

    @classmethod
    def _update_shared_cache(
        cls,
        shared_metadata: TactileFieldSensorMetadata,
        shared_ground_truth_cache: torch.Tensor,
        shared_cache: torch.Tensor,
        buffered_data: "TensorRingBuffer",
    ):
        buffered_data.append(shared_ground_truth_cache)
        cls._apply_delay_to_shared_cache(shared_metadata, shared_cache, buffered_data)
        cls._add_noise_drift_bias(shared_metadata, shared_cache)
        cls._quantize_to_resolution(shared_metadata.resolution, shared_cache)

    def _draw_debug(self, context: "RasterizerContext", buffer_updates: dict[str, np.ndarray]):
        """
        Draw debug visualization of tactile points and forces.
        """
        if not self._options.draw_debug:
            return

        env_idx = context.rendered_envs_idx[0] if self._manager._sim.n_envs > 0 else 0

        # Get link pose
        link_pos = self._link.get_pos(envs_idx=env_idx if self._manager._sim.n_envs > 0 else None)
        link_quat = self._link.get_quat(envs_idx=env_idx if self._manager._sim.n_envs > 0 else None)

        if link_pos.dim() > 1:
            link_pos = link_pos.squeeze(0)
            link_quat = link_quat.squeeze(0)

        # Get force readings
        forces_flat = self.read(envs_idx=env_idx if self._manager._sim.n_envs > 0 else None)
        if forces_flat.dim() > 1:
            forces_flat = forces_flat.squeeze(0)
        forces = forces_flat.reshape(self._n_tactile_points, 3)

        # Transform tactile points to world and draw
        for i in range(self._n_tactile_points):
            point_local = self._tactile_points_local[i]
            point_world = link_pos + transform_by_quat(point_local.unsqueeze(0), link_quat.unsqueeze(0)).squeeze(0)

            # Draw sphere at tactile point (color based on force magnitude)
            force_mag = torch.norm(forces[i]).item()
            color = (min(force_mag / 10.0, 1.0), 1.0 - min(force_mag / 10.0, 1.0), 0.0, 0.7)

            context.draw_debug_sphere(
                pos=point_world.cpu().numpy(),
                radius=0.002,
                color=color
            )
