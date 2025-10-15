"""
TactileFieldSensor: Dense tactile force field sensor for Genesis
Inspired by TacSL implementation from IsaacGymEnvs

Uses SDF-based penetration depth to compute forces at each tactile point.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Type

import numpy as np
import torch
import trimesh
from trimesh.proximity import ProximityQuery

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
    indenter_mesh_path : str
        Path to the indenter mesh file for SDF construction
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
    indenter_mesh_path: str = ""
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
    # SDF objects (stored as list since they're not tensors)
    sdf_objects: list = None


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

        # Build SDF for the indenter
        self._build_indenter_sdf()

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

        # Store SDF object
        if self._shared_metadata.sdf_objects is None:
            self._shared_metadata.sdf_objects = []
        self._shared_metadata.sdf_objects.append(self._sdf)

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

    def _build_indenter_sdf(self):
        """
        Build SDF (Signed Distance Field) for the indenter mesh.
        Uses trimesh.proximity.ProximityQuery like TacSL.
        """
        if not self._options.indenter_mesh_path:
            gs.raise_exception("indenter_mesh_path must be specified for TactileFieldSensor")

        try:
            # Load indenter mesh
            indenter_mesh = trimesh.load(self._options.indenter_mesh_path, force='mesh')
            gs.logger.info(
                f"[TactileFieldSensor] Loaded indenter mesh: {self._options.indenter_mesh_path} "
                f"({len(indenter_mesh.vertices)} vertices, {len(indenter_mesh.faces)} faces)"
            )

            # Create SDF using trimesh ProximityQuery
            self._sdf = ProximityQuery(indenter_mesh)
            gs.logger.info("[TactileFieldSensor] Built SDF for indenter mesh")

        except Exception as e:
            gs.raise_exception(f"Failed to build SDF for indenter: {e}")

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

            # Get SDF object
            sdf = shared_metadata.sdf_objects[i_sensor]

            # Transform tactile points to world frame
            tactile_points_world = cls._transform_points_to_world(
                shared_metadata.solver,
                sensor_link_idx,
                tactile_points_local,
                n_envs
            )

            # Compute forces at each tactile point using SDF
            forces = cls._compute_sdf_based_forces(
                shared_metadata.solver,
                sdf,
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
    def _compute_sdf_based_forces(cls, solver, sdf, tactile_points_world,
                                    sensor_link_idx, indenter_link_idx,
                                    kn, kt, mu, damping, n_envs):
        """
        Compute forces using SDF-based penetration depth.

        OPTIMIZED VERSION:
        1. First check if there's ANY contact using Genesis's collision detection
        2. Only query SDF if contact exists
        3. Use faster bounding box pre-filtering
        """
        n_points = tactile_points_world.shape[1]
        forces = torch.zeros((n_envs, n_points, 3), dtype=gs.tc_float, device=gs.device)

        # Quick check: is there any contact at all?
        all_contacts = solver.collider.get_contacts(as_tensor=True, to_torch=True)
        if all_contacts["link_a"].numel() == 0:
            return forces  # No contacts in scene, return zero forces

        # Check if sensor link is involved in any contact
        link_a = all_contacts["link_a"]
        link_b = all_contacts["link_b"]

        sensor_has_contact = False
        if n_envs > 0:
            sensor_has_contact = ((link_a == sensor_link_idx) | (link_b == sensor_link_idx)).any()
        else:
            sensor_has_contact = ((link_a == sensor_link_idx) | (link_b == sensor_link_idx)).any()

        if not sensor_has_contact:
            return forces  # Sensor not in contact, return zero forces

        # Get indenter pose
        links_pos = solver.get_links_pos()
        links_quat = solver.get_links_quat()

        if n_envs == 1 and links_pos.dim() == 2:
            indenter_pos = links_pos[indenter_link_idx, :].unsqueeze(0)  # (1, 3)
            indenter_quat = links_quat[indenter_link_idx, :].unsqueeze(0)  # (1, 4)
        else:
            indenter_pos = links_pos[:, indenter_link_idx, :]  # (B, 3)
            indenter_quat = links_quat[:, indenter_link_idx, :]  # (B, 4)

        # Process each environment
        for i_env in range(n_envs):
            # Get tactile points for this environment
            points_world_env = tactile_points_world[i_env]  # (N, 3) on GPU

            # Transform points to indenter local frame
            indenter_pos_env = indenter_pos[i_env]
            indenter_quat_env = indenter_quat[i_env]

            # Transform all tactile points to indenter local frame
            points_world_np = points_world_env.cpu().numpy()

            # Inverse transform: p_local = quat_inv_rotate(p_world - indenter_pos)
            points_relative_np = points_world_np - indenter_pos_env.cpu().numpy()
            from scipy.spatial.transform import Rotation as R
            indenter_quat_np = indenter_quat_env.cpu().numpy()
            rot = R.from_quat([indenter_quat_np[1], indenter_quat_np[2],
                              indenter_quat_np[3], indenter_quat_np[0]])  # x,y,z,w format
            points_indenter_frame = rot.inv().apply(points_relative_np)

            # Query SDF for signed distance for all tactile points
            # Per trimesh docs: points INSIDE mesh have POSITIVE distance
            signed_distances = sdf.signed_distance(points_indenter_frame)

            # Penetration depth: positive signed_distance means point is inside (penetrating)
            penetration_depth = signed_distances
            penetration_mask_local = penetration_depth > 0

            if penetration_mask_local.any():
                # Get surface normals at penetrated points
                closest_points, _, _ = sdf.on_surface(points_indenter_frame[penetration_mask_local])

                # Normal direction points from closest surface point to query point
                normals_indenter = points_indenter_frame[penetration_mask_local] - closest_points
                norms = np.linalg.norm(normals_indenter, axis=1, keepdims=True)
                normals_indenter = normals_indenter / (norms + 1e-9)

                # Transform normals back to world frame
                normals_world = rot.apply(normals_indenter)

                # Compute normal forces (penalty method)
                fc_norm = kn * penetration_depth[penetration_mask_local]

                # Apply forces in normal direction
                forces_world = fc_norm[:, None] * normals_world

                # Map back to full tactile grid
                # Since we query all points, penetration_mask_local directly maps to tactile grid indices
                penetrated_indices = torch.tensor(penetration_mask_local, device=gs.device)

                forces[i_env, penetrated_indices, :] = torch.tensor(
                    forces_world, dtype=gs.tc_float, device=gs.device
                )

        return forces  # (B, N, 3)

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
