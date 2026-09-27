"""
Correctness tests for the TactileField sensor.

Uses the same Wuji hand (v5 model) + cylinder setup as examples/tactile/tactile_field_hand.py,
with the v5 tactile grid from examples/tactile/full_hand_tactile_v5.json.

NOTE: The batched physics solver produces different trajectories depending on n_envs
(batch size affects contact solver numerics). Comparisons with free cylinders are therefore
made WITHIN a single simulation, between envs assigned to the same variant. Comparisons
across simulations use fixed cylinders, where the position-controlled hand follows the same
trajectory regardless of batch size.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

import genesis as gs

TACTILE_GRID_PATH = Path(__file__).parents[1] / "examples" / "tactile" / "full_hand_tactile_v5.json"
URDF_PATH = "urdf/wujihand_v5/wujihand_right_v5.urdf"

HAND_POS = (0, 0.1, 0.1)
HAND_EULER = (90, 0, 0)
CYL_POS = (0.03, -0.01, 0.1)
CYL_HEIGHT = 0.2
RADII = (0.008, 0.012, 0.016)
KN = 2000.0
N_STEPS = 250

JOINTS_NAME = tuple(f"finger{i}_joint{j}" for i in range(1, 6) for j in range(1, 5))

POSE = np.array([
    0.7, -0.16, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
])
DELTA_POSE = np.array([
    0.00, 0.00, 0.00, 0.00,
    0.01, 0.00, 0.01, 0.01,
    0.01, 0.00, 0.01, 0.01,
    0.01, 0.00, 0.01, 0.01,
    0.01, 0.00, 0.01, 0.01,
])

# Links that make contact early in the hand-closing motion
TEST_LINKS = (
    "finger5_link3",
    "finger4_link3",
    "finger3_link3",
    "finger2_link3",
)


def run_sim(cyl_morphs, n_envs):
    """Close the hand on the cylinder(s) and record the tactile force magnitudes.

    `cyl_morphs` is a single morph, or a list of per-env variants. Returns link_name -> tensor of
    shape (N_STEPS, n_envs, n_points), or (N_STEPS, n_points) when n_envs is 0.
    """
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        rigid_options=gs.options.RigidOptions(
            enable_collision=True,
            enable_self_collision=True,
        ),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    obj = scene.add_entity(morph=cyl_morphs)
    hand = scene.add_entity(
        gs.morphs.URDF(file=URDF_PATH, merge_fixed_links=False, fixed=True, pos=HAND_POS, euler=HAND_EULER),
        vis_mode="collision",
    )

    with open(TACTILE_GRID_PATH) as f:
        links_data = json.load(f)["links"]
    sensors = {}
    for link_name in TEST_LINKS:
        sensors[link_name] = scene.add_sensor(
            gs.sensors.TactileField(
                entity_idx=hand.idx,
                link_idx_local=hand.get_link(link_name).idx_local,
                indenter_entity_idx=obj.idx,
                indenter_link_idx_local=0,
                tactile_points_local=np.array(links_data[link_name]["points"], dtype=np.float32),
                kn=KN,
            )
        )

    scene.build(n_envs=n_envs)
    motors_dof_idx = [hand.get_joint(name).dofs_idx_local[0] for name in JOINTS_NAME]
    hand.set_dofs_kp(np.full(len(motors_dof_idx), 20.0), motors_dof_idx)
    hand.set_dofs_kv(np.full(len(motors_dof_idx), 1.0), motors_dof_idx)

    history = {link_name: [] for link_name in TEST_LINKS}
    for i in range(N_STEPS):
        hand.control_dofs_position(POSE + i * DELTA_POSE, motors_dof_idx)
        scene.step()
        for link_name, sensor in sensors.items():
            force = sensor.read()
            history[link_name].append(force.reshape(*force.shape[:-1], -1, 3).norm(dim=-1).cpu())

    scene.destroy()
    return {link_name: torch.stack(forces) for link_name, forces in history.items()}


def variant_env_blocks(n_envs, n_variants):
    """Envs assigned to each variant (balanced contiguous blocks, same rule as the rigid solver)."""
    base, extra = divmod(n_envs, n_variants)
    sizes = [base + 1] * extra + [base] * (n_variants - extra)
    bounds = np.cumsum([0, *sizes])
    return [list(range(bounds[v], bounds[v + 1])) for v in range(n_variants)]


@pytest.mark.parametrize("n_envs", [6, 30])
def test_heterogeneous_variants(n_envs):
    """Envs sharing a cylinder variant get identical forces; different radii give different forces."""
    morphs = [gs.morphs.Cylinder(radius=r, height=CYL_HEIGHT, pos=CYL_POS) for r in RADII]
    forces = run_sim(morphs, n_envs)
    blocks = variant_env_blocks(n_envs, len(RADII))

    within_variant_error = max(
        (forces[link_name][:, envs[1:]] - forces[link_name][:, envs[:1]]).abs().max().item()
        for link_name in TEST_LINKS
        for envs in blocks
    )
    variant_totals = [sum(forces[link_name][:, envs[0]].sum().item() for link_name in TEST_LINKS) for envs in blocks]
    between_variant_diff = max(variant_totals) - min(variant_totals)

    assert max(variant_totals) > 0.1, f"no contact detected: {variant_totals}"
    assert within_variant_error < 1e-5
    assert between_variant_diff > 0.1, f"all variants have the same total force: {variant_totals}"


def test_single_vs_parallel_fixed():
    """Single-env runs match the heterogeneous parallel run, with FIXED cylinders.

    With fixed cylinders, the hand (position-controlled, fixed base) follows the same
    trajectory regardless of batch size. Any tactile force difference is a sensor bug.
    """
    morphs = [gs.morphs.Cylinder(radius=r, height=CYL_HEIGHT, pos=CYL_POS, fixed=True) for r in RADII]
    baselines = [run_sim(morph, n_envs=0) for morph in morphs]
    parallel = run_sim(morphs, n_envs=len(RADII))

    max_error = max(
        (baselines[env_idx][link_name] - parallel[link_name][:, env_idx]).abs().max().item()
        for link_name in TEST_LINKS
        for env_idx in range(len(RADII))
    )
    total_force = sum(parallel[link_name].sum().item() for link_name in TEST_LINKS)

    assert total_force > 0.1, "no contact detected"
    assert max_error < 1e-3
