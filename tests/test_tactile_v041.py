"""
Tactile sensor correctness tests for Genesis v0.4.1 merge.

Uses the same Wuji hand + cylinder setup as examples/tactile/tactile_field_hand.py
with actual tactile grid points from the JSON file.

NOTE: The batched physics solver produces different trajectories depending on n_envs
(batch size affects contact solver numerics). Therefore all correctness comparisons
are WITHIN a single simulation run — comparing envs assigned to the same variant.

Tests:
1. Same-variant consistency: 6 envs, 3 variants → env pairs with same variant match exactly
2. Variant differentiation: different cylinder radii produce different tactile forces
3. Scaling test: 30 envs, 3 variants → 10 envs/variant all match
4. Speed benchmark: 10 vs 100 envs
"""

import json
import time

import numpy as np
import torch

import genesis as gs

TACTILE_GRID_PATH = "examples/tactile/full_hand_tactile.json"
URDF_PATH = "genesis/assets/urdf/wujihand-urdf/urdf/right.urdf"

HAND_POS = (0, 0.1, 0.1)
HAND_EULER = (90, 0, 0)
CYL_POS = (0.03, -0.01, 0.1)
CYL_HEIGHT = 0.2
KN = 2000.0

JOINTS_NAME = (
    "finger1_joint1", "finger1_joint2", "finger1_joint3", "finger1_joint4",
    "finger2_joint1", "finger2_joint2", "finger2_joint3", "finger2_joint4",
    "finger3_joint1", "finger3_joint2", "finger3_joint3", "finger3_joint4",
    "finger4_joint1", "finger4_joint2", "finger4_joint3", "finger4_joint4",
    "finger5_joint1", "finger5_joint2", "finger5_joint3", "finger5_joint4",
)

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
TEST_LINKS = [
    "finger5_link3",
    "finger4_link3",
    "finger3_link3",
    "finger2_link3",
]


def load_tactile_grid():
    with open(TACTILE_GRID_PATH) as f:
        data = json.load(f)
    return data["links"]


def make_hand_morph():
    return gs.morphs.URDF(
        file=URDF_PATH, merge_fixed_links=False, fixed=True,
        pos=HAND_POS, euler=HAND_EULER,
    )


def setup_hand_control(hand):
    motors_dof_idx = [hand.get_joint(name).dofs_idx_local[0] for name in JOINTS_NAME]
    hand.set_dofs_kp(np.array([20] * len(motors_dof_idx)), motors_dof_idx)
    hand.set_dofs_kv(np.array([1] * len(motors_dof_idx)), motors_dof_idx)
    return motors_dof_idx


def run_parallel_env(morphs, links_data, test_link_names, n_steps, n_envs):
    """Run parallel heterogeneous simulation. Returns dict of link_name -> list of (n_envs, n_pts) tensors."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        rigid_options=gs.options.RigidOptions(
            enable_collision=True, enable_self_collision=True,
        ),
        show_viewer=False,
    )

    scene.add_entity(gs.morphs.Plane())
    obj = scene.add_entity(morph=morphs)
    hand = scene.add_entity(make_hand_morph(), vis_mode="collision")

    sensors = {}
    for link_name in test_link_names:
        ld = links_data[link_name]
        local_positions = np.array(ld["points"], dtype=np.float32)
        sensor = scene.add_sensor(
            gs.sensors.TactileField(
                entity_idx=hand.idx,
                link_idx_local=ld["link_idx_local"],
                indenter_entity_idx=obj.idx,
                indenter_link_idx_local=0,
                tactile_points_local=local_positions,
                kn=KN,
            )
        )
        sensors[link_name] = (sensor, len(local_positions))

    scene.build(n_envs=n_envs)
    motors_dof_idx = setup_hand_control(hand)

    forces_history = {name: [] for name in test_link_names}
    for i in range(n_steps):
        hand.control_dofs_position(POSE + i * DELTA_POSE, motors_dof_idx)
        scene.step()
        for link_name, (sensor, n_pts) in sensors.items():
            raw = sensor.read()
            force_3d = raw.reshape(n_envs, n_pts, 3)
            magnitudes = torch.norm(force_3d, dim=-1)
            forces_history[link_name].append(magnitudes.detach().cpu().clone())

    scene.destroy()
    return forces_history


def compute_variant_env_mapping(n_envs, n_variants):
    """Compute balanced block assignment: variant -> list of env indices."""
    base = n_envs // n_variants
    extra = n_envs % n_variants
    sizes = np.concatenate([np.full(extra, base + 1, dtype=int),
                            np.full(n_variants - extra, base, dtype=int)])
    mapping = {}
    cursor = 0
    for v, sz in enumerate(sizes):
        mapping[v] = list(range(cursor, cursor + sz))
        cursor += sz
    return mapping


def test_same_variant_consistency():
    """Test 1: 6 envs, 3 variants → env pairs with same variant produce identical forces."""
    print("\n" + "=" * 70)
    print("TEST 1: Same-variant consistency (6 envs, 3 variants)")
    print("=" * 70)

    radii = [0.008, 0.012, 0.016]
    n_envs = 6
    n_steps = 250
    links_data = load_tactile_grid()
    morphs = [gs.morphs.Cylinder(radius=r, height=CYL_HEIGHT, pos=CYL_POS) for r in radii]

    print(f"\nRunning {n_envs}-env simulation...")
    forces = run_parallel_env(morphs, links_data, TEST_LINKS, n_steps, n_envs)

    # Balanced block: 6 envs, 3 variants → {0: [0,1], 1: [2,3], 2: [4,5]}
    mapping = compute_variant_env_mapping(n_envs, len(radii))
    print(f"Variant-to-env mapping: {mapping}")

    print("\nChecking envs within same variant produce identical forces...")
    max_error = 0.0
    for link_name in TEST_LINKS:
        link_err = 0.0
        for step in range(n_steps):
            for v, envs in mapping.items():
                ref = forces[link_name][step][envs[0]]
                for e in envs[1:]:
                    err = torch.abs(ref - forces[link_name][step][e]).max().item()
                    link_err = max(link_err, err)
        max_error = max(max_error, link_err)
        print(f"  {link_name}: max within-variant error = {link_err:.8f}")

    print(f"\n  Overall max error: {max_error:.8f}")
    passed = max_error < 1e-5
    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


def test_variant_differentiation():
    """Test 2: Different cylinder radii must produce different tactile force profiles."""
    print("\n" + "=" * 70)
    print("TEST 2: Variant differentiation (different radii → different forces)")
    print("=" * 70)

    radii = [0.008, 0.012, 0.016]
    n_envs = 6
    n_steps = 250
    links_data = load_tactile_grid()
    morphs = [gs.morphs.Cylinder(radius=r, height=CYL_HEIGHT, pos=CYL_POS) for r in radii]

    print(f"\nRunning {n_envs}-env simulation...")
    forces = run_parallel_env(morphs, links_data, TEST_LINKS, n_steps, n_envs)

    mapping = compute_variant_env_mapping(n_envs, len(radii))

    # Compute total force per variant (use first env of each variant)
    variant_totals = {v: 0.0 for v in range(len(radii))}
    for link_name in TEST_LINKS:
        for step in range(n_steps):
            for v, envs in mapping.items():
                variant_totals[v] += forces[link_name][step][envs[0]].sum().item()

    for v in range(len(radii)):
        print(f"  Variant {v} (radius={radii[v]}): total force = {variant_totals[v]:.4f}")

    has_contact = any(t > 0.1 for t in variant_totals.values())
    # Check at least two variants differ significantly
    vals = list(variant_totals.values())
    max_diff = max(abs(vals[i] - vals[j]) for i in range(len(vals)) for j in range(i + 1, len(vals)))
    forces_differ = max_diff > 0.1

    print(f"\n  Max difference between variants: {max_diff:.4f}")
    print(f"  Has contact: {has_contact}")

    passed = has_contact and forces_differ
    if passed:
        print("  PASS: Contact detected and variants produce different forces")
    else:
        reasons = []
        if not has_contact:
            reasons.append("no contact detected")
        if not forces_differ:
            reasons.append("all variants have identical forces")
        print(f"  FAIL: {', '.join(reasons)}")
    return passed


def test_scaling():
    """Test 3: 30 envs, 3 variants → 10 envs/variant, all same-variant envs match."""
    print("\n" + "=" * 70)
    print("TEST 3: Scaling (30 envs, 3 variants, 10 envs/variant)")
    print("=" * 70)

    radii = [0.008, 0.012, 0.016]
    n_envs = 30
    n_steps = 250
    links_data = load_tactile_grid()
    morphs = [gs.morphs.Cylinder(radius=r, height=CYL_HEIGHT, pos=CYL_POS) for r in radii]

    print(f"\nRunning {n_envs}-env simulation...")
    forces = run_parallel_env(morphs, links_data, TEST_LINKS, n_steps, n_envs)

    mapping = compute_variant_env_mapping(n_envs, len(radii))
    print(f"Variant-to-env mapping: v0→{len(mapping[0])} envs, v1→{len(mapping[1])} envs, v2→{len(mapping[2])} envs")

    print("\nChecking all same-variant envs match...")
    max_error = 0.0
    for link_name in TEST_LINKS:
        link_err = 0.0
        for step in range(n_steps):
            for v, envs in mapping.items():
                ref = forces[link_name][step][envs[0]]
                for e in envs[1:]:
                    err = torch.abs(ref - forces[link_name][step][e]).max().item()
                    link_err = max(link_err, err)
        max_error = max(max_error, link_err)
        print(f"  {link_name}: max within-variant error = {link_err:.8f}")

    # Also verify variants differ
    variant_totals = {v: 0.0 for v in range(len(radii))}
    for link_name in TEST_LINKS:
        for step in range(n_steps):
            for v, envs in mapping.items():
                variant_totals[v] += forces[link_name][step][envs[0]].sum().item()

    for v in range(len(radii)):
        print(f"  Variant {v} (radius={radii[v]}): total force = {variant_totals[v]:.4f}")

    vals = list(variant_totals.values())
    max_diff = max(abs(vals[i] - vals[j]) for i in range(len(vals)) for j in range(i + 1, len(vals)))

    print(f"\n  Max within-variant error: {max_error:.8f}")
    print(f"  Max between-variant difference: {max_diff:.4f}")

    within_ok = max_error < 1e-5
    forces_differ = max_diff > 0.1

    passed = within_ok and forces_differ
    if passed:
        print("  PASS")
    else:
        reasons = []
        if not within_ok:
            reasons.append(f"within-variant mismatch (err={max_error})")
        if not forces_differ:
            reasons.append("variants identical")
        print(f"  FAIL: {', '.join(reasons)}")
    return passed


def test_speed():
    """Test 4: Speed benchmark — sensor time should scale sub-linearly with envs."""
    print("\n" + "=" * 70)
    print("TEST 4: Speed benchmark (10 vs 100 envs)")
    print("=" * 70)

    radii = [0.008, 0.012, 0.016]
    n_steps = 100
    links_data = load_tactile_grid()

    for n_envs in [10, 100]:
        morphs = [gs.morphs.Cylinder(radius=radii[i % len(radii)], height=CYL_HEIGHT, pos=CYL_POS)
                   for i in range(min(n_envs, len(radii)))]

        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=0.01),
            rigid_options=gs.options.RigidOptions(
                enable_collision=True, enable_self_collision=True,
            ),
            show_viewer=False,
        )

        scene.add_entity(gs.morphs.Plane())
        obj = scene.add_entity(morph=morphs)
        hand = scene.add_entity(make_hand_morph(), vis_mode="collision")

        sensors_list = []
        for link_name in TEST_LINKS:
            ld = links_data[link_name]
            local_positions = np.array(ld["points"], dtype=np.float32)
            sensor = scene.add_sensor(
                gs.sensors.TactileField(
                    entity_idx=hand.idx,
                    link_idx_local=ld["link_idx_local"],
                    indenter_entity_idx=obj.idx,
                    indenter_link_idx_local=0,
                    tactile_points_local=local_positions,
                    kn=KN,
                )
            )
            sensors_list.append(sensor)

        scene.build(n_envs=n_envs)
        motors_dof_idx = setup_hand_control(hand)

        # Warmup
        for i in range(10):
            hand.control_dofs_position(POSE + i * DELTA_POSE, motors_dof_idx)
            scene.step()
            for s in sensors_list:
                s.read()

        # Timed
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(10, 10 + n_steps):
            hand.control_dofs_position(POSE + i * DELTA_POSE, motors_dof_idx)
            scene.step()
            for s in sensors_list:
                s.read()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        print(f"  {n_envs:3d} envs: {elapsed / n_steps * 1000:.2f} ms/step ({elapsed:.2f}s total)")
        scene.destroy()


def make_fixed_cyl_morph(radius):
    """Create a fixed cylinder morph."""
    return gs.morphs.Cylinder(radius=radius, height=CYL_HEIGHT, pos=CYL_POS, fixed=True)


def run_single_env_fixed_cyl(cyl_morph, links_data, test_link_names, n_steps):
    """Run a single-env simulation with a FIXED cylinder. Returns dict of link_name -> list of (n_pts,) tensors."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        rigid_options=gs.options.RigidOptions(
            enable_collision=True, enable_self_collision=True,
        ),
        show_viewer=False,
    )

    scene.add_entity(gs.morphs.Plane())
    obj = scene.add_entity(cyl_morph)
    hand = scene.add_entity(make_hand_morph(), vis_mode="collision")

    sensors = {}
    for link_name in test_link_names:
        ld = links_data[link_name]
        local_positions = np.array(ld["points"], dtype=np.float32)
        sensor = scene.add_sensor(
            gs.sensors.TactileField(
                entity_idx=hand.idx,
                link_idx_local=ld["link_idx_local"],
                indenter_entity_idx=obj.idx,
                indenter_link_idx_local=0,
                tactile_points_local=local_positions,
                kn=KN,
            )
        )
        sensors[link_name] = (sensor, len(local_positions))

    scene.build()
    motors_dof_idx = setup_hand_control(hand)

    forces_history = {name: [] for name in test_link_names}
    for i in range(n_steps):
        hand.control_dofs_position(POSE + i * DELTA_POSE, motors_dof_idx)
        scene.step()
        for link_name, (sensor, n_pts) in sensors.items():
            raw = sensor.read()
            force_3d = raw.reshape(n_pts, 3)
            magnitudes = torch.norm(force_3d, dim=-1)
            forces_history[link_name].append(magnitudes.detach().cpu().clone())

    scene.destroy()
    return forces_history


def run_parallel_env_fixed_cyl(morphs, links_data, test_link_names, n_steps, n_envs):
    """Run parallel heterogeneous simulation with FIXED cylinders."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        rigid_options=gs.options.RigidOptions(
            enable_collision=True, enable_self_collision=True,
        ),
        show_viewer=False,
    )

    scene.add_entity(gs.morphs.Plane())
    obj = scene.add_entity(morph=morphs)
    hand = scene.add_entity(make_hand_morph(), vis_mode="collision")

    sensors = {}
    for link_name in test_link_names:
        ld = links_data[link_name]
        local_positions = np.array(ld["points"], dtype=np.float32)
        sensor = scene.add_sensor(
            gs.sensors.TactileField(
                entity_idx=hand.idx,
                link_idx_local=ld["link_idx_local"],
                indenter_entity_idx=obj.idx,
                indenter_link_idx_local=0,
                tactile_points_local=local_positions,
                kn=KN,
            )
        )
        sensors[link_name] = (sensor, len(local_positions))

    scene.build(n_envs=n_envs)
    motors_dof_idx = setup_hand_control(hand)

    forces_history = {name: [] for name in test_link_names}
    for i in range(n_steps):
        hand.control_dofs_position(POSE + i * DELTA_POSE, motors_dof_idx)
        scene.step()
        for link_name, (sensor, n_pts) in sensors.items():
            raw = sensor.read()
            force_3d = raw.reshape(n_envs, n_pts, 3)
            magnitudes = torch.norm(force_3d, dim=-1)
            forces_history[link_name].append(magnitudes.detach().cpu().clone())

    scene.destroy()
    return forces_history


def test_single_vs_parallel_fixed():
    """Test 4: Single-env baselines vs parallel heterogeneous with FIXED cylinders.

    With fixed cylinders, the hand (position-controlled, fixed base) follows the same
    trajectory regardless of batch size. Any tactile force difference is a sensor bug.
    """
    print("\n" + "=" * 70)
    print("TEST 4: Single-env vs parallel (fixed cylinders, 3 radii)")
    print("=" * 70)

    radii = [0.008, 0.012, 0.016]
    n_steps = 250
    links_data = load_tactile_grid()

    # Single-env baselines (fixed cylinders)
    print("\nRunning single-env baselines (fixed cylinders)...")
    baselines = []
    for idx, r in enumerate(radii):
        print(f"  Variant {idx}: radius={r}")
        morph = make_fixed_cyl_morph(r)
        forces = run_single_env_fixed_cyl(morph, links_data, TEST_LINKS, n_steps)
        baselines.append(forces)

    # Parallel heterogeneous (fixed cylinders)
    print("\nRunning 3-env heterogeneous simulation (fixed cylinders)...")
    morphs = [make_fixed_cyl_morph(r) for r in radii]
    parallel = run_parallel_env_fixed_cyl(morphs, links_data, TEST_LINKS, n_steps, n_envs=3)

    # Compare
    print("\nComparing single-env vs parallel forces...")
    max_error = 0.0
    total_force = 0.0
    for link_name in TEST_LINKS:
        link_max_err = 0.0
        link_total = 0.0
        for step in range(n_steps):
            for env_idx in range(3):
                baseline = baselines[env_idx][link_name][step]
                par = parallel[link_name][step][env_idx]
                err = torch.abs(baseline - par).max().item()
                link_max_err = max(link_max_err, err)
                link_total += par.sum().item()
        max_error = max(max_error, link_max_err)
        total_force += link_total
        print(f"  {link_name}: max_error={link_max_err:.6f}, total_force={link_total:.2f}")

    print(f"\n  Overall max error: {max_error:.6f}")
    print(f"  Overall total force: {total_force:.2f}")

    passed = max_error < 1e-3 and total_force > 0.1
    if passed:
        print("  PASS: Single-env and parallel forces match!")
    else:
        if total_force < 0.1:
            print("  FAIL: No significant contact detected")
        else:
            print(f"  FAIL: Force mismatch (max_error={max_error:.6f})")
    return passed


if __name__ == "__main__":
    gs.init(backend=gs.gpu, logging_level="warning")

    print("=" * 70)
    print("Tactile Sensor Correctness Tests (Genesis v0.4.1)")
    print("=" * 70)

    r1 = test_same_variant_consistency()
    r2 = test_variant_differentiation()
    r3 = test_scaling()
    r4 = test_single_vs_parallel_fixed()
    test_speed()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    results = [
        ("Test 1: Same-variant consistency (6 envs)", r1),
        ("Test 2: Variant differentiation", r2),
        ("Test 3: Scaling (30 envs)", r3),
        ("Test 4: Single-env vs parallel (fixed cyl)", r4),
    ]
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\nAll tests PASSED!")
    else:
        print("\nSome tests FAILED!")
