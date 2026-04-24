"""
Full hand tactile sensing demo using TactileFieldSensor for Genesis v0.3.10.

This script demonstrates tactile sensors on multiple links of the Wuji hand,
using precomputed tactile point grids from a JSON file.

Visualizes tactile forces as a 24x32 image using the TactileVisualizer.

Usage:
    # With tactile viewer
    python tactile_field_hand.py --visualize

    # Without tactile viewer
    python tactile_field_hand.py --no-visualize

    # Specific links only
    python tactile_field_hand.py --sensor-links palm_link,finger2_link3
"""
import argparse

import numpy as np
import torch

import genesis as gs
from genesis.vis import TactileVisualizer


def main():
    parser = argparse.ArgumentParser(description="Full Hand Tactile Sensing Demo")
    parser.add_argument("--tactile-grid", type=str,
                        default="examples/tactile/full_hand_tactile.json",
                        help="Path to tactile grid JSON file")
    parser.add_argument("--pixel-mapping", type=str,
                        default="examples/tactile/tactile_pixel_mapping.json",
                        help="Path to tactile pixel mapping JSON file")
    parser.add_argument("--sensor-links", type=str, default=None,
                        help="Comma-separated list of link names to add tactile sensors to (default: all links in grid file)")
    parser.add_argument("--visualize", action="store_true", default=True,
                        help="Show tactile map visualization")
    parser.add_argument("--no-visualize", action="store_false", dest="visualize",
                        help="Disable tactile visualization")
    parser.add_argument("--kn", type=float, default=2000.0,
                        help="Normal stiffness coefficient")
    parser.add_argument("--num-steps", type=int, default=300,
                        help="Number of simulation steps")
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(backend=gs.gpu, logging_level="info")

    ########################## create a scene ##########################
    viewer_options = gs.options.ViewerOptions(
        camera_pos=(0, -0.5, 0.5),
        camera_lookat=(0.0, 0.0, 0.0),
        camera_fov=40,
        max_FPS=15,
    )

    scene = gs.Scene(
        viewer_options=viewer_options,
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=True,
            enable_self_collision=True,
        ),
        show_viewer=args.visualize,
    )

    ########################## entities ##########################
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    # First object - cylinder
    obj1 = scene.add_entity(
        gs.morphs.Cylinder(
            radius=0.01, height=0.2,
            pos=(0.03, -0.01, 0.1),
        ),
    )
    # Second object - demonstrates multi-indenter support
    obj2 = scene.add_entity(
        gs.morphs.Cylinder(
            radius=0.01, height=0.2,
            pos=(0.03, 0.025, 0.1),
        ),
    )
    wuji_hand = scene.add_entity(
        gs.morphs.URDF(
            file="/home/hanyang/code/humanoid/GenesisPlayground/assets/robot/xarm/wujihand_left_v5.urdf",
            merge_fixed_links=False,
            fixed=True,
            pos=(0.07, 0.13, 0.1),
            euler=(90, 180, 0),
        ),
        vis_mode="collision"
    )

    ########################## Tactile Visualizer ##########################
    print(f"\n{'='*70}")
    print("Initializing TactileVisualizer...")
    print(f"{'='*70}")

    tactile_vis = TactileVisualizer(
        tactile_grid_path=args.tactile_grid,
        pixel_mapping_path=args.pixel_mapping,
        num_envs=1,
        show_viewer=args.visualize,
    )

    links_data = tactile_vis.links_data
    print(f"Image shape: {tactile_vis.image_shape}")
    print(f"Total tactile points: {tactile_vis.num_tactile_points}")
    print(f"Pixels with mappings: {tactile_vis.num_pixels_with_points}")

    # Parse sensor links - use all available links if not specified
    # IMPORTANT: Keep the same order as in the tactile grid JSON to match global indices
    if args.sensor_links is None:
        sensor_link_names = list(links_data.keys())  # Preserves JSON order
        print(f"\nNo sensor links specified, using all available links from tactile grid")
    else:
        # Filter but preserve original order from JSON
        requested_links = set(s.strip() for s in args.sensor_links.split(',') if s.strip())
        sensor_link_names = [link for link in links_data.keys() if link in requested_links]

    print(f"Configuring tactile sensors for links (in JSON order): {sensor_link_names}")

    # Validate sensor links and store tactile points
    tactile_points = {}  # link_name -> np.array of local positions

    for link_name in sensor_link_names:
        if link_name not in links_data:
            gs.raise_exception(f"Link '{link_name}' not found in tactile grid JSON")

        link_data = links_data[link_name]
        points = link_data.get('points', [])

        if not points:
            gs.raise_exception(f"No tactile points found for link '{link_name}'")

        # Store local positions
        local_positions = np.array(points, dtype=np.float32)
        tactile_points[link_name] = local_positions

        print(f"  {link_name}: {len(local_positions)} tactile points")

    ########################## Add TactileFieldSensors ##########################
    print(f"\n{'='*70}")
    print("Adding TactileFieldSensors")
    print(f"{'='*70}")

    sensors = {}  # link_name -> sensor object
    sensor_configs = {}  # link_name -> config dict

    for link_name in sensor_link_names:
        link_data = links_data[link_name]
        link_idx_local = link_data['link_idx_local']
        num_points = link_data['num_points']
        local_positions = tactile_points[link_name]

        print(f"\nSensor for {link_name}:")
        print(f"  Link index (local): {link_idx_local}")
        print(f"  Tactile points: {num_points}")

        # Create TactileFieldSensor with custom tactile points
        # Multi-indenter: pass list of entity indices for both objects
        sensor = scene.add_sensor(
            gs.sensors.TactileField(
                entity_idx=wuji_hand.idx,
                link_idx_local=link_idx_local,
                indenter_entity_idx=[obj1.idx, obj2.idx],  # Multiple indenters!
                indenter_link_idx_local=[0, 0],
                tactile_points_local=local_positions,  # Use custom points!
                kn=args.kn,
            )
        )

        sensors[link_name] = sensor
        sensor_configs[link_name] = {
            'num_points': num_points,
            'link_idx_local': link_idx_local,
        }

        print(f"  TactileFieldSensor added")

    ########################## build ##########################
    print(f"\n{'='*70}")
    print("Building scene...")
    print(f"{'='*70}")
    scene.build()
    print("Scene built")

    # Wuji Hand joints: 5 fingers, each with 4 joints
    joints_name = (
        "finger1_joint1", "finger1_joint2", "finger1_joint3", "finger1_joint4",  # Thumb
        "finger2_joint1", "finger2_joint2", "finger2_joint3", "finger2_joint4",  # Index
        "finger3_joint1", "finger3_joint2", "finger3_joint3", "finger3_joint4",  # Middle
        "finger4_joint1", "finger4_joint2", "finger4_joint3", "finger4_joint4",  # Ring
        "finger5_joint1", "finger5_joint2", "finger5_joint3", "finger5_joint4",  # Pinky
    )
    motors_dof_idx = [wuji_hand.get_joint(name).dofs_idx_local[0] for name in joints_name]

    # Set control gains
    wuji_hand.set_dofs_kp(
        np.array([20] * len(motors_dof_idx)),
        motors_dof_idx,
    )
    wuji_hand.set_dofs_kv(
        np.array([1] * len(motors_dof_idx)),
        motors_dof_idx,
    )

    pose = np.array([
        0.7, -0.16, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
    ])
    delta_pose = np.array([
        # Finger 1 (Thumb)
        0.00, 0.00, 0.00, 0.00,
        # Finger 2 (Index)
        0.01, 0.00, 0.01, 0.01,
        # Finger 3 (Middle)
        0.01, 0.00, 0.01, 0.01,
        # Finger 4 (Ring)
        0.01, 0.00, 0.01, 0.01,
        # Finger 5 (Pinky)
        0.01, 0.00, 0.01, 0.01,
    ])

    ########################## Helper functions ##########################
    device = tactile_vis.device

    def read_all_tactile_forces():
        """Read all sensors and return concatenated force magnitudes as (num_total_points,) tensor."""
        all_magnitudes = []

        # Iterate in JSON order to match global indices
        for link_name in links_data.keys():
            if link_name not in sensors:
                # Link not in sensor list, fill with zeros
                num_pts = links_data[link_name]['num_points']
                all_magnitudes.append(torch.zeros(num_pts, device=device))
                continue

            sensor = sensors[link_name]
            config = sensor_configs[link_name]
            num_pts = config['num_points']

            # Read sensor data and compute magnitudes
            force_field_full = sensor.read()
            force_field_3d = force_field_full.reshape(num_pts, 3)
            force_magnitudes = torch.norm(force_field_3d, dim=-1)

            all_magnitudes.append(force_magnitudes)

        return torch.cat(all_magnitudes, dim=0)

    def print_force_summary():
        """Print summary of tactile forces."""
        print("\n" + "="*70)
        print("Tactile Force Summary")
        print("="*70)

        for link_name in sensor_link_names:
            sensor = sensors[link_name]
            force_field_full = sensor.read()

            config = sensor_configs[link_name]
            num_points = config['num_points']

            force_field_3d = force_field_full.reshape(num_points, 3)
            force_magnitudes = torch.norm(force_field_3d, dim=-1)

            total_force = force_magnitudes.sum().item()
            max_force = force_magnitudes.max().item()
            mean_force = force_magnitudes.mean().item()

            print(f"{link_name}:")
            print(f"  Total force: {total_force:.2f} N")
            print(f"  Max force:   {max_force:.2f} N")
            print(f"  Mean force:  {mean_force:.2f} N")

    ########################## Simulation loop ##########################
    print(f"\n{'='*70}")
    print(f"Running simulation for {args.num_steps} steps...")
    print(f"{'='*70}")

    for i in range(args.num_steps):
        wuji_hand.control_dofs_position(
            pose + i * delta_pose,
            motors_dof_idx,
        )
        scene.step()

        # Read tactile forces and update visualization
        force_magnitudes = read_all_tactile_forces()  # (N,)
        tactile_maps = tactile_vis.update(force_magnitudes.unsqueeze(0))  # (1, N) -> (1, H, W)

        # Print progress every 50 steps
        if i % 50 == 0:
            max_force = tactile_maps[0].max().item()
            print(f"Step {i:4d}/{args.num_steps}: Max force = {max_force:.2f} N")

    # Print final summary
    print_force_summary()

    # Show viewer (blocking) if visualization enabled
    tactile_vis.show()

    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
