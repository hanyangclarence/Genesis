import argparse
import json
from pathlib import Path

import numpy as np
import torch

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tactile-grid", type=str, default="examples/sensors/merged_tactile_grid.json",
                        help="Path to tactile grid JSON file")
    parser.add_argument("--marker-size", type=float, default=0.0005,
                        help="Size of tactile point markers in meters")
    parser.add_argument("--save-render", type=str, default=None,
                        help="Path to save rendered scene video (camera view)")
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(backend=gs.cpu)

    ########################## create a scene ##########################
    viewer_options = gs.options.ViewerOptions(
        camera_pos=(0, -0.5, 0.5),
        camera_lookat=(0.0, 0.0, 0.0),
        camera_fov=40,
        max_FPS=5,
    )

    scene = gs.Scene(
        viewer_options=viewer_options,
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        # rigid_options=gs.options.RigidOptions(
        #     enable_self_collision=False,
        # ),
        show_viewer=True,
    )

    ########################## entities ##########################
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    # obj = scene.add_entity(
    #     gs.morphs.Cylinder(
    #         radius=0.02, height=0.2,
    #         pos=(0.03, 0.01, 0.1),
    #     ),
    # )
    wuji_hand = scene.add_entity(
        gs.morphs.URDF(
            file="/home/hanyang/code/humanoid/GenesisPlayground/assets/robot/xarm/wujihand_left_v2.urdf",
            merge_fixed_links=False,
            fixed=True,
            pos=(0, 0.1, 0.1),
            euler=(90, 0, 0),
        ),
        vis_mode="collision"
    )

    ########################## Load tactile grid ##########################
    tactile_grid_path = Path(args.tactile_grid)
    if tactile_grid_path.exists():
        print(f"\n{'='*70}")
        print(f"Loading tactile grid from: {tactile_grid_path}")
        print(f"{'='*70}")

        with open(tactile_grid_path, 'r') as f:
            tactile_data = json.load(f)

        # Store tactile points data for each link
        tactile_points = {}  # link_name -> list of local positions
        tactile_spheres = {}  # link_name -> list of sphere entities

        links_data = tactile_data.get('links', {})
        total_markers = 0

        for link_name, link_data in links_data.items():
            points = link_data.get('points', [])
            if not points:
                continue

            # Store local positions
            local_positions = []
            spheres = []

            for point_data in points:
                # Handle list format [x, y, z]
                if isinstance(point_data, list):
                    local_pos = np.array(point_data, dtype=np.float32)
                    local_positions.append(local_pos)

                    # Create sphere marker (position will be updated each step)
                    sphere = scene.add_entity(
                        gs.morphs.Sphere(
                            radius=args.marker_size,
                            pos=(0, 0, 0),  # Will be updated
                            fixed=True,
                            collision=False,
                        )
                    )
                    spheres.append(sphere)
                    total_markers += 1

            tactile_points[link_name] = np.array(local_positions)
            tactile_spheres[link_name] = spheres
            print(f"  {link_name}: {len(local_positions)} tactile points")

        print(f"\nTotal tactile markers: {total_markers}")
        print(f"Marker size: {args.marker_size*1000:.1f}mm")
    else:
        print(f"\nWarning: Tactile grid file not found: {tactile_grid_path}")
        print("Continuing without tactile visualization...")
        tactile_points = {}
        tactile_spheres = {}

    ########################## camera setup ##########################
    # Add camera for rendering if requested (must be before scene.build())
    cam = None
    if args.save_render:
        cam = scene.add_camera(
            res=(1280, 720),
            pos=(0.3, 0.3, 0.3),
            lookat=(0, 0, 0.1),
            fov=40,
            GUI=False,  # Headless camera
        )

    ########################## build ##########################
    scene.build()

    # Wuji Hand joints: 5 fingers, each with 4 joints
    joints_name = (
        "finger1_joint1", "finger1_joint2", "finger1_joint3", "finger1_joint4",  # Thumb
        "finger2_joint1", "finger2_joint2", "finger2_joint3", "finger2_joint4",  # Index
        "finger3_joint1", "finger3_joint2", "finger3_joint3", "finger3_joint4",  # Middle
        "finger4_joint1", "finger4_joint2", "finger4_joint3", "finger4_joint4",  # Ring
        "finger5_joint1", "finger5_joint2", "finger5_joint3", "finger5_joint4",  # Pinky
    )
    motors_dof_idx = [wuji_hand.get_joint(name).dofs_idx_local[0] for name in joints_name]

    # Optional: set control gains
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
    ]) * 0
    

    def update_tactile_markers():
        """Update tactile marker positions based on current link transforms."""
        for link_name, local_positions in tactile_points.items():
            try:
                # Get current link transform
                link = wuji_hand.get_link(link_name)
                link_pos = link.get_pos()
                link_quat = link.get_quat()

                # Convert to torch tensors
                link_pos_torch = torch.tensor(link_pos, dtype=gs.tc_float)
                link_quat_torch = torch.tensor(link_quat, dtype=gs.tc_float)

                # Get spheres for this link
                spheres = tactile_spheres[link_name]

                # Transform each local point to world coordinates
                for j, local_pos in enumerate(local_positions):
                    local_pos_torch = torch.tensor(local_pos, dtype=gs.tc_float)

                    # Transform: local frame -> world frame
                    world_pos_torch = gs.utils.geom.transform_by_trans_quat(
                        local_pos_torch, link_pos_torch, link_quat_torch
                    )
                    world_pos = world_pos_torch.cpu().numpy()

                    # Update sphere position
                    spheres[j].set_pos(world_pos)

            except Exception as e:
                print(f"Warning: Could not update markers for {link_name}: {e}")

    # Start camera recording if camera was added
    if cam is not None:
        cam.start_recording()

    def grasp():
        # PD control
        for i in range(10000):
            wuji_hand.control_dofs_position(
                pose + i * delta_pose,
                motors_dof_idx,
            )
            scene.step(refresh_visualizer=False)

            # Render camera frame if recording
            if cam is not None:
                cam.render()

            # Update tactile marker positions
            if tactile_points:
                update_tactile_markers()

        # Stop camera recording if it was started
        if cam is not None:
            cam.stop_recording(save_to_filename=args.save_render, fps=30)
            print(f"\nVideo saved to: {args.save_render}")

    import threading
    threading.Thread(target=grasp).start()
    scene.viewer.run()

if __name__ == "__main__":
    main()