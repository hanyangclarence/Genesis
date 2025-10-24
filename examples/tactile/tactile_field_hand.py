"""
Full hand tactile sensing demo using TactileFieldSensor.

This script extends the basic allegro.py demo to include tactile sensors
on specified links of the Wuji hand, using the precomputed tactile point
grids from the JSON file.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import genesis as gs


link_yz_offsets = {
    "palm_link": (0.0, 0.0),
    "finger1_link1": (0.05, 0.02),
    "finger1_link2": (0.05, 0.05),
    "finger1_link3": (0.05, 0.08),
    "finger1_link4": (0.05, 0.11),
    "finger2_link1": (0.02, 0.06),
    "finger2_link2": (0.02, 0.10),
    "finger2_link3": (0.02, 0.14),
    "finger2_link4": (0.02, 0.18),
    "finger3_link1": (0.0, 0.06),
    "finger3_link2": (0.0, 0.10),
    "finger3_link3": (0.0, 0.14),
    "finger3_link4": (0.0, 0.18),
    "finger4_link1": (-0.02, 0.06),
    "finger4_link2": (-0.02, 0.10),
    "finger4_link3": (-0.02, 0.14),
    "finger4_link4": (-0.02, 0.18),
    "finger5_link1": (-0.04, 0.04),
    "finger5_link2": (-0.04, 0.08),
    "finger5_link3": (-0.04, 0.12),
    "finger5_link4": (-0.04, 0.16),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tactile-grid", type=str, default="examples/tactile/merged_tactile_grid.json",
                        help="Path to tactile grid JSON file")
    parser.add_argument("--sensor-links", type=str, default=None,
                        help="Comma-separated list of link names to add tactile sensors to (default: all links in grid file)")
    parser.add_argument("--visualize", action="store_true", default=True,
                        help="Show force magnitude visualization")
    parser.add_argument("--no-visualize", action="store_false", dest="visualize",
                        help="Disable force visualization")
    parser.add_argument("--kn", type=float, default=2000.0,
                        help="Normal stiffness coefficient")
    parser.add_argument("--save-video", type=str, default=None,
                        help="Path to save tactile force video")
    parser.add_argument("--save-render", type=str, default=None,
                        help="Path to save rendered scene video (camera view)")
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
            # substeps=10,
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
    obj = scene.add_entity(
        gs.morphs.Cylinder(
            radius=0.02, height=0.2,
            pos=(0.03, 0.01, 0.1),
        ),
    )
    wuji_hand = scene.add_entity(
        gs.morphs.URDF(
            file="assets/wujihand-urdf/urdf/right.urdf",
            merge_fixed_links=True,
            fixed=True,
            pos=(0, 0.1, 0.1),
            euler=(90, 0, 0),
        ),
        vis_mode="collision"
    )

    ########################## Load tactile grid ##########################
    tactile_grid_path = Path(args.tactile_grid)
    if not tactile_grid_path.exists():
        gs.raise_exception(f"Tactile grid file not found: {tactile_grid_path}")

    print(f"\n{'='*70}")
    print(f"Loading tactile grid from: {tactile_grid_path}")
    print(f"{'='*70}")

    with open(tactile_grid_path, 'r') as f:
        tactile_data = json.load(f)

    links_data = tactile_data.get('links', {})

    # Parse sensor links - use all available links if not specified
    if args.sensor_links is None:
        sensor_link_names = list(links_data.keys())
        print(f"\nNo sensor links specified, using all available links from tactile grid")
    else:
        sensor_link_names = [s.strip() for s in args.sensor_links.split(',') if s.strip()]

    print(f"Configuring tactile sensors for links: {sensor_link_names}")

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
    print(f"Adding TactileFieldSensors")
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
        sensor = scene.add_sensor(
            gs.sensors.TactileField(
                entity_idx=wuji_hand.idx,
                link_idx_local=link_idx_local,
                indenter_entity_idx=obj.idx,
                indenter_link_idx_local=0,
                tactile_points_local=local_positions,  # Use custom points!
                kn=args.kn,
            )
        )

        sensors[link_name] = sensor
        sensor_configs[link_name] = {
            'num_points': num_points,
            'link_idx_local': link_idx_local,
        }

        print(f"  ✓ TactileFieldSensor added")

    ########################## camera setup ##########################
    # Add camera for rendering if requested (must be before scene.build())
    cam = None
    if args.save_render:
        print(f"\n{'='*70}")
        print(f"Adding camera for scene rendering...")
        print(f"{'='*70}")
        cam = scene.add_camera(
            res=(1280, 720),
            pos=(0.3, -0.3, 0.3),
            lookat=(0, 0, 0.1),
            fov=40,
            GUI=False,  # Headless camera
        )
        print(f"✓ Camera added for rendering to: {args.save_render}")

    ########################## build ##########################
    print(f"\n{'='*70}")
    print(f"Building scene...")
    print(f"{'='*70}")
    scene.build()
    print(f"✓ Scene built")

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

    ########################## Visualization setup ##########################
    # Storage for video frames (if saving video)
    force_field_frames = {link_name: [] for link_name in sensor_link_names}

    if args.visualize:
        plt.ion()
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        ax.set_title("Full Hand Tactile Force Field")
        ax.set_xlabel("Y (m)")
        ax.set_ylabel("Z (m)")
        ax.set_aspect('equal')

        scatter_plots = {}
        for link_name in sensor_link_names:
            local_positions = tactile_points[link_name]

            # Get YZ offset for this link
            yz_offset = link_yz_offsets.get(link_name, (0.0, 0.0))

            # Apply offset to local positions (project to YZ plane and offset)
            if link_name != "finger1_link2":
                offset_positions = local_positions[:, 1:3].copy()  # Y, Z coordinates
                offset_positions[:, 0] += yz_offset[0]  # Y offset
                offset_positions[:, 1] += yz_offset[1]  # Z offset
            else:
                # project to xz plane
                offset_positions = local_positions[:, [0, 2]].copy()  # X, Z coordinates
                offset_positions[:, 0] += yz_offset[0]  # X offset
                offset_positions[:, 1] += yz_offset[1]  # Z offset

            # Create scatter plot for tactile points
            scatter = ax.scatter(
                offset_positions[:, 0],  # Y coordinate (with offset)
                offset_positions[:, 1],  # Z coordinate (with offset)
                c=np.zeros(len(local_positions)),
                cmap='hot',
                vmin=0,
                vmax=10,
                s=20,
                edgecolors='black',
                linewidths=0.5,
                label=link_name
            )
            scatter_plots[link_name] = (scatter, offset_positions)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, label='Force (N)')
        plt.tight_layout()

    def update_visualization():
        """Update force magnitude visualization."""
        if not args.visualize:
            return

        max_force_global = 0.0
        for link_name in sensor_link_names:
            sensor = sensors[link_name]
            config = sensor_configs[link_name]

            # Read sensor data
            # sensor.read() now returns the correct size (num_points * 3) for each sensor
            force_field_full = sensor.read()
            num_points = config['num_points']

            # Reshape to (num_points, 3)
            force_field_3d = force_field_full.reshape(num_points, 3)

            # Compute force magnitudes
            force_magnitudes = torch.norm(force_field_3d, dim=-1)  # (num_points,)

            # Update scatter plot colors
            force_mag_np = force_magnitudes.cpu().numpy()
            scatter, offset_positions = scatter_plots[link_name]
            scatter.set_array(force_mag_np)

            # Track max force for global scaling
            max_force_global = max(max_force_global, force_mag_np.max())

            # Store for video
            if args.save_video:
                force_field_frames[link_name].append(force_field_3d.cpu().numpy())

        # Auto-scale colorbar globally
        vmax = max(max_force_global, 1.0)
        for link_name in sensor_link_names:
            scatter, _ = scatter_plots[link_name]
            scatter.set_clim(vmin=0, vmax=vmax)

        plt.pause(0.001)

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

            # sensor.read() now returns the correct size (num_points * 3)
            force_field_3d = force_field_full.reshape(num_points, 3)
            force_magnitudes = torch.norm(force_field_3d, dim=-1)

            total_force = force_magnitudes.sum().item()
            max_force = force_magnitudes.max().item()
            mean_force = force_magnitudes.mean().item()

            print(f"{link_name}:")
            print(f"  Total force: {total_force:.2f} N")
            print(f"  Max force:   {max_force:.2f} N")
            print(f"  Mean force:  {mean_force:.2f} N")

    # Start camera recording if camera was added
    if cam is not None:
        cam.start_recording()
        print(f"\n{'='*70}")
        print("Camera recording started...")
        print(f"{'='*70}")

    # PD control loop - run directly without viewer threading
    for i in range(300):

        if i == 200:
            print('here')

        wuji_hand.control_dofs_position(
            pose + i * delta_pose,
            motors_dof_idx,
        )
        scene.step()

        # Render camera frame if recording
        if cam is not None:
            cam.render()

        # Update force visualization every step
        update_visualization()

    # Stop camera recording if it was started
    if cam is not None:
        print(f"\n{'='*70}")
        print("Stopping camera recording and saving video...")
        print(f"{'='*70}")
        cam.stop_recording(save_to_filename=args.save_render, fps=30)
        print(f"✓ Rendered scene video saved to: {args.save_render}")

    # Print final summary
    print_force_summary()

    if args.visualize:
        plt.ioff()

    # Generate video if requested
    if args.save_video and force_field_frames:
        print(f"\n{'='*70}")
        print("GENERATING TACTILE FORCE VIDEO")
        print(f"{'='*70}")

        fig_vid, ax_vid = plt.subplots(1, 1, figsize=(10, 8))

        # Find global max force for consistent scaling
        max_force_all = 0
        for link_name in sensor_link_names:
            frames = force_field_frames[link_name]
            if frames:
                max_force = max([np.linalg.norm(frame, axis=-1).max() for frame in frames])
                max_force_all = max(max_force_all, max_force)

        print(f"Max force magnitude across all sensors: {max_force_all:.2f} N")

        # Animation update function
        def update(frame_idx):
            ax_vid.clear()
            ax_vid.set_title(f"Full Hand Tactile Force Field | Step {frame_idx}")
            ax_vid.set_xlabel("Y (m)")
            ax_vid.set_ylabel("Z (m)")
            ax_vid.set_aspect('equal')

            for link_name in sensor_link_names:
                local_positions = tactile_points[link_name]
                force_data = force_field_frames[link_name][frame_idx]  # (N, 3)
                force_magnitudes = np.linalg.norm(force_data, axis=-1)

                # Get YZ offset for this link
                yz_offset = link_yz_offsets.get(link_name, (0.0, 0.0))

                # Apply offset to local positions
                offset_positions = local_positions[:, 1:3].copy()
                offset_positions[:, 0] += yz_offset[0]
                offset_positions[:, 1] += yz_offset[1]

                # Scatter plot with offset positions
                scatter = ax_vid.scatter(
                    offset_positions[:, 0],
                    offset_positions[:, 1],
                    c=force_magnitudes,
                    cmap='hot',
                    vmin=0,
                    vmax=max_force_all,
                    s=50,
                    edgecolors='black',
                    linewidths=0.5,
                    label=link_name
                )

            plt.tight_layout()
            return []

        # Create animation
        num_frames = len(force_field_frames[sensor_link_names[0]])
        print(f"Creating animation with {num_frames} frames...")
        anim = animation.FuncAnimation(fig_vid, update, frames=num_frames,
                                       interval=50, blit=False, repeat=True)

        # Save video
        print(f"Saving video to: {args.save_video}")
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=30, metadata=dict(artist='Genesis'), bitrate=3600)
        anim.save(args.save_video, writer=writer)

        print(f"✓ Video saved successfully: {args.save_video}")
        plt.close(fig_vid)

    if args.visualize:
        plt.show()


if __name__ == "__main__":
    main()
