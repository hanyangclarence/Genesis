"""
Full hand tactile sensing demo using TactileFieldSensor for Genesis v0.3.10.

This script demonstrates tactile sensors on multiple links of the Wuji hand,
using precomputed tactile point grids from a JSON file.

Visualizes tactile points and 3D force vectors in local link coordinates.

Usage:
    # Visualize a specific link (default: first link in grid)
    python tactile_field_hand.py --viz-link finger2_link3

    # Save tactile force video
    python tactile_field_hand.py --viz-link palm_link --save-video tactile.mp4

    # Load sensors for specific links only
    python tactile_field_hand.py --sensor-links palm_link,finger2_link3
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

import genesis as gs


def main():
    parser = argparse.ArgumentParser(description="Full Hand Tactile Sensing Demo")
    parser.add_argument("--tactile-grid", type=str,
                        default="examples/tactile/merged_tactile_grid.json",
                        help="Path to tactile grid JSON file")
    parser.add_argument("--sensor-links", type=str, default=None,
                        help="Comma-separated list of link names to add tactile sensors to (default: all links in grid file)")
    parser.add_argument("--visualize", action="store_true", default=True,
                        help="Show force magnitude visualization")
    parser.add_argument("--no-visualize", action="store_false", dest="visualize",
                        help="Disable force visualization")
    parser.add_argument("--viz-link", type=str, default=None,
                        help="Specific link to visualize in 3D local coordinates (e.g., 'finger2_link3')")
    parser.add_argument("--kn", type=float, default=2000.0,
                        help="Normal stiffness coefficient")
    parser.add_argument("--save-video", type=str, default=None,
                        help="Path to save tactile force video")
    parser.add_argument("--save-render", type=str, default=None,
                        help="Path to save rendered scene video (camera view)")
    parser.add_argument("--num-steps", type=int, default=600,
                        help="Number of simulation steps")
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(backend=gs.gpu, logging_level="info")

    ########################## create a scene ##########################
    viewer_options = gs.options.ViewerOptions(
        camera_pos=(0, -0.5, 0.5),
        camera_lookat=(0.0, 0.0, 0.0),
        camera_fov=40,
        max_FPS=1,
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
    obj = scene.add_entity(
        gs.morphs.Cylinder(
            radius=0.02, height=0.2,
            pos=(0.03, 0.01, 0.1),
        ),
    )
    wuji_hand = scene.add_entity(
        gs.morphs.URDF(
            file="genesis/assets/urdf/wujihand-urdf/urdf/right.urdf",
            merge_fixed_links=False,
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
            gs.sensors.TactileField3D(
                entity_idx=wuji_hand.idx,
                link_idx_local=link_idx_local,
                indenter_entity_idx=obj.idx,
                indenter_link_idx_local=0,
                tactile_points_local=local_positions,  # Use custom points!
                kn=args.kn,
                kt=200.0,
                mu=1.0,
            )
        )

        sensors[link_name] = sensor
        sensor_configs[link_name] = {
            'num_points': num_points,
            'link_idx_local': link_idx_local,
        }

        print(f"  TactileFieldSensor added")

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
        print(f"Camera added for rendering to: {args.save_render}")

    ########################## build ##########################
    print(f"\n{'='*70}")
    print(f"Building scene...")
    print(f"{'='*70}")
    scene.build()
    print(f"Scene built")

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

    # Visualization link (default to first sensor link if not specified)
    viz_link = args.viz_link if args.viz_link else sensor_link_names[0]
    if viz_link not in sensor_link_names:
        gs.raise_exception(f"Visualization link '{viz_link}' not in sensor links: {sensor_link_names}")
    print(f"\nVisualization: {viz_link} in local 3D coordinates")

    # Get local positions for visualization link
    viz_local_positions = tactile_points[viz_link]  # (N, 3) in local coordinates

    fig = None
    ax = None
    if args.visualize:
        plt.ion()
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(f"Tactile Force: {viz_link}")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.view_init(elev=25, azim=-90)
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)

    # Fixed arrow scale: 1 N = 1 cm (0.01 m) arrow length
    arrow_scale = 0.005

    def update_visualization():
        """Update 3D force vector visualization for single link."""
        # Read sensor data for visualization link
        sensor = sensors[viz_link]
        config = sensor_configs[viz_link]
        num_points = config['num_points']

        force_field_full = sensor.read()
        force_field_5d = force_field_full.reshape(num_points, 5)
        force_field_3d = force_field_5d[:, :3].cpu().numpy()  # (N, 3)

        # Compute max force
        force_magnitudes = np.linalg.norm(force_field_3d, axis=-1)
        max_force = force_magnitudes.max()

        # Store frames for video
        if args.save_video:
            for link_name in sensor_link_names:
                s = sensors[link_name]
                c = sensor_configs[link_name]
                ff = s.read().reshape(c['num_points'], 5)[:, :3].cpu().numpy()
                force_field_frames[link_name].append(ff.copy())

        # Draw visualization
        if args.visualize and ax is not None:
            ax.clear()

            # Plot tactile points
            ax.scatter(viz_local_positions[:, 0],
                      viz_local_positions[:, 1],
                      viz_local_positions[:, 2],
                      c='blue', s=20, alpha=0.6, label='Tactile points')

            # Plot force vectors
            for i in range(num_points):
                fx, fy, fz = force_field_3d[i]
                force_mag = force_magnitudes[i]

                if force_mag < 0.01:
                    continue

                x, y, z = viz_local_positions[i]

                # Color based on magnitude (1.0 N = full red)
                color_intensity = min(force_mag / 1.0, 1.0)
                color = plt.cm.jet(color_intensity)

                # Draw force arrow
                ax.quiver(x, y, z,
                         fx * arrow_scale, fy * arrow_scale, fz * arrow_scale,
                         color=color, arrow_length_ratio=0.3, linewidth=1.5)

            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title(f'{viz_link} | Max: {max_force:.2f} N | Scale: 1N = 1cm')

            # Auto-scale axes based on point positions
            margin = 0.01
            ax.set_xlim(viz_local_positions[:, 0].min() - margin,
                       viz_local_positions[:, 0].max() + margin)
            ax.set_ylim(viz_local_positions[:, 1].min() - margin,
                       viz_local_positions[:, 1].max() + margin)
            ax.set_zlim(viz_local_positions[:, 2].min() - margin,
                       viz_local_positions[:, 2].max() + margin)

            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.01)

        return max_force

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

            # Reshape to (num_points, 5) - [fx, fy, fz, fn_mag, ft_mag]
            force_field_5d = force_field_full.reshape(num_points, 5)
            force_field_3d = force_field_5d[:, :3]  # 3D force vector
            fn_magnitude = force_field_5d[:, 3]     # Normal force magnitude
            ft_magnitude = force_field_5d[:, 4]     # Tangential force magnitude

            force_magnitudes = torch.norm(force_field_3d, dim=-1)

            total_force = force_magnitudes.sum().item()
            max_force = force_magnitudes.max().item()
            mean_force = force_magnitudes.mean().item()
            max_fn = fn_magnitude.max().item()
            max_ft = ft_magnitude.max().item()

            print(f"{link_name}:")
            print(f"  Total force: {total_force:.2f} N")
            print(f"  Max force:   {max_force:.2f} N")
            print(f"  Mean force:  {mean_force:.2f} N")
            print(f"  Max normal:  {max_fn:.2f} N")
            print(f"  Max tangent: {max_ft:.2f} N")

    # Start camera recording if camera was added
    if cam is not None:
        cam.start_recording()
        print(f"\n{'='*70}")
        print("Camera recording started...")
        print(f"{'='*70}")
    
    obj_mass = obj.get_mass()
    external_force = torch.tensor([0.0, 0.0, obj_mass * 9.81 * 1.05])

    # PD control loop
    print(f"\n{'='*70}")
    print(f"Running simulation for {args.num_steps} steps...")
    print(f"{'='*70}")

    for i in range(args.num_steps):
        scene.rigid_solver.apply_links_external_force(
            force=external_force,
            links_idx=obj.links[0].idx,
            ref="link_com",
        )

        wuji_hand.control_dofs_position(
            pose + i * delta_pose,
            motors_dof_idx,
        )
        scene.step()

        # Render camera frame if recording
        if cam is not None:
            cam.render()

        # Update force visualization every step
        max_force = update_visualization()

        # Print progress every 50 steps
        if i % 50 == 0:
            print(f"Step {i:4d}/{args.num_steps}: Max force = {max_force:.2f} N")

    # Stop camera recording if it was started
    if cam is not None:
        print(f"\n{'='*70}")
        print("Stopping camera recording and saving video...")
        print(f"{'='*70}")
        cam.stop_recording(save_to_filename=args.save_render, fps=30)
        print(f"Rendered scene video saved to: {args.save_render}")

    # Print final summary
    print_force_summary()

    if args.visualize:
        plt.ioff()

    # Generate video if requested
    if args.save_video and force_field_frames[viz_link]:
        print(f"\n{'='*70}")
        print(f"GENERATING 3D TACTILE FORCE VIDEO FOR {viz_link}")
        print(f"{'='*70}")

        fig_vid = plt.figure(figsize=(10, 8))
        ax_vid = fig_vid.add_subplot(111, projection='3d')

        frames = force_field_frames[viz_link]
        max_force_all = max([np.linalg.norm(frame, axis=-1).max() for frame in frames])

        # Fixed arrow scale: 1 N = 1 cm (0.01 m) arrow length
        video_arrow_scale = 0.005

        print(f"Max force magnitude: {max_force_all:.2f} N")
        print(f"Arrow scale: 1 N = 1 cm")

        def update(frame_idx):
            ax_vid.clear()

            force_data = frames[frame_idx]
            force_magnitudes = np.linalg.norm(force_data, axis=-1)
            max_force_frame = force_magnitudes.max()

            # Plot tactile points
            ax_vid.scatter(viz_local_positions[:, 0],
                          viz_local_positions[:, 1],
                          viz_local_positions[:, 2],
                          c='blue', s=20, alpha=0.6)

            # Plot force vectors
            for i in range(len(viz_local_positions)):
                fx, fy, fz = force_data[i]
                force_mag = force_magnitudes[i]

                if force_mag < 0.01:
                    continue

                x, y, z = viz_local_positions[i]
                # Color based on magnitude (1.0 N = full red)
                color_intensity = min(force_mag / 1.0, 1.0)
                color = plt.cm.jet(color_intensity)

                ax_vid.quiver(x, y, z,
                             fx * video_arrow_scale, fy * video_arrow_scale, fz * video_arrow_scale,
                             color=color, arrow_length_ratio=0.3, linewidth=1.5)

            ax_vid.set_xlabel('X (m)')
            ax_vid.set_ylabel('Y (m)')
            ax_vid.set_zlabel('Z (m)')
            ax_vid.set_title(f'{viz_link} | Step {frame_idx} | Max: {max_force_frame:.2f} N | 1N=1cm')

            margin = 0.01
            ax_vid.set_xlim(viz_local_positions[:, 0].min() - margin,
                           viz_local_positions[:, 0].max() + margin)
            ax_vid.set_ylim(viz_local_positions[:, 1].min() - margin,
                           viz_local_positions[:, 1].max() + margin)
            ax_vid.set_zlim(viz_local_positions[:, 2].min() - margin,
                           viz_local_positions[:, 2].max() + margin)
            ax_vid.view_init(elev=25, azim=-90)

            return []

        num_frames = len(frames)
        print(f"Creating animation with {num_frames} frames...")
        anim = animation.FuncAnimation(fig_vid, update, frames=num_frames,
                                       interval=50, blit=False, repeat=True)

        print(f"Saving video to: {args.save_video}")
        saved = False

        if 'ffmpeg' in animation.writers.list():
            try:
                Writer = animation.writers['ffmpeg']
                writer = Writer(fps=20, metadata=dict(artist='Genesis'), bitrate=3600)
                anim.save(args.save_video, writer=writer)
                saved = True
                print(f"Video saved: {args.save_video}")
            except Exception as e:
                print(f"ffmpeg failed: {e}")

        if not saved:
            try:
                gif_path = args.save_video.replace('.mp4', '.gif')
                anim.save(gif_path, writer='pillow', fps=20)
                saved = True
                print(f"Video saved as GIF: {gif_path}")
            except Exception as e:
                print(f"pillow failed: {e}")

        if not saved:
            print("ERROR: Could not save video. Install ffmpeg: sudo apt-get install ffmpeg")

        plt.close(fig_vid)

    if args.visualize:
        plt.show()

    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
