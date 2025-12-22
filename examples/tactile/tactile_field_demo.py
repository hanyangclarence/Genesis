"""
TactileFieldSensor Demo for Genesis v0.3.10

This script demonstrates the TactileFieldSensor, which computes dense tactile
force fields using SDF-based penetration depth queries.

Supports headless operation with video saving capabilities.

Usage:
    # With viewer
    python tactile_field_demo.py --visualize

    # Headless with video outputs
    python tactile_field_demo.py --no-visualize --save-video tactile.mp4 --save-render scene.mp4
"""
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

import genesis as gs


def main():
    parser = argparse.ArgumentParser(description="TactileFieldSensor Demo")
    parser.add_argument("-v", "--visualize", action="store_true", default=False,
                        help="Show viewer (default: False for headless)")
    parser.add_argument("-nv", "--no-visualize", action="store_false", dest="visualize",
                        help="Disable viewer")
    parser.add_argument("--shape", type=str, default="sphere", choices=["sphere", "cube"],
                        help="Indenter shape")
    parser.add_argument("--size", type=float, default=0.03,
                        help="Indenter size (radius for sphere, side length for cube)")
    parser.add_argument("--kn", type=float, default=2000.0,
                        help="Normal stiffness coefficient")
    parser.add_argument("--num-steps", type=int, default=100,
                        help="Number of simulation steps")
    parser.add_argument("--save-video", type=str, default=None,
                        help="Path to save tactile force field video (e.g., tactile.mp4)")
    parser.add_argument("--save-render", type=str, default=None,
                        help="Path to save rendered scene video (camera view)")
    parser.add_argument("--num-rows", type=int, default=10,
                        help="Number of tactile points in rows")
    parser.add_argument("--num-cols", type=int, default=10,
                        help="Number of tactile points in columns")
    args = parser.parse_args()

    # Initialize Genesis
    gs.init(backend=gs.gpu, logging_level="info")

    # Create scene
    viewer_options = gs.options.ViewerOptions(
        camera_pos=(0.3, -0.3, 0.3),
        camera_lookat=(0.0, 0.0, 0.1),
        camera_fov=40,
        max_FPS=60,
    )

    scene = gs.Scene(
        viewer_options=viewer_options,
        sim_options=gs.options.SimOptions(
            dt=0.01,
            substeps=10,
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=True,
            constraint_timeconst=0.01,
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame=True,
            world_frame_size=0.1,
        ),
        show_viewer=args.visualize,
    )

    # Add ground plane
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    gs.logger.info("Ground plane added at z=0")

    # Add a tactile sensor pad (the object with tactile sensors)
    sensor_pad = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.02),
            pos=(0.0, 0.0, 0.05),
            fixed=True,
        ),
    )
    gs.logger.info("Sensor pad added (fixed box)")

    # Add an indenter (the object that will contact the sensor)
    drop_height = 0.15
    if args.shape == "sphere":
        indenter = scene.add_entity(
            gs.morphs.Sphere(
                radius=args.size,
                pos=(0.0, 0.0, drop_height),
            ),
        )
        gs.logger.info(f"Created sphere indenter (radius={args.size}m) at height {drop_height}m")
    else:  # cube
        indenter = scene.add_entity(
            gs.morphs.Box(
                size=(args.size, args.size, args.size),
                pos=(0.0, 0.0, drop_height),
            ),
        )
        gs.logger.info(f"Created cube indenter (size={args.size}m) at height {drop_height}m")

    # Create tactile sensor grid positions on the top surface of the sensor pad
    num_rows, num_cols = args.num_rows, args.num_cols
    width, height = 0.08, 0.08  # Slightly smaller than the pad
    x = np.linspace(-width/2, width/2, num_cols)
    y = np.linspace(-height/2, height/2, num_rows)
    xv, yv = np.meshgrid(x, y)
    # Z is at the top surface of the sensor pad (pad is 0.02m thick, centered at z=0.05)
    z = np.ones_like(xv) * 0.01  # Top surface offset from pad center
    tactile_points = np.stack([xv.flatten(), yv.flatten(), z.flatten()], axis=-1).astype(np.float32)

    print(f"\n{'='*60}")
    print(f"Creating TactileFieldSensor with {len(tactile_points)} tactile points ({num_rows}x{num_cols})")
    print(f"{'='*60}")

    # Add TactileFieldSensor
    tactile_sensor = scene.add_sensor(
        gs.sensors.TactileField(
            entity_idx=sensor_pad.idx,
            link_idx_local=0,
            indenter_entity_idx=indenter.idx,
            indenter_link_idx_local=0,
            tactile_points_local=tactile_points,
            kn=args.kn,
        )
    )
    gs.logger.info(f"TactileFieldSensor added with {len(tactile_points)} tactile points")

    # Add camera for rendering if requested (must be before scene.build())
    cam = None
    if args.save_render:
        gs.logger.info("Adding camera for scene rendering...")
        cam = scene.add_camera(
            res=(1280, 720),
            pos=(0.3, -0.3, 0.3),
            lookat=(0, 0, 0.1),
            fov=40,
            GUI=False,  # Headless camera
        )
        gs.logger.info(f"Camera added for rendering to: {args.save_render}")

    # Build the scene
    print(f"\n{'='*60}")
    print("Building scene...")
    print(f"{'='*60}")
    scene.build()
    print("Scene built successfully!")

    # Start camera recording if camera was added
    if cam is not None:
        cam.start_recording()
        gs.logger.info("Camera recording started...")

    # Storage for force field frames (for video)
    force_field_frames = []

    # Simulation loop
    print(f"\n{'='*60}")
    print(f"Running simulation for {args.num_steps} steps...")
    print(f"{'='*60}")

    max_force_seen = 0.0
    for step in range(args.num_steps):
        scene.step()

        # Render camera frame if recording
        if cam is not None:
            cam.render()

        # Read tactile sensor data
        force_field = tactile_sensor.read()  # Shape: (num_points * 3,)

        # Reshape to (num_rows, num_cols, 3) for analysis
        force_field_3d = force_field.reshape(num_rows, num_cols, 3)

        # Store for video
        if args.save_video:
            force_field_frames.append(force_field_3d.cpu().numpy())

        # Compute force magnitudes
        force_magnitudes = torch.norm(force_field_3d, dim=-1)
        total_force = force_magnitudes.sum().item()
        max_force = force_magnitudes.max().item()

        if max_force > max_force_seen:
            max_force_seen = max_force

        # Print status every 10 steps or when significant force detected
        if step % 10 == 0 or max_force > 0.1:
            print(f"Step {step:4d}: Total force = {total_force:8.2f} N, "
                  f"Max force = {max_force:8.2f} N")

    # Stop camera recording if it was started
    if cam is not None:
        gs.logger.info("\nStopping camera recording and saving video...")
        cam.stop_recording(save_to_filename=args.save_render, fps=20)
        gs.logger.info(f"Rendered scene video saved to: {args.save_render}")

    # Final summary
    print(f"\n{'='*60}")
    print("Simulation Complete!")
    print(f"{'='*60}")
    print(f"Maximum force observed: {max_force_seen:.2f} N")

    # Read final sensor state
    final_forces = tactile_sensor.read().reshape(num_rows, num_cols, 3)
    final_magnitudes = torch.norm(final_forces, dim=-1)

    print(f"\nFinal state:")
    print(f"  Total force: {final_magnitudes.sum().item():.2f} N")
    print(f"  Max force:   {final_magnitudes.max().item():.2f} N")
    print(f"  Mean force:  {final_magnitudes.mean().item():.2f} N")

    # Show points with contact
    contact_mask = final_magnitudes > 0.01
    num_contacts = contact_mask.sum().item()
    print(f"  Points in contact: {num_contacts}/{len(tactile_points)}")

    # Generate and save 3D vector field video if requested
    if args.save_video and force_field_frames:
        print(f"\n{'='*60}")
        print("GENERATING 3D FORCE VECTOR VIDEO")
        print(f"{'='*60}")

        # Create 3D figure
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Create grid coordinates
        x_grid = np.linspace(-width/2, width/2, num_cols)
        y_grid = np.linspace(-height/2, height/2, num_rows)
        X, Y = np.meshgrid(x_grid, y_grid)

        # Find global max force magnitude for consistent scaling
        max_force_all = max([np.linalg.norm(frame, axis=-1).max() for frame in force_field_frames])
        if max_force_all < 1e-6:
            max_force_all = 1.0  # Avoid division by zero

        # Scaling factor for arrow length
        arrow_scale = 0.01 / (max_force_all + 1e-6)  # Scale so max arrow is ~1cm

        print(f"Max force magnitude: {max_force_all:.2f} N")
        print(f"Arrow scale factor: {arrow_scale:.6f}")

        # Animation update function
        def update(frame_idx):
            ax.clear()

            # Get force data for this frame
            force_data = force_field_frames[frame_idx]  # Shape: (num_rows, num_cols, 3)

            # Draw the tactile grid base
            ax.plot_surface(X, Y, np.zeros_like(X), alpha=0.2, color='lightgray')

            # Draw force vectors
            for i in range(num_rows):
                for j in range(num_cols):
                    fx, fy, fz = force_data[i, j]
                    force_mag = np.sqrt(fx**2 + fy**2 + fz**2)

                    if force_mag < 0.01:  # Skip very small forces
                        continue

                    x_pos, y_pos = X[i, j], Y[i, j]
                    z_pos = 0.0  # Base of sensor surface

                    # Scale arrow length
                    dx = fx * arrow_scale
                    dy = fy * arrow_scale
                    dz = fz * arrow_scale

                    # Color based on magnitude (normalized)
                    color_intensity = min(force_mag / max_force_all, 1.0)
                    color = plt.cm.jet(color_intensity)

                    # Draw arrow
                    ax.quiver(x_pos, y_pos, z_pos, dx, dy, dz,
                              color=color, arrow_length_ratio=0.3, linewidth=2)

            # Calculate total force and components
            total_force = np.linalg.norm(force_data, axis=-1).sum()
            max_force_frame = np.linalg.norm(force_data, axis=-1).max()
            normal_force = force_data[:, :, 2].sum()
            tangential_force = np.linalg.norm(force_data[:, :, :2], axis=-1).sum()

            # Set labels and title
            ax.set_xlabel('X (m)', fontsize=10)
            ax.set_ylabel('Y (m)', fontsize=10)
            ax.set_zlabel('Force (scaled)', fontsize=10)
            ax.set_title(f'Step {frame_idx} | Total: {total_force:.2f}N | '
                        f'Normal: {normal_force:.2f}N | Tangential: {tangential_force:.2f}N',
                        fontsize=11, fontweight='bold')

            # Set consistent axis limits
            ax.set_xlim(-width/2 - 0.01, width/2 + 0.01)
            ax.set_ylim(-height/2 - 0.01, height/2 + 0.01)
            ax.set_zlim(-0.01, 0.03)

            # Better viewing angle
            ax.view_init(elev=25, azim=45)

            return []

        # Create animation
        print(f"Creating 3D animation with {len(force_field_frames)} frames...")
        anim = animation.FuncAnimation(fig, update, frames=len(force_field_frames),
                                       interval=50, blit=False, repeat=True)

        # Save video - try multiple writers
        print(f"Saving video to: {args.save_video}")
        saved = False

        # Try ffmpeg first
        if not saved and 'ffmpeg' in animation.writers.list():
            try:
                Writer = animation.writers['ffmpeg']
                writer = Writer(fps=20, metadata=dict(artist='Genesis'), bitrate=3600)
                anim.save(args.save_video, writer=writer)
                saved = True
                print(f"Video saved with ffmpeg: {args.save_video}")
            except Exception as e:
                print(f"ffmpeg failed: {e}")

        # Try pillow (for gif)
        if not saved:
            try:
                gif_path = args.save_video.replace('.mp4', '.gif')
                anim.save(gif_path, writer='pillow', fps=20)
                saved = True
                print(f"Video saved as GIF with pillow: {gif_path}")
            except Exception as e:
                print(f"pillow failed: {e}")

        # Try saving individual frames as images
        if not saved:
            try:
                import os
                frames_dir = args.save_video.replace('.mp4', '_frames')
                os.makedirs(frames_dir, exist_ok=True)
                print(f"Saving individual frames to: {frames_dir}/")
                for i in range(len(force_field_frames)):
                    update(i)
                    fig.savefig(f"{frames_dir}/frame_{i:04d}.png", dpi=100)
                saved = True
                print(f"Saved {len(force_field_frames)} frames to {frames_dir}/")
                print("To convert to video: ffmpeg -framerate 20 -i frame_%04d.png -c:v libx264 output.mp4")
            except Exception as e:
                print(f"Frame saving failed: {e}")

        if saved:
            print(f"  - Frames: {len(force_field_frames)}")
            print(f"  - Max force magnitude: {max_force_all:.2f} N")
        else:
            print("ERROR: Could not save video with any available method")
            print("Install ffmpeg: sudo apt-get install ffmpeg")
            print("Or install imageio: pip install imageio imageio-ffmpeg")

        plt.close(fig)

    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
