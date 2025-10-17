"""
Test TactileFieldSensor with angled sphere drop scenario.

Setup:
- Flat rectangular tactile sensor pad (fixed in space)
- Sphere drops from above at an angle onto the sensor
- Initial velocity gives the sphere angular momentum for angled impact
- Force field readout shows spatial distribution of contact forces
"""

import argparse
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=True, help="Show visualization GUI")
    parser.add_argument("-nv", "--no-vis", action="store_false", dest="vis", help="Disable visualization GUI")
    parser.add_argument("--shape", type=str, default="sphere", choices=["sphere", "cube"], help="Indenter shape")
    parser.add_argument("--size", type=float, default=0.03, help="Indenter size (radius for sphere, side length for cube)")
    parser.add_argument("--save-video", type=str, default="tactile_video.mp4", help="Path to save force field video")
    parser.add_argument("--save-render", type=str, default=None, help="Path to save rendered scene video (camera view)")
    parser.add_argument("--angle", type=float, default=30.0, help="Impact angle in degrees (0=vertical, 90=horizontal)")
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(backend=gs.gpu, logging_level="info")

    ########################## scene setup ##########################
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            substeps=10,
        ),
        rigid_options=gs.options.RigidOptions(
            constraint_timeconst=0.01,  # the smaller, the stiffer the constraint
            enable_collision=True,
            enable_self_collision=False,
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame=True,
            world_frame_size=0.1,
        ),
        show_viewer=args.vis,
    )

    ########################## entities ##########################
    # Ground plane at z=0
    scene.add_entity(gs.morphs.Plane(pos=(0, 0, 0)))
    gs.logger.info("Ground plane added at z=0")

    # Tactile sensor pad resting on the ground (not fixed, preserving link structure)
    # The sensor base bottom is at z=0, top is at z=0.01
    # The elastomer layer top is at z=0.015
    sensor_tilt_angle = np.radians(args.angle)  # Tilt sensor to meet sphere
    sensor_quat = gs.utils.geom.xyz_to_quat(np.array([sensor_tilt_angle, 0, 0]))

    sensor_pad = scene.add_entity(
        gs.morphs.URDF(
            file="assets/tactile_pad_sensor.urdf",
            pos=(0.0, 0.0, 0.005),  # Base center at z=0.005 (base is 0.01m thick, so bottom touches ground)
            quat=sensor_quat,
            fixed=False,  # Not fixed - will rest on ground naturally, preserving link structure
        )
    )
    gs.logger.info(f"Sensor pad added on ground, tilted {args.angle:.1f}° around X-axis")

    # Create indenter shape - drop from above
    shape_size = args.size
    drop_height = 0.15  # Drop from 15cm above the sensor surface

    if args.shape == "sphere":
        shape = scene.add_entity(
            gs.morphs.Sphere(
                radius=shape_size,
                pos=(0, 0.0, drop_height),
                fixed=False,
            )
        )
        gs.logger.info(f"Created sphere indenter (radius={shape_size}m) at height {drop_height}m")
    elif args.shape == "cube":
        shape = scene.add_entity(
            gs.morphs.Box(
                size=(shape_size, shape_size, shape_size),
                pos=(0.0, 0.0, drop_height),
                fixed=False,
            )
        )
        gs.logger.info(f"Created cube indenter (size={shape_size}m) at height {drop_height}m")
    else:
        raise ValueError("Invalid shape choice")
    gs.logger.info(f"Object will drop onto sensor at {args.angle}° angle")

    ########################## sensors ##########################
    # Add sensor BEFORE building scene
    gs.logger.info("Adding tactile field sensor...")
    sensor = scene.add_sensor(
        gs.sensors.TactileField(
            entity_idx=sensor_pad.idx,
            link_idx_local=0,  # sensor_base will be link 0
            indenter_entity_idx=shape.idx,
            indenter_link_idx_local=0,  # Shape has only one link
            num_rows=15,
            num_cols=15,
            surface_size=(0.08, 0.08),
            kn=2000.0,  # Normal stiffness
            kt=200.0,   # Tangential stiffness
            mu=0.5,     # Friction coefficient
            draw_debug=False,
        )
    )
    gs.logger.info(f"✓ Added TactileFieldSensor with {15*15} tactile points")

    # Add camera for rendering if requested
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
        gs.logger.info(f"✓ Camera added for rendering to: {args.save_render}")


    # Now build the scene
    gs.logger.info("Building scene...")
    scene.build()
    gs.logger.info(f"✓ Scene built. Sensor pad idx: {sensor_pad.idx}, Shape idx: {shape.idx}")

    # Customize softness: set the entire sensor pad (merged as single link) to be soft
    gs.logger.info("Customizing sensor pad softness...")

    # Calculate minimum allowed timeconst
    substep_dt = scene.sim_options.dt / scene.sim_options.substeps
    min_timeconst = 2.0 * substep_dt
    gs.logger.info(f"  Simulation: dt={scene.sim_options.dt}, substeps={scene.sim_options.substeps}")
    gs.logger.info(f"  Substep dt={substep_dt:.6f}, min_timeconst={min_timeconst:.6f}")

    # Set softness for the entire sensor pad (all geometries in all links)
    soft_timeconst = 0.01  # Soft compliance (10x larger than default 0.01)
    soft_params = np.array([soft_timeconst, 0.5, 1e-4, 1e-4, 0.0, 1e-4, 1.0])

    gs.logger.info(f"  Setting sensor pad (entity {sensor_pad.idx}, {len(sensor_pad.links)} link(s)) to soft")
    for link in sensor_pad.links:
        for geom in link.geoms:
            gs.logger.info(f"    Link {link.idx}, geom {geom.idx}: before timeconst={geom.sol_params[0]:.6f}")
            geom.set_sol_params(soft_params)
            gs.logger.info(f"    Link {link.idx}, geom {geom.idx}: after  timeconst={geom.sol_params[0]:.6f}")
        link.set_friction(0.8)  # High friction for tactile grip

    gs.logger.info(f"  ✓ Sensor pad: timeconst={soft_timeconst} (soft), friction=0.8")

    # Start camera recording if camera was added
    if cam is not None:
        cam.start_recording()
        gs.logger.info("Camera recording started...")

    max_steps = 70

    # Storage for video frames (store full 3D force vectors)
    force_field_frames = []

    for step in range(max_steps):
        scene.step()

        if step == 13:
            print('here')

        # Render camera frame if recording
        if cam is not None:
            cam.render()

        # Read sensor data
        force_field = sensor.read()  # Shape: (num_rows * num_cols * 3,)

        # Reshape to (num_rows, num_cols, 3) for analysis
        force_field_3d = force_field.reshape(15, 15, 3)
        # force_field_3d[..., :2] = 0.0  # Zero out tangential forces for visualization

        # Store full 3D force vectors for video
        force_field_frames.append(force_field_3d.cpu().numpy())

    # Stop camera recording if it was started
    if cam is not None:
        gs.logger.info("\nStopping camera recording and saving video...")
        cam.stop_recording(save_to_filename=args.save_render, fps=20)
        gs.logger.info(f"✓ Rendered scene video saved to: {args.save_render}")

    # Final report
    gs.logger.info("\n" + "="*60)
    gs.logger.info("SIMULATION COMPLETE")
    gs.logger.info("="*60)

    gs.logger.info("\nVisualization:")
    gs.logger.info("- Colored spheres = tactile points (color = force magnitude)")
    gs.logger.info(f"- Sphere impacts at {args.angle}° angle")

    # Generate and save 3D vector field video
    gs.logger.info(f"\n{'='*60}")
    gs.logger.info("GENERATING 3D FORCE VECTOR VIDEO")
    gs.logger.info(f"{'='*60}")

    # Create 3D figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Create grid coordinates
    num_rows, num_cols = 15, 15
    x_grid = np.linspace(-0.02, 0.02, num_cols)  # Center around origin
    y_grid = np.linspace(-0.02, 0.02, num_rows)
    X, Y = np.meshgrid(x_grid, y_grid)

    # Find global max force magnitude for consistent scaling
    max_force_all = max([np.linalg.norm(frame, axis=-1).max() for frame in force_field_frames])

    # Scaling factor for arrow length
    arrow_scale = 0.005 / (max_force_all + 1e-6)  # Scale so max arrow is ~5mm

    gs.logger.info(f"Max force magnitude: {max_force_all:.2f} N")
    gs.logger.info(f"Arrow scale factor: {arrow_scale:.6f}")

    # Animation update function
    def update(frame_idx):
        ax.clear()

        # Get force data for this frame
        force_data = force_field_frames[frame_idx]  # Shape: (15, 15, 3)

        # Draw the tactile grid base (tilted to match sensor orientation)
        ax.plot_surface(X, Y, np.zeros_like(X), alpha=0.2, color='lightgray')

        # Draw force vectors
        num_vectors_drawn = 0
        for i in range(num_rows):
            for j in range(num_cols):
                fx, fy, fz = force_data[i, j]
                force_mag = np.sqrt(fx**2 + fy**2 + fz**2)

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
                num_vectors_drawn += 1

        # Calculate total force and components
        total_force = np.linalg.norm(force_data, axis=-1).sum()
        max_force_frame = np.linalg.norm(force_data, axis=-1).max()
        normal_force = force_data[:, :, 2].sum()
        tangential_force = np.linalg.norm(force_data[:, :, :2], axis=-1).sum()

        # Set labels and title
        ax.set_xlabel('X (m)', fontsize=10)
        ax.set_ylabel('Y (m)', fontsize=10)
        ax.set_zlabel('Force (N × scale)', fontsize=10)
        ax.set_title(f'Step {frame_idx} | Total: {total_force:.2f}N | Normal: {normal_force:.2f}N | Tangential: {tangential_force:.2f}N | Vectors: {num_vectors_drawn}',
                    fontsize=11, fontweight='bold')

        # Set consistent axis limits
        ax.set_xlim(-0.02, 0.02)
        ax.set_ylim(-0.02, 0.02)
        ax.set_zlim(-0.01, 0.02)

        # Better viewing angle
        ax.view_init(elev=25, azim=45)

        return []

    # Create animation
    gs.logger.info(f"Creating 3D animation with {len(force_field_frames)} frames...")
    anim = animation.FuncAnimation(fig, update, frames=len(force_field_frames),
                                   interval=50, blit=False, repeat=True)

    # Save video
    gs.logger.info(f"Saving video to: {args.save_video}")
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=20, metadata=dict(artist='Genesis'), bitrate=3600)
    anim.save(args.save_video, writer=writer)

    gs.logger.info(f"✓ Video saved successfully: {args.save_video}")
    gs.logger.info(f"  - Frames: {len(force_field_frames)}")
    gs.logger.info(f"  - Impact angle: {args.angle}°")
    gs.logger.info(f"  - Max force magnitude: {max_force_all:.2f} N")
    gs.logger.info(f"  - FPS: 20")

    plt.close(fig)


if __name__ == "__main__":
    main()
