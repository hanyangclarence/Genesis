"""
Show a tactile grid's points on the hand, to check where generate_grid_tactile_points.py placed them.

Usage:
    python examples/tactile/visualize_tactile_sensor.py \
        --tactile-grid examples/tactile/full_hand_tactile_v5.json

    # Headless: record a camera video instead of opening the viewer
    python examples/tactile/visualize_tactile_sensor.py --save-render tactile_points.mp4
"""

import argparse
import json

import numpy as np
import torch

import genesis as gs
from genesis.utils import geom as gu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tactile-grid", type=str, default="examples/tactile/full_hand_tactile_v5.json",
                        help="Path to tactile grid JSON file")
    parser.add_argument("--urdf", type=str, default="genesis/assets/urdf/wujihand_v5/wujihand_right_v5.urdf",
                        help="Hand URDF the tactile grid was generated for")
    parser.add_argument("--marker-size", type=float, default=0.0005,
                        help="Radius of tactile point markers in meters")
    parser.add_argument("--save-render", type=str, default=None,
                        help="Record a camera video to this path instead of opening the viewer")
    parser.add_argument("--num-steps", type=int, default=1000,
                        help="Number of simulation steps")
    args = parser.parse_args()

    gs.init(backend=gs.cpu)

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0, -0.5, 0.5),
            camera_lookat=(0.0, 0.0, 0.0),
            camera_fov=40,
        ),
        sim_options=gs.options.SimOptions(dt=0.01),
        show_viewer=args.save_render is None,
    )
    scene.add_entity(gs.morphs.Plane())
    hand = scene.add_entity(
        gs.morphs.URDF(
            file=args.urdf,
            merge_fixed_links=False,
            fixed=True,
            pos=(0, 0.1, 0.1),
            euler=(90, 0, 0),
        ),
        vis_mode="collision",
    )
    cam = None
    if args.save_render:
        cam = scene.add_camera(res=(1280, 720), pos=(0.3, 0.3, 0.3), lookat=(0, 0, 0.1), fov=40, GUI=False)
    scene.build()

    with open(args.tactile_grid) as f:
        links_data = json.load(f)["links"]
    links = [hand.get_link(link_name) for link_name in links_data]
    points_local = [torch.tensor(ld["points"], dtype=gs.tc_float, device=gs.device) for ld in links_data.values()]
    for link_name, points in zip(links_data, points_local):
        print(f"  {link_name}: {len(points)} tactile points")
    print(f"Total tactile points: {sum(len(points) for points in points_local)}")

    # Hold the hand in a slightly open pose
    joints_name = [f"finger{i}_joint{j}" for i in range(1, 6) for j in range(1, 5)]
    motors_dof_idx = [hand.get_joint(name).dofs_idx_local[0] for name in joints_name]
    hand.set_dofs_kp(np.full(len(motors_dof_idx), 20.0), motors_dof_idx)
    hand.set_dofs_kv(np.full(len(motors_dof_idx), 1.0), motors_dof_idx)
    pose = np.zeros(len(motors_dof_idx))
    pose[:2] = (0.7, -0.16)  # thumb

    if cam is not None:
        cam.start_recording()
    for _ in range(args.num_steps):
        hand.control_dofs_position(pose, motors_dof_idx)
        scene.step()

        points_world = torch.cat(
            [gu.transform_by_trans_quat(p, link.get_pos(), link.get_quat()) for link, p in zip(links, points_local)]
        )
        scene.clear_debug_objects()
        scene.draw_debug_spheres(points_world, radius=args.marker_size, color=(1.0, 0.0, 0.0, 1.0))

        if cam is not None:
            cam.render()
    if cam is not None:
        cam.stop_recording(save_to_filename=args.save_render, fps=30)
        print(f"Video saved to: {args.save_render}")


if __name__ == "__main__":
    main()
