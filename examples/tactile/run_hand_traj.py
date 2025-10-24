import argparse

import numpy as np

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(backend=gs.cpu)

    ########################## create a scene ##########################
    viewer_options = gs.options.ViewerOptions(
        camera_pos=(0, -0.5, 0.5),
        camera_lookat=(0.0, 0.0, 0.0),
        camera_fov=40,
        # max_FPS=5,
    )

    scene = gs.Scene(
        viewer_options=viewer_options,
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        # rigid_options=gs.options.RigidOptions(
        #     enable_self_collision=False,
        # ),
        show_viewer=False,
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
        1.6, -0.16, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
    ])
    delta_pose = np.array([
        # Finger 1 (Thumb)
        0.00, 0.00, 0.01, 0.01,
        # Finger 2 (Index)
        0.01, 0.00, 0.01, 0.01,
        # Finger 3 (Middle)
        0.01, 0.00, 0.01, 0.01,
        # Finger 4 (Ring)
        0.01, 0.00, 0.01, 0.01,
        # Finger 5 (Pinky)
        0.01, 0.00, 0.01, 0.01,
    ])
    

    def grasp():
        # PD control
        for i in range(300):
            wuji_hand.control_dofs_position(
                pose + i * delta_pose,
                motors_dof_idx,
            )
            scene.step(refresh_visualizer=False)

    import threading
    threading.Thread(target=grasp).start()
    scene.viewer.run()

if __name__ == "__main__":
    main()