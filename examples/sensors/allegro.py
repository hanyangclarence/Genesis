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
        max_FPS=60,
    )

    scene = gs.Scene(
        viewer_options=viewer_options,
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        show_viewer=True,
    )

    ########################## entities ##########################
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    obj = scene.add_entity(
        gs.morphs.Cylinder(
            radius=0.02, height=0.2,
            pos=(0.07, 0, 0.1),
        ),
    )
    allegro = scene.add_entity(
        gs.morphs.URDF(
            file="assets/allegro_hand/allegro_hand_right_glb.urdf",
            merge_fixed_links=True,
            fixed=True,
            pos=(0, 0, 0.1),
            euler=(90, 0, 0),
        ),
    )
    ########################## build ##########################
    scene.build()

    joints_name = (
        "joint_0.0",
        "joint_4.0",
        "joint_8.0",
        "joint_12.0",
        "joint_1.0",
        "joint_5.0",
        "joint_9.0",
        "joint_13.0",
        "joint_2.0",
        "joint_6.0",
        "joint_10.0",
        "joint_14.0",
        "joint_3.0",
        "joint_7.0",
        "joint_11.0",
        "joint_15.0",
    )
    motors_dof_idx = [allegro.get_joint(name).dofs_idx_local[0] for name in joints_name]

    # Optional: set control gains
    allegro.set_dofs_kp(
        np.array([200, 200, 200, 200,] * 4),
        motors_dof_idx,
    )
    allegro.set_dofs_kv(
        np.array([10, 10, 10, 10] * 4),
        motors_dof_idx,
    )

    def grasp():
        # PD control
        for i in range(100):
            allegro.control_dofs_position(
                i * np.array([0.01] * 16),
                motors_dof_idx,
            )
            scene.step(refresh_visualizer=False)

    import threading
    threading.Thread(target=grasp).start()
    scene.viewer.run()

if __name__ == "__main__":
    main()