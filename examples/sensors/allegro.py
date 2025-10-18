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
        max_FPS=15,
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
    shadow_hand = scene.add_entity(
        gs.morphs.URDF(
            file="assets/shadow_hand/shadow_hand_right_glb.urdf",
            merge_fixed_links=True,
            fixed=True,
            pos=(0, 0.33, 0.1),
            euler=(90, 0, 0),
        ),
    )
    ########################## build ##########################
    scene.build()

    # Shadow Hand joints: WR (wrist), FF (first finger/index), MF (middle), RF (ring), LF (little), TH (thumb)
    joints_name = (
        "WRJ2", #"WRJ1",  # Wrist
        "FFJ4", "FFJ3", "FFJ2", "FFJ1",  # First Finger (Index)
        "MFJ4", "MFJ3", "MFJ2", "MFJ1",  # Middle Finger
        "RFJ4", "RFJ3", "RFJ2", "RFJ1",  # Ring Finger
        "LFJ5", "LFJ4", "LFJ3", "LFJ2", "LFJ1",  # Little Finger
        "THJ5", "THJ4", "THJ3", "THJ2", "THJ1",  # Thumb
    )
    motors_dof_idx = [shadow_hand.get_joint(name).dofs_idx_local[0] for name in joints_name]

    # Optional: set control gains
    shadow_hand.set_dofs_kp(
        np.array([200] * len(motors_dof_idx)),
        motors_dof_idx,
    )
    shadow_hand.set_dofs_kv(
        np.array([10] * len(motors_dof_idx)),
        motors_dof_idx,
    )

    pose = np.array([
        0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
        -1.0, 1.0, 0.0, 0.0, 0.0,
    ])
    delta_pose = np.array([
        0.01,
        0.00, 0.02, 0.01, 0.01,
        0.00, 0.02, 0.01, 0.01,
        0.00, 0.02, 0.01, 0.01,
        0.00, 0.02, 0.02, 0.01, 0.01,
        0.00, -0.01, 0.01, 0.01, 0.01,
    ])
    

    def grasp():
        # PD control
        for i in range(150):
            shadow_hand.control_dofs_position(
                pose + i * delta_pose,
                motors_dof_idx,
            )
            scene.step(refresh_visualizer=False)

    import threading
    threading.Thread(target=grasp).start()
    scene.viewer.run()

if __name__ == "__main__":
    main()