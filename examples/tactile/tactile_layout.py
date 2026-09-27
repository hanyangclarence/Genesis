"""
2D layout of the hand's tactile points, shared by create_tactile_mapping.py and
compute_tactile_mapping.py.

Each link's points are projected onto two of the link's local axes and shifted by a
per-link offset, so the whole hand lies flat in one 2D view. The mapping tool shows
this layout, and the assignment of pixels to tactile points is computed in it.

The per-link settings are tuned for the left v5 hand, which the tactile grids are
generated on (see README.md).
"""

import numpy as np


# Per-link 2D offsets for the flattened hand layout
LINK_YZ_OFFSETS = {
    "palm_link": (0.0, 0.0),
    "finger1_link1": (-0.05, 0.02),
    "finger1_link2": (-0.05, 0.05),
    "finger1_link3": (-0.05, 0.08),
    "finger1_link4": (-0.05, 0.11),
    "finger1_tip_link": (-0.05, 0.14),
    "finger2_link1": (-0.024, 0.055),
    "finger2_link2": (-0.024, 0.095),
    "finger2_link3": (-0.024, 0.135),
    "finger2_link4": (-0.024, 0.175),
    "finger2_tip_link": (-0.024, 0.205),
    "finger3_link1": (-0.003, 0.055),
    "finger3_link2": (-0.003, 0.095),
    "finger3_link3": (-0.003, 0.135),
    "finger3_link4": (-0.003, 0.175),
    "finger3_tip_link": (-0.003, 0.205),
    "finger4_link1": (0.015, 0.049),
    "finger4_link2": (0.015, 0.089),
    "finger4_link3": (0.015, 0.129),
    "finger4_link4": (0.015, 0.169),
    "finger4_tip_link": (0.015, 0.199),
    "finger5_link1": (0.031, 0.034),
    "finger5_link2": (0.031, 0.074),
    "finger5_link3": (0.031, 0.124),
    "finger5_link4": (0.031, 0.154),
    "finger5_tip_link": (0.031, 0.174),
}


# Per-link local axis dropped when flattening (0=X, 1=Y, 2=Z). Links not listed drop X
# and keep YZ; only links whose sensing surface faces along another axis need an entry.
LINK_COLLAPSE_AXIS = {
    "finger2_link2": 1,  # drop Y, keep XZ
    "finger3_link2": 1,
    "finger4_link2": 1,
    "finger5_link2": 1,
    "finger1_link2": 1,
}


def get_collapse_axis(link_name):
    return LINK_COLLAPSE_AXIS.get(link_name, 0)


def parse_tactile_points(tactile_data):
    """Flatten a tactile grid into one list of points, in global (JSON) order.

    Each link's points keep their two in-plane local axes (LINK_COLLAPSE_AXIS names the
    dropped one) and are shifted by the link's LINK_YZ_OFFSETS entry. Returns a list of
    dicts with link_name, point_idx, local_pos, offset_pos_2d, and collapse_axis.
    """
    tactile_points = []
    links_data = tactile_data.get('links', {})
    axis_names = ['X', 'Y', 'Z']

    for link_name, link_data in links_data.items():
        points = link_data.get('points', [])
        if not points:
            continue

        collapse_axis = get_collapse_axis(link_name)
        keep_axes = [a for a in range(3) if a != collapse_axis]
        offset = LINK_YZ_OFFSETS.get(link_name, (0.0, 0.0))
        a, b = keep_axes

        print(f"  {link_name}: drop {axis_names[collapse_axis]}, "
              f"keep {axis_names[a]}{axis_names[b]} ({len(points)} points)")

        for point_idx, point in enumerate(points):
            local_pos = np.array(point['local'] if isinstance(point, dict) else point)
            offset_pos_2d = np.array([local_pos[a] + offset[0],
                                       local_pos[b] + offset[1]])
            tactile_points.append({
                'link_name': link_name,
                'point_idx': point_idx,
                'local_pos': local_pos,
                'offset_pos_2d': offset_pos_2d,
                'collapse_axis': collapse_axis,
            })

    return tactile_points
