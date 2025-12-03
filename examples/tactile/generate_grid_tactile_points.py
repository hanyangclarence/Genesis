"""
Generate Grid-Like Tactile Points on Palm Surfaces

This script generates a proper grid of tactile points by:
1. Dense sampling on palm-facing surfaces
2. Projecting to 2D plane (YZ for vertical surfaces, XY for horizontal)
3. Organizing into rows by Z-coordinate intervals
4. Selecting evenly-spaced points along each row
5. Ensuring uniform coverage on curved and flat surfaces

This produces a clean grid pattern suitable for tactile sensor arrays.
"""

import argparse
import json
import numpy as np
from pathlib import Path

import genesis as gs
from genesis.utils import geom as gu


def get_palm_facing_triangles(vertices, faces, normals, hand_forward_dir, threshold=-0.3):
    """Filter triangles to only include those facing the palm.

    Args:
        vertices: Mesh vertices
        faces: Triangle faces
        normals: Face normals
        hand_forward_dir: Direction vector pointing from palm (toward fingers)
        threshold: Dot product threshold. More negative = stricter filtering.
                   -0.5: Very strict (only strongly palm-facing surfaces)
                   -0.3: Moderate (recommended, filters out sides/back)
                   -0.1: Loose (may include side surfaces)
    """
    # Calculate face centers
    face_centers = vertices[faces].mean(axis=1)

    # Palm-facing surfaces have normals pointing opposite to hand_forward_dir
    dot_products = np.sum(normals * hand_forward_dir, axis=1)
    palm_face_mask = dot_products < threshold

    palm_faces = np.where(palm_face_mask)[0]
    return palm_faces, palm_face_mask


def dense_sample_on_faces(vertices, faces, face_indices, num_samples=2000):
    """Densely sample points on selected faces."""
    if len(face_indices) == 0:
        return np.array([]), np.array([]), np.array([])

    # Calculate face properties
    selected_faces = faces[face_indices]
    v0 = vertices[selected_faces[:, 0]]
    v1 = vertices[selected_faces[:, 1]]
    v2 = vertices[selected_faces[:, 2]]

    # Face normals and areas
    cross = np.cross(v1 - v0, v2 - v0)
    areas = np.linalg.norm(cross, axis=1) / 2.0
    normals = cross / (2.0 * areas[:, None] + 1e-10)

    # Sample faces proportional to area
    probabilities = areas / areas.sum()
    probabilities = probabilities / probabilities.sum()

    sampled_face_idx = np.random.choice(
        len(face_indices), size=num_samples, p=probabilities
    )

    # Random barycentric coordinates for uniform distribution
    r1 = np.random.random(num_samples)
    r2 = np.random.random(num_samples)

    # Ensure points are inside triangle
    mask = r1 + r2 > 1
    r1[mask] = 1 - r1[mask]
    r2[mask] = 1 - r2[mask]

    barycentric = np.stack([1 - r1 - r2, r1, r2], axis=1)

    # Interpolate positions
    actual_face_indices = face_indices[sampled_face_idx]
    sampled_faces = faces[actual_face_indices]

    points = (barycentric[:, 0:1] * vertices[sampled_faces[:, 0]] +
              barycentric[:, 1:2] * vertices[sampled_faces[:, 1]] +
              barycentric[:, 2:3] * vertices[sampled_faces[:, 2]])

    sampled_normals = normals[sampled_face_idx]

    return points, sampled_normals, barycentric


def select_grid_points(points, normals, grid_spacing=0.01, num_rows=None, num_cols=None, z_margin=0.001):
    """
    Select grid points from dense samples with adaptive density based on surface geometry.

    Algorithm:
    1. Use PCA to find the principal axes of the surface
    2. Project points to surface-aligned 2D coordinate system
    3. Divide vertical extent into horizontal slices
    4. For each slice, determine the width and place points with uniform spacing
    5. This creates more points where the surface is wider (e.g., middle of palm)
       and fewer points where it's narrower (e.g., bottom of palm)

    Args:
        points: (N, 3) dense sampled points
        normals: (N, 3) surface normals
        grid_spacing: target distance between adjacent grid points (in meters)
        num_rows: (optional) if provided, overrides grid_spacing for rows
        num_cols: (optional) if provided, overrides grid_spacing for columns
        z_margin: margin for row selection (in meters)

    Returns:
        grid_points: (M, 3) selected grid points (M varies based on geometry)
        grid_normals: (M, 3) corresponding normals
        grid_indices: (M,) indices into original points array
    """
    if len(points) == 0:
        return np.array([]), np.array([]), np.array([])

    # Use PCA to find surface-aligned coordinate system
    # This ensures consistent grid generation regardless of surface orientation
    points_centered = points - points.mean(axis=0)

    # Compute covariance matrix
    cov = np.cov(points_centered.T)

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    # Sort eigenvectors by eigenvalues (largest to smallest)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Principal axes:
    # - axis 0: direction of maximum variance
    # - axis 1: direction of medium variance
    # - axis 2: direction of minimum variance (normal to surface)

    # Project points to PCA coordinate system
    points_pca = points_centered @ eigenvectors

    # Determine which axis should be "height" (vertical) vs "width" (horizontal)
    # Use the average normal direction to guide this choice
    avg_normal = normals.mean(axis=0)
    avg_normal = avg_normal / (np.linalg.norm(avg_normal) + 1e-10)

    # The "up" direction for the grid should be perpendicular to avg_normal
    # and should align with the direction of maximum spread
    # For a cylinder: the axis runs along max variance, should be "height"
    # For a flat surface: max variance is "width", medium variance is "height"

    # Check alignment of each PCA axis with the average normal
    axis0_normal_dot = abs(np.dot(eigenvectors[:, 0], avg_normal))
    axis1_normal_dot = abs(np.dot(eigenvectors[:, 1], avg_normal))

    # If axis 0 is more perpendicular to normal (smaller dot product),
    # it lies in the surface plane and should be the height direction
    # This happens for elongated surfaces like cylinder sides
    if axis0_normal_dot < axis1_normal_dot:
        # axis 0 is more tangent to surface -> use as height
        # axis 1 is more normal-aligned -> use as width
        height_dim = 0
        width_dim = 1
    else:
        # axis 0 is more normal-aligned -> use as width
        # axis 1 is more tangent to surface -> use as height
        height_dim = 1
        width_dim = 0

    # Use only the two principal axes with largest variance
    points_2d = points_pca[:, :2]

    # Find bounds with small margin (10% on each side)
    margin = 0.1
    min_2d = points_2d.min(axis=0)
    max_2d = points_2d.max(axis=0)
    range_2d = max_2d - min_2d

    grid_min = min_2d + range_2d * margin
    grid_max = max_2d - range_2d * margin

    # Determine number of rows (horizontal slices)
    if num_rows is not None:
        n_rows = num_rows
    else:
        height = grid_max[height_dim] - grid_min[height_dim]
        n_rows = max(3, int(np.round(height / grid_spacing)))

    # Create row centers
    row_centers = np.linspace(grid_min[height_dim], grid_max[height_dim], n_rows)

    # Determine global column positions based on the overall width extent
    # Use grid_spacing to determine number of global columns
    global_width = grid_max[width_dim] - grid_min[width_dim]

    if num_cols is not None:
        n_global_cols = num_cols
    else:
        n_global_cols = max(1, int(np.round(global_width / grid_spacing)))

    # Create global column positions spanning the entire width
    if n_global_cols == 1:
        global_col_positions = np.array([(grid_min[width_dim] + grid_max[width_dim]) / 2])
    else:
        global_col_positions = np.linspace(grid_min[width_dim], grid_max[width_dim], n_global_cols)

    grid_points = []
    grid_normals = []
    grid_indices = []
    used_indices = set()

    # For each row (horizontal slice)
    for row_idx, row_center in enumerate(row_centers):
        # Find points in this horizontal slice
        row_mask = np.abs(points_2d[:, height_dim] - row_center) < (range_2d[height_dim] / (2 * n_rows))

        if not np.any(row_mask):
            continue

        row_points_2d = points_2d[row_mask]
        row_point_indices = np.where(row_mask)[0]

        # Determine actual width of this row
        row_width_min = row_points_2d[:, width_dim].min()
        row_width_max = row_points_2d[:, width_dim].max()

        # For each global column position, check if it falls within this row's width
        for col_pos in global_col_positions:
            # Skip this column if it's outside the row's actual width
            # Add small tolerance for numerical precision
            tolerance = grid_spacing * 0.1
            if col_pos < row_width_min - tolerance or col_pos > row_width_max + tolerance:
                continue

            # Create target grid point in 2D
            target_2d = np.zeros(2)
            target_2d[height_dim] = row_center
            target_2d[width_dim] = col_pos

            # Find closest point in the original dense samples
            distances = np.linalg.norm(points_2d - target_2d, axis=1)

            # Find closest point that hasn't been used yet
            sorted_indices = np.argsort(distances)
            selected_idx = None

            for idx in sorted_indices:
                if idx not in used_indices:
                    selected_idx = idx
                    break

            if selected_idx is None:
                # All points used - skip this grid point
                continue

            used_indices.add(selected_idx)

            grid_points.append(points[selected_idx])
            grid_normals.append(normals[selected_idx])
            grid_indices.append(selected_idx)

    if len(grid_points) == 0:
        return np.array([]), np.array([]), np.array([])

    return np.array(grid_points), np.array(grid_normals), np.array(grid_indices)


def main():
    parser = argparse.ArgumentParser(description="Generate grid-like tactile points")
    parser.add_argument("--hand", type=str, default="wuji")
    parser.add_argument("--output", type=str, default="grid_tactile_points.json")
    parser.add_argument("--grid-spacing", type=float, default=None,
                        help="Target spacing between grid points in meters (e.g., 0.01 for 1cm). "
                             "If set, grid adapts to surface geometry with uniform spacing.")
    parser.add_argument("--grid-rows", type=int, default=5,
                        help="Number of rows in tactile grid (used if --grid-spacing not set)")
    parser.add_argument("--grid-cols", type=int, default=5,
                        help="Number of columns in tactile grid (used if --grid-spacing not set)")
    parser.add_argument("--links", type=str, default=None,
                        help="Comma-separated link names")
    parser.add_argument("--dense-samples", type=int, default=2000,
                        help="Number of dense samples per geom for grid selection")
    parser.add_argument("--palm-threshold", type=float, default=-0.5,
                        help="Dot product threshold for palm detection (more negative = stricter). "
                             "Recommended: -0.5 (very strict), -0.3 (moderate), -0.1 (loose)")
    parser.add_argument("--palm-facing-angle", type=float, default=180.0,
                        help="Palm-facing direction angle in degrees (rotation around Z-axis). "
                             "0=+X, 90=+Y, 180=-X, 270=-Y. Default: 180 (-X direction)")
    parser.add_argument("--z-min", type=float, default=None,
                        help="Minimum Z height in link's local coordinate (meters). "
                             "Only generate tactile points above this height.")
    parser.add_argument("--z-max", type=float, default=None,
                        help="Maximum Z height in link's local coordinate (meters). "
                             "Only generate tactile points below this height.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("="*70)
    print("GENERATE GRID-LIKE TACTILE POINTS ON PALM SURFACES")
    print("="*70)
    if args.grid_spacing is not None:
        print(f"Grid spacing: {args.grid_spacing*1000:.1f}mm (adaptive to geometry)")
    else:
        print(f"Grid size: {args.grid_rows} rows × {args.grid_cols} columns")
    print(f"Dense sampling: {args.dense_samples} points per geom")
    print(f"Palm threshold: {args.palm_threshold} (more negative = stricter)")
    if args.z_min is not None or args.z_max is not None:
        z_min_str = f"{args.z_min:.4f}" if args.z_min is not None else "-inf"
        z_max_str = f"{args.z_max:.4f}" if args.z_max is not None else "+inf"
        print(f"Z-height range (link local): [{z_min_str}, {z_max_str}] meters")

    np.random.seed(args.seed)

    ########################## init ##########################
    gs.init(backend=gs.cpu)

    ########################## create scene ##########################
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        show_viewer=False,
    )

    ########################## entities ##########################
    scene.add_entity(gs.morphs.Plane())

    hand = scene.add_entity(
        gs.morphs.URDF(
            file="assets/wujihand-urdf/urdf/right.urdf",
            merge_fixed_links=False,
            fixed=True,
            pos=(0, 0, 0.1),
            euler=(90, 0, 0),
        ),
    )
    print(f"\n✓ Loaded hand: {args.hand}")

    scene.build()

    # Parse link filter
    target_links = None
    if args.links:
        target_links = set(args.links.split(','))
        print(f"  Filtering to links: {target_links}")

    ########################## Generate grid points ##########################
    print(f"\n{'='*70}")
    print("GENERATING GRID TACTILE POINTS")
    print(f"{'='*70}")

    tactile_data = {
        'hand_type': args.hand,
        'sampling_method': 'palm_grid_adaptive' if args.grid_spacing else 'palm_grid',
        'grid_spacing': args.grid_spacing,
        'grid_rows': args.grid_rows if args.grid_spacing is None else None,
        'grid_cols': args.grid_cols if args.grid_spacing is None else None,
        'z_min': args.z_min,
        'z_max': args.z_max,
        'links': {}
    }

    total_points = 0

    for link in hand.links:
        link_name = link.name

        if target_links and link_name not in target_links:
            continue

        print(f"\n[{link.idx_local:2d}] {link_name}")

        # Get link pose
        link_pos = link.get_pos().cpu().numpy()
        link_quat = link.get_quat().cpu().numpy()

        # Calculate palm-facing direction from angle (rotation around Z-axis in link frame)
        angle_rad = np.radians(args.palm_facing_angle)
        palm_forward_local = np.array([np.cos(angle_rad), np.sin(angle_rad), 0.0]) 

        link_data = {
            'link_idx_local': link.idx_local,
            'points': []
        }

        for geom_idx, geom in enumerate(link.geoms):
            if not hasattr(geom, 'init_verts') or geom.init_verts is None:
                continue

            # Get vertices in geom's local frame
            vertices_geom_frame = geom.init_verts
            faces = geom.init_faces if hasattr(geom, 'init_faces') else np.array([])

            if len(faces) == 0:
                continue

            # Transform vertices from geom frame to link frame
            import torch
            geom_pos = np.array(geom.init_pos)
            geom_quat = np.array(geom.init_quat)

            vertices_geom_torch = torch.tensor(vertices_geom_frame, dtype=gs.tc_float)
            geom_pos_torch = torch.tensor(geom_pos, dtype=gs.tc_float)
            geom_quat_torch = torch.tensor(geom_quat, dtype=gs.tc_float)

            # Transform: geom frame -> link frame
            vertices_link_torch = gu.transform_by_trans_quat(vertices_geom_torch, geom_pos_torch, geom_quat_torch)
            vertices_local = vertices_link_torch.cpu().numpy()

            # Calculate face normals in link frame
            v0 = vertices_local[faces[:, 0]]
            v1 = vertices_local[faces[:, 1]]
            v2 = vertices_local[faces[:, 2]]
            cross = np.cross(v1 - v0, v2 - v0)
            areas = np.linalg.norm(cross, axis=1) / 2.0
            normals = cross / (2.0 * areas[:, None] + 1e-10)

            # Filter palm-facing triangles (now in link frame)
            palm_faces, _ = get_palm_facing_triangles(
                vertices_local, faces, normals, palm_forward_local, threshold=args.palm_threshold
            )

            print(f"  Geom {geom_idx}: {len(vertices_local)} verts, {len(faces)} faces")
            print(f"    → Palm-facing: {len(palm_faces)} faces ({100*len(palm_faces)/len(faces):.1f}%)")

            if len(palm_faces) == 0:
                continue

            # Step 1: Dense sampling
            dense_points, dense_normals, _ = dense_sample_on_faces(
                vertices_local, faces, palm_faces, num_samples=args.dense_samples
            )

            print(f"    → Dense sampled: {len(dense_points)} points")

            # Step 2: Select grid points
            if args.grid_spacing is not None:
                # Adaptive grid based on spacing
                grid_points, grid_normals, grid_indices = select_grid_points(
                    dense_points, dense_normals,
                    grid_spacing=args.grid_spacing
                )
            else:
                # Fixed grid size
                grid_points, grid_normals, grid_indices = select_grid_points(
                    dense_points, dense_normals,
                    num_rows=args.grid_rows,
                    num_cols=args.grid_cols
                )

            print(f"    → Grid selected: {len(grid_points)} points")

            # Apply Z-height filtering if specified
            if args.z_min is not None or args.z_max is not None:
                z_mask = np.ones(len(grid_points), dtype=bool)
                if args.z_min is not None:
                    z_mask &= (grid_points[:, 2] >= args.z_min)
                if args.z_max is not None:
                    z_mask &= (grid_points[:, 2] <= args.z_max)

                grid_points = grid_points[z_mask]
                grid_normals = grid_normals[z_mask]

                print(f"    → After Z-filter [{args.z_min}, {args.z_max}]: {len(grid_points)} points")

            # Store points
            import torch
            for i in range(len(grid_points)):
                point_local_torch = torch.tensor(grid_points[i], dtype=gs.tc_float).reshape(1, 3)
                link_pos_torch = torch.tensor(link_pos, dtype=gs.tc_float)
                link_quat_torch = torch.tensor(link_quat, dtype=gs.tc_float)

                point_world_torch = gu.transform_by_trans_quat(point_local_torch, link_pos_torch, link_quat_torch)
                point_world = point_world_torch[0].cpu().numpy()

                normal_local_torch = torch.tensor(grid_normals[i], dtype=gs.tc_float).reshape(1, 3)
                normal_world_torch = gu.transform_by_quat(normal_local_torch, link_quat_torch)
                normal_world = normal_world_torch[0].cpu().numpy()

                link_data['points'].append({
                    'local': grid_points[i].tolist(),
                    'world': point_world.tolist(),
                    'normal_local': grid_normals[i].tolist(),
                    'normal_world': normal_world.tolist(),
                    'geom_idx': geom_idx,
                })

                total_points += 1

        if len(link_data['points']) > 0:
            link_data['num_points'] = len(link_data['points'])
            tactile_data['links'][link_name] = link_data

    ########################## Save ##########################
    tactile_data['total_points'] = total_points

    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(tactile_data, f, indent=2)

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Total grid points: {total_points}")
    if args.grid_spacing is not None:
        print(f"Grid spacing: {args.grid_spacing*1000:.1f}mm (adaptive to geometry)")
    else:
        print(f"Grid size: {args.grid_rows}×{args.grid_cols}")
    print(f"\nPoints per link:")
    for link_name, link_data in tactile_data['links'].items():
        print(f"  {link_name:30s}: {link_data['num_points']:3d} points")

    print(f"\n✓ Saved to: {output_path.absolute()}")
    print(f"\nVisualize with:")
    print(f"  python examples/sensors/visualize_tactile_points.py --input {args.output}")


if __name__ == "__main__":
    main()
