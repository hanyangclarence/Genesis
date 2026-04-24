"""
Compute Optimal Tactile Point to Pixel Mapping

This script takes the raw group mappings created by create_tactile_mapping.py
and computes the optimal assignment between tactile points and pixels using
the Hungarian algorithm.

Input:
    - Raw mapping JSON from create_tactile_mapping.py
    - Original tactile grid JSON (for 2D positions)

Output:
    - Clean mapping file with point_to_pixel and pixel_to_points

Usage:
    python compute_tactile_mapping.py \
        --raw-mapping tactile_to_image_mapping.json \
        --tactile-grid merged_tactile_grid.json \
        --output tactile_pixel_mapping.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# Per-finger base colors for the preview plot (matches visualize_tactile_mapping_3d.py)
FINGER_COLORS = {
    "finger1": "#e41a1c",  # Thumb
    "finger2": "#377eb8",  # Index
    "finger3": "#4daf4a",  # Middle
    "finger4": "#984ea3",  # Ring
    "finger5": "#ff7f00",  # Pinky
    "palm":    "#a65628",  # Palm
}


def _link_color(link_name):
    for key, c in FINGER_COLORS.items():
        if key in link_name:
            return c
    return "#999999"


def preview_merged_points(tactile_points, title="Merged tactile points (close to continue)"):
    """Show all tactile points in the merged 2D coordinate system, colored by
    link. Blocks until the user closes the window.
    """
    _, ax = plt.subplots(figsize=(9, 10))
    by_link = {}
    for p in tactile_points:
        by_link.setdefault(p['link_name'], []).append(p['offset_pos_2d'])

    for link_name, coords in sorted(by_link.items()):
        arr = np.array(coords)
        ax.scatter(arr[:, 0], arr[:, 1],
                   c=_link_color(link_name), s=18,
                   edgecolors='black', linewidths=0.3,
                   label=f"{link_name} ({len(arr)})")
        # Annotate each link's cluster centroid with its name
        cx, cy = arr.mean(axis=0)
        ax.annotate(link_name, (cx, cy), fontsize=7, ha='center', va='center',
                    color='black', alpha=0.7)

    ax.set_xlabel("Horizontal (m)")
    ax.set_ylabel("Vertical (m)")
    ax.set_title(title)
    ax.set_aspect('equal')
    # Same orientation as create_tactile_mapping.py: Y+ left, Z+ down
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.legend(loc='upper right', fontsize=7, ncol=2)
    plt.tight_layout()
    plt.show()


# YZ offsets for visualization layout (must match create_tactile_mapping.py)
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


# Per-link collapse axis — must match create_tactile_mapping.py.
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
    """Parse tactile points; 2D layout uses each link's two in-plane local
    axes (the surface-normal axis, detected as the smallest-spread axis, is
    dropped) plus the per-link layout offset. Must match the projection used
    in create_tactile_mapping.py.
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


def compute_optimal_assignment(tactile_points, tactile_indices, pixels, image_shape):
    """
    Compute optimal assignment between tactile points and pixels.

    The assignment ensures that EVERY PIXEL gets at least one tactile point assigned.
    This is important for generating tactile images where every pixel needs a value.

    Strategy:
    - Each pixel is assigned to its nearest tactile point(s)
    - Multiple pixels can share the same tactile point
    - All pixels will have at least one tactile point

    Args:
        tactile_points: List of all tactile point dicts (with offset_pos_2d)
        tactile_indices: List of global indices for this mapping group
        pixels: List of (row, col) tuples for this mapping group
        image_shape: (rows, cols) of the tactile image

    Returns:
        point_to_pixel: dict mapping tactile_idx -> list of [row, col] (a point can map to multiple pixels)
        pixel_to_points: dict mapping "row,col" -> [tactile_idx, ...] (each pixel has at least one point)
    """
    tactile_indices = [int(x) for x in tactile_indices]
    pixels = [(int(row), int(col)) for row, col in pixels]

    n_tactile = len(tactile_indices)
    n_pixels = len(pixels)

    if n_tactile == 0 or n_pixels == 0:
        return {}, {}

    # Get 2D positions for tactile points (normalized within this group)
    tactile_2d = np.array([tactile_points[idx]['offset_pos_2d'] for idx in tactile_indices])

    # Normalize to [0, 1]
    t_min = tactile_2d.min(axis=0)
    t_max = tactile_2d.max(axis=0)
    t_range = t_max - t_min
    t_range[t_range < 1e-10] = 1.0  # Avoid division by zero
    tactile_2d_norm = (tactile_2d - t_min) / t_range

    # Get 2D positions for pixels (normalized within this group)
    pixels_arr = np.array(pixels, dtype=float)
    p_min = pixels_arr.min(axis=0)
    p_max = pixels_arr.max(axis=0)
    p_range = p_max - p_min
    p_range[p_range < 1e-10] = 1.0  # Avoid division by zero
    pixels_norm = (pixels_arr - p_min) / p_range

    # Swap pixel coordinates to match tactile_2d format
    # tactile_2d is (y_offset, z_offset) ~ (horizontal, vertical)
    # pixels are (row, col) where row is vertical, col is horizontal
    # So we need to swap: (row_norm, col_norm) -> (col_norm, row_norm)
    pixels_norm = pixels_norm[:, ::-1]

    # Flip horizontal direction: tactile Y increases right, but we want
    # increasing Y to map to decreasing col (to match hand orientation)
    pixels_norm[:, 0] = 1 - pixels_norm[:, 0]

    # Compute cost matrix (pairwise Euclidean distances)
    # cost_matrix[i, j] = distance from tactile point i to pixel j
    cost_matrix = np.zeros((n_tactile, n_pixels))
    for i in range(n_tactile):
        for j in range(n_pixels):
            cost_matrix[i, j] = np.linalg.norm(tactile_2d_norm[i] - pixels_norm[j])

    point_to_pixels = {tactile_indices[i]: [] for i in range(n_tactile)}
    pixel_to_points = {}

    # For EACH PIXEL, find the nearest tactile point
    # This ensures every pixel is covered
    for j in range(n_pixels):
        # Find nearest tactile point to this pixel
        distances = cost_matrix[:, j]
        nearest_i = int(np.argmin(distances))

        tactile_idx = tactile_indices[nearest_i]
        pixel = [pixels[j][0], pixels[j][1]]

        # Add this pixel to the tactile point's list
        if pixel not in point_to_pixels[tactile_idx]:
            point_to_pixels[tactile_idx].append(pixel)

        # Add tactile point to this pixel's list
        pixel_key = f"{pixel[0]},{pixel[1]}"
        if pixel_key not in pixel_to_points:
            pixel_to_points[pixel_key] = []
        if tactile_idx not in pixel_to_points[pixel_key]:
            pixel_to_points[pixel_key].append(tactile_idx)

    # Convert point_to_pixels: if a point has multiple pixels, keep all of them
    # If a point has no pixels, assign to its nearest pixel (and update pixel_to_points too!)
    point_to_pixel = {}
    for tactile_idx, pixel_list in point_to_pixels.items():
        if pixel_list:
            # Store all pixels this point maps to (use the first one as primary)
            point_to_pixel[tactile_idx] = pixel_list[0]
        else:
            # Fallback: find nearest pixel for this tactile point
            i = tactile_indices.index(tactile_idx)
            distances = cost_matrix[i, :]
            nearest_j = int(np.argmin(distances))
            pixel = [pixels[nearest_j][0], pixels[nearest_j][1]]
            point_to_pixel[tactile_idx] = pixel

            # Also add to pixel_to_points so the pixel knows about this tactile point
            pixel_key = f"{pixel[0]},{pixel[1]}"
            if pixel_key not in pixel_to_points:
                pixel_to_points[pixel_key] = []
            pixel_to_points[pixel_key].append(tactile_idx)

    return point_to_pixel, pixel_to_points


def main():
    parser = argparse.ArgumentParser(description="Compute optimal tactile-to-pixel mapping")
    parser.add_argument("--raw-mapping", type=str,
                        default="tactile_to_image_mapping.json",
                        help="Path to raw mapping JSON from create_tactile_mapping.py")
    parser.add_argument("--tactile-grid", type=str,
                        default="examples/tactile/merged_tactile_grid.json",
                        help="Path to original tactile grid JSON")
    parser.add_argument("--output", type=str,
                        default="tactile_pixel_mapping.json",
                        help="Output path for computed mapping")
    parser.add_argument("--no-preview", action="store_true",
                        help="Skip the 2D preview of merged tactile points before "
                             "running the assignment")
    args = parser.parse_args()

    # Load raw mapping
    raw_mapping_path = Path(args.raw_mapping)
    if not raw_mapping_path.exists():
        print(f"Error: Raw mapping file not found: {raw_mapping_path}")
        return

    print(f"Loading raw mapping from: {raw_mapping_path}")
    with open(raw_mapping_path, 'r') as f:
        raw_mapping = json.load(f)

    # Load tactile grid
    tactile_grid_path = Path(args.tactile_grid)
    if not tactile_grid_path.exists():
        print(f"Error: Tactile grid file not found: {tactile_grid_path}")
        return

    print(f"Loading tactile grid from: {tactile_grid_path}")
    with open(tactile_grid_path, 'r') as f:
        tactile_data = json.load(f)

    # Parse tactile points
    tactile_points = parse_tactile_points(tactile_data)
    print(f"Parsed {len(tactile_points)} tactile points")

    if not args.no_preview:
        print("Previewing merged 2D layout — close the window to continue. "
              "Edit LINK_YZ_OFFSETS in this file if you want to move clusters around.")
        preview_merged_points(tactile_points)

    # Get image shape
    image_shape = tuple(raw_mapping.get('image_shape', [24, 32]))
    print(f"Image shape: {image_shape}")

    # Process each mapping group
    mappings = raw_mapping.get('mappings', [])
    print(f"Processing {len(mappings)} mapping groups...")

    all_point_to_pixel = {}
    all_pixel_to_points = {}

    for mapping in mappings:
        mapping_idx = mapping['mapping_idx']

        # Extract tactile indices
        tactile_info = mapping['tactile_points']
        tactile_indices = [p['global_idx'] for p in tactile_info]

        # Extract pixels
        pixels = [tuple(p) for p in mapping['pixels']]

        print(f"  Mapping {mapping_idx}: {len(tactile_indices)} tactile points -> {len(pixels)} pixels")

        # Compute optimal assignment
        point_to_pixel, pixel_to_points = compute_optimal_assignment(
            tactile_points, tactile_indices, pixels, image_shape
        )

        # Merge results
        all_point_to_pixel.update(point_to_pixel)
        for key, value in pixel_to_points.items():
            if key not in all_pixel_to_points:
                all_pixel_to_points[key] = []
            all_pixel_to_points[key].extend(value)

    # Sanity check: verify point_to_pixel and pixel_to_points are consistent
    print(f"\n{'='*70}")
    print("SANITY CHECK: Verifying mapping consistency...")
    print(f"{'='*70}")

    errors = []

    # Check 1: Every point in point_to_pixel should appear in pixel_to_points
    for tactile_idx, pixel in all_point_to_pixel.items():
        pixel_key = f"{pixel[0]},{pixel[1]}"
        if pixel_key not in all_pixel_to_points:
            errors.append(f"Point {tactile_idx} maps to pixel {pixel}, but pixel not in pixel_to_points")
        elif tactile_idx not in all_pixel_to_points[pixel_key]:
            errors.append(f"Point {tactile_idx} maps to pixel {pixel}, but pixel_to_points[{pixel_key}] doesn't contain it")

    # Check 2: Every point in pixel_to_points should appear in point_to_pixel with matching pixel
    for pixel_key, point_indices in all_pixel_to_points.items():
        row, col = map(int, pixel_key.split(','))
        for tactile_idx in point_indices:
            if tactile_idx not in all_point_to_pixel:
                errors.append(f"Pixel {pixel_key} contains point {tactile_idx}, but point not in point_to_pixel")
            else:
                mapped_pixel = all_point_to_pixel[tactile_idx]
                if mapped_pixel[0] != row or mapped_pixel[1] != col:
                    errors.append(f"Pixel {pixel_key} contains point {tactile_idx}, but point_to_pixel[{tactile_idx}] = {mapped_pixel}")

    # Check 3: Count consistency
    total_points_in_pixel_to_points = sum(len(pts) for pts in all_pixel_to_points.values())
    if total_points_in_pixel_to_points != len(all_point_to_pixel):
        errors.append(f"Count mismatch: pixel_to_points has {total_points_in_pixel_to_points} entries, point_to_pixel has {len(all_point_to_pixel)}")

    if errors:
        print(f"FAILED: {len(errors)} errors found:")
        for err in errors[:10]:  # Show first 10 errors
            print(f"  - {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
    else:
        print("PASSED: point_to_pixel and pixel_to_points are consistent!")
        print(f"  - {len(all_point_to_pixel)} tactile points mapped")
        print(f"  - {len(all_pixel_to_points)} pixels used")
        print(f"  - {total_points_in_pixel_to_points} total point-to-pixel mappings")

    # Build output structure
    output = {
        'image_shape': list(image_shape),
        'num_tactile_points': len(tactile_points),
        'num_mapped_points': len(all_point_to_pixel),
        'num_pixels_used': len(all_pixel_to_points),
        'point_to_pixel': {str(k): v for k, v in all_point_to_pixel.items()},
        'pixel_to_points': all_pixel_to_points,
        # Also include per-link breakdown for convenience
        'per_link_mapping': {},
    }

    # Build per-link mapping
    for tactile_idx, pixel in all_point_to_pixel.items():
        point = tactile_points[tactile_idx]
        link_name = point['link_name']
        point_idx = point['point_idx']

        if link_name not in output['per_link_mapping']:
            output['per_link_mapping'][link_name] = {}

        output['per_link_mapping'][link_name][str(point_idx)] = {
            'global_idx': tactile_idx,
            'pixel': pixel,
            'local_pos': [float(x) for x in point['local_pos']],
        }

    # Save output
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*70}")
    print("MAPPING COMPLETE")
    print(f"{'='*70}")
    print(f"Total tactile points: {len(tactile_points)}")
    print(f"Mapped points: {len(all_point_to_pixel)}")
    print(f"Pixels used: {len(all_pixel_to_points)}")
    print(f"Unmapped points: {len(tactile_points) - len(all_point_to_pixel)}")
    print(f"\nOutput saved to: {output_path}")

    # Print per-link summary
    print(f"\nPer-link summary:")
    for link_name in sorted(output['per_link_mapping'].keys()):
        link_mapping = output['per_link_mapping'][link_name]
        print(f"  {link_name}: {len(link_mapping)} points mapped")

    # Check for unmapped points
    mapped_indices = set(all_point_to_pixel.keys())
    unmapped = [i for i in range(len(tactile_points)) if i not in mapped_indices]
    if unmapped:
        print(f"\nWarning: {len(unmapped)} unmapped tactile points!")
        # Group by link
        unmapped_by_link = {}
        for idx in unmapped:
            link_name = tactile_points[idx]['link_name']
            if link_name not in unmapped_by_link:
                unmapped_by_link[link_name] = []
            unmapped_by_link[link_name].append(idx)

        for link_name, indices in unmapped_by_link.items():
            print(f"  {link_name}: {len(indices)} unmapped")


if __name__ == "__main__":
    main()
