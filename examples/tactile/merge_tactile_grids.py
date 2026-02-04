#!/usr/bin/env python3
"""
Merge multiple tactile grid JSON files into a single file.
Only keeps local coordinates for each point.
"""

import json
import argparse
from pathlib import Path


def merge_grids(input_files, output_file):
    """
    Merge multiple grid JSON files into one.

    Args:
        input_files: List of input JSON file paths
        output_file: Output JSON file path
    """
    merged_data = {
        "hand_type": None,
        "links": {}
    }

    total_points = 0

    for input_file in input_files:
        print(f"Reading {input_file}...")
        with open(input_file, 'r') as f:
            data = json.load(f)

        # Use hand_type from first file
        if merged_data["hand_type"] is None:
            merged_data["hand_type"] = data.get("hand_type", "unknown")

        # Merge links
        for link_name, link_data in data.get("links", {}).items():
            # Extract only local coordinates as a list of 3D points
            simplified_points = []
            for point in link_data.get("points", []):
                simplified_points.append(point["local"])

            if link_name in merged_data["links"]:
                print(f"  Warning: Link '{link_name}' already exists")
                assert merged_data["links"][link_name]["link_idx_local"] == link_data.get("link_idx_local"), \
                    f"Link index mismatch for '{link_name}'"
                merged_data["links"][link_name]["points"].extend(simplified_points)
                merged_data["links"][link_name]["num_points"] += len(simplified_points)
            else:
                merged_data["links"][link_name] = {
                    "link_idx_local": link_data.get("link_idx_local"),
                    "points": simplified_points,
                    "num_points": len(simplified_points)
                }

            total_points += len(simplified_points)
            print(f"  Added {link_name}: {len(simplified_points)} points")

    merged_data["total_points"] = total_points

    # Write output
    print(f"\nWriting merged data to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(merged_data, f, indent=2)

    print(f"Done! Merged {len(merged_data['links'])} links with {total_points} total points")


def main():
    parser = argparse.ArgumentParser(description="Merge tactile grid JSON files")
    parser.add_argument("input_files", nargs="+", help="Input JSON files to merge")
    parser.add_argument("-o", "--output", required=True, help="Output merged JSON file")

    args = parser.parse_args()

    # Verify input files exist
    for input_file in args.input_files:
        if not Path(input_file).exists():
            print(f"Error: Input file not found: {input_file}")
            return 1

    merge_grids(args.input_files, args.output)
    return 0


if __name__ == "__main__":
    exit(main())