# Genesis with tactile sensing

A fork of [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) that adds a dense tactile
sensor, `gs.sensors.TactileField`, and tools to place tactile points on a hand and map them to a
24x32 tactile image.

The v5 Wuji hand grids and mappings are in `examples/tactile/` (`full_hand_tactile*_v5.json`,
`tactile_pixel_mapping*_v5.json`). They were generated on the left v5 hand and mirrored to the
right hand; the 2D layout used for mapping (`examples/tactile/tactile_layout.py`) is tuned for the
left hand.

## Tactile point pipeline

Run from the repository root. Scripts default to the right v5 hand in
`genesis/assets/urdf/wujihand_v5/`; pass `--urdf` for another hand.

**1. Generate a tactile array for each link.** Links differ in shape and size, so each link, or
each group of links that shares settings (e.g. the same segment of the four fingers), gets its own
run and its own output file. A run samples the link's palm-facing surface densely and keeps a
regular grid of points on it, in the link's local frame. The main settings are the grid spacing
(`--grid-spacing-h/-v`), which surfaces count as palm-facing (`--palm-threshold`,
`--palm-facing-angle`), and the height range along the link (`--z-min/--z-max`). Repeat until every
sensing link is covered:

```bash
python examples/tactile/generate_grid_tactile_points.py \
    --links finger2_link3,finger3_link3,finger4_link3,finger5_link3 \
    --grid-spacing-h 0.003 --grid-spacing-v 0.0038 --palm-threshold -0.3 \
    --z-min 0.003 --z-max 0.027 --dense-samples 200000 --output finger_link3.json
```

**2. Remove stray points (optional).** Clean up points that landed in the wrong place on one link of
a per-link file (lasso-select, `d` to delete, `s` to save):

```bash
python examples/tactile/edit_tactile_points.py palm.json --link palm_link
```

**3. Merge into a single config.** Combine the per-link files into one tactile grid covering the
whole hand. This is the file the sensor and the mapping tools read. The order of the input files
sets the order of links, and therefore of points, in the sensor readout:

```bash
python examples/tactile/merge_tactile_grids.py finger*.json palm.json --output full_hand_tactile.json
```

**4. Visualize** the points on the hand:

```bash
python examples/tactile/visualize_tactile_sensor.py --tactile-grid full_hand_tactile.json
```

Steps 5 and 6 map the simulated points onto the real tactile glove's readout. The glove reports a
24x32 tactile map, while the simulation reports one force per tactile point (643 for the v5 grid).
A mapping from points to pixels lets the simulated forces be rendered as the same 24x32 map
(`genesis.vis.TactileVisualizer`), so simulated and real tactile readings can be compared directly,
for example in a reward.

**5. Create the raw mapping.** Pair each region of the hand with the block of pixels that region
covers on the glove's map. The tool shows all points flattened into 2D next to the 24x32 grid;
select a region's points (e.g. one finger segment) and its pixels, then press `m` to record that
pair. Repeat for every region and press `s` to save:

```bash
python examples/tactile/create_tactile_mapping.py \
    --tactile-grid full_hand_tactile.json --output tactile_to_image_mapping.json
```

**6. Compute the final point-to-pixel mapping.** Within each recorded pair, every pixel takes its
nearest tactile point in the 2D layout, and any point left without a pixel goes to its nearest pixel.
A pixel's value is then the average force of its points. The output file is what
`TactileVisualizer` loads:

```bash
python examples/tactile/compute_tactile_mapping.py \
    --raw-mapping tactile_to_image_mapping.json --tactile-grid full_hand_tactile.json \
    --output tactile_pixel_mapping.json
```

**7. Visualize the mapping.** Shows the points in 3D on the hand next to the 24x32 map: click a
point to highlight its pixel, or a pixel to highlight its points (`c` or right-click clears). Pass
the URDF the grid was generated for:

```bash
python examples/tactile/visualize_tactile_mapping_3d.py \
    --tactile-grid full_hand_tactile.json --mapping tactile_pixel_mapping.json \
    --urdf genesis/assets/urdf/wujihand_v5/wujihand_right_v5.urdf
```

## License

Apache 2.0, as upstream Genesis. See [LICENSE](LICENSE).
