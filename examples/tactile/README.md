# Tactile sensing

Dense tactile sensing on the Wuji hand with `gs.sensors.TactileField`, and the pipeline that
places tactile points on the hand's surface and maps them to a 24x32 tactile image.

| File | Purpose |
| --- | --- |
| `tactile_field_hand.py` | Demo: a `TactileField` sensor on every link in the grid, the right hand closing on two cylinders, and the live 24x32 tactile map. |
| `full_hand_tactile_v5.json`, `tactile_pixel_mapping_v5.json` | v5 tactile grid (643 points on 16 links, in each link's local frame) and its 24x32 pixel mapping for the right hand (`genesis/assets/urdf/wujihand_v5/wujihand_right_v5.urdf`). |
| `full_hand_tactile_left_v5.json`, `tactile_pixel_mapping_left_v5.json` | The same for the left v5 hand. |
| `generate_grid_tactile_points.py`, `edit_tactile_points.py`, `merge_tactile_grids.py` | Build a tactile grid (steps 1-3 below). |
| `create_tactile_mapping.py`, `compute_tactile_mapping.py` | Build the pixel mapping (steps 4-5 below). Both use the 2D hand layout in `tactile_layout.py`. |
| `visualize_tactile_sensor.py` | Shows a grid's points on the hand. |
| `visualize_tactile_mapping_3d.py` | Shows which pixel each tactile point maps to, and vice versa. |

Run everything from the repository root.

```bash
python examples/tactile/tactile_field_hand.py
```

## How the v5 grid was built

The v5 grid and mapping were generated on the left v5 hand, with the commands in steps 1-5
below. This repository ships only the right v5 hand, so pass the left hand's URDF with `--urdf`
(`LEFT_URDF` below). The right-hand files are the left-hand ones mirrored: each link's points
are reflected into the right hand's link frame, and the image columns are flipped.

### Step 1: Generate tactile points for groups of links

`generate_grid_tactile_points.py` samples the palm-facing surface of each link's collision mesh
densely, then keeps points on a regular grid. Each command covers a group of links that share
the same parameters:

```bash
G="python examples/tactile/generate_grid_tactile_points.py --urdf $LEFT_URDF --dense-samples 200000"

$G --links finger1_link4 --grid-spacing-h 0.003 --grid-spacing-v 0.003 --palm-threshold -0.3 --z-min 0.0005 --output finger1_link4.json
$G --links finger1_link3 --grid-spacing-h 0.003 --grid-spacing-v 0.003 --palm-threshold -0.5 --z-min 0.0 --z-max 0.026 --output finger1_link3.json
$G --links finger1_link2 --grid-spacing-h 0.003 --grid-spacing-v 0.003 --palm-threshold -0.5 --z-min 0.002 --z-max 0.025 --palm-facing-angle 110 --output finger1_link2.json
$G --links finger2_link2,finger3_link2,finger4_link2 --grid-spacing-h 0.003 --grid-spacing-v 0.004 --palm-threshold -0.3 --z-min -0.005 --z-max 0.037 --palm-facing-angle 90 --output finger_link2_1.json
$G --links finger5_link2 --grid-spacing-h 0.003 --grid-spacing-v 0.004 --palm-threshold -0.5 --z-min -0.005 --z-max 0.037 --palm-facing-angle 90 --output finger_link2_2.json
$G --links finger2_link3,finger3_link3,finger4_link3,finger5_link3 --grid-spacing-h 0.003 --grid-spacing-v 0.0038 --palm-threshold -0.3 --z-min 0.003 --z-max 0.027 --output finger_link3.json
$G --links finger2_link4,finger3_link4,finger4_link4,finger5_link4 --grid-spacing-h 0.003 --grid-spacing-v 0.003 --palm-threshold -0.17 --z-min 0.001 --output finger_link4.json
$G --links palm_link --grid-spacing-h 0.005 --grid-spacing-v 0.005 --palm-threshold -0.8 --output palm.json
```

- `--grid-spacing-h` / `--grid-spacing-v`: point spacing across and along the link, in meters.
- `--palm-facing-angle`: direction in the link's XY plane that the palm faces away from, in
  degrees (0 = +X, 90 = +Y, 180 = -X, the default).
- `--palm-threshold`: how strictly a triangle must face the palm to get points (more negative is
  stricter).
- `--z-min` / `--z-max`: keep only points within this height range along the link's Z axis.
- `--urdf`: the hand model (defaults to the right v5 hand shipped here).

### Step 2: Remove stray points by hand

```bash
python examples/tactile/edit_tactile_points.py palm.json --link palm_link
```

Lasso-select points in any 2D view, press `d` to delete them and `s` to save (overwrites the file).
For the v5 grid, step 1 reproduces every shipped point exactly; points were then removed this way
from `palm_link` (210 → 141), `finger1_link4` (58 → 51), `finger2_link2`-`finger4_link2`
(42 → 40 each), and `finger2_link4`-`finger5_link4` (down to 30 each).

### Step 3: Merge the per-group files

```bash
python examples/tactile/merge_tactile_grids.py \
    finger1_*.json finger_link*.json palm.json \
    --output full_hand_tactile.json
```

The order of links in the merged file fixes the global point order, which the sensor readout
and the pixel mapping both follow.

Check the result on the hand:

```bash
python examples/tactile/visualize_tactile_sensor.py --tactile-grid full_hand_tactile.json --urdf $LEFT_URDF
```

### Step 4: Create the raw mapping (interactive)

Select groups of tactile points and the pixel region each group should fill:

```bash
python examples/tactile/create_tactile_mapping.py \
    --tactile-grid full_hand_tactile.json \
    --output tactile_to_image_mapping.json
```

Controls:
- Left drag: select points/pixels
- Right drag: deselect
- `m`: create a mapping from the current selection
- `u`: undo the last mapping
- `s`: save mappings
- `c`: clear selection
- `q`: quit

### Step 5: Compute the point-to-pixel assignment

Within each group, every pixel is assigned its nearest tactile point in the 2D layout, and
points left without a pixel go to their nearest pixel:

```bash
python examples/tactile/compute_tactile_mapping.py \
    --raw-mapping tactile_to_image_mapping.json \
    --tactile-grid full_hand_tactile.json \
    --output tactile_pixel_mapping.json
```

It first shows the 2D layout (skip with `--no-preview`); edit `LINK_YZ_OFFSETS` in
`tactile_layout.py` to move link clusters.

Check the mapping:

```bash
python examples/tactile/visualize_tactile_mapping_3d.py \
    --tactile-grid full_hand_tactile.json \
    --mapping tactile_pixel_mapping.json \
    --urdf $LEFT_URDF
```

- Click a tactile point to highlight its pixel.
- Click a pixel to highlight its tactile points.
