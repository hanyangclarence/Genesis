```
python examples/tactile/generate_grid_tactile_points.py --grid-spacing-h 0.005 --grid-spacing-v 0.003 --output fingertip.json --dense-samples 200000 --links finger2_tip_link,finger3_tip_link,finger4_tip_link,finger5_tip_link --palm-threshold -0.3

python examples/tactile/generate_grid_tactile_points.py --grid-spacing-h 0.003 --grid-spacing-v 0.0046 --output finger_link2_1.json --dense-samples 200000 --links finger2_link2,finger3_link2 --palm-threshold -0.3 --z-min 0.003 --z-max 0.035
python examples/tactile/generate_grid_tactile_points.py --grid-spacing-h 0.003 --grid-spacing-v 0.0046 --output finger_link2_2.json --dense-samples 200000 --links finger4_link2 --palm-threshold -0.3 --z-min 0.005 --z-max 0.035
python examples/tactile/generate_grid_tactile_points.py --grid-spacing-h 0.003 --grid-spacing-v 0.0046 --output finger_link2_3.json --dense-samples 200000 --links finger5_link2 --palm-threshold -0.3 --z-min 0.008 --z-max 0.035

python examples/tactile/generate_grid_tactile_points.py --grid-spacing-h 0.003 --grid-spacing-v 0.0046 --output finger_link3.json --dense-samples 200000 --links finger2_link3,finger3_link3,finger4_link3,finger5_link3 --palm-threshold -0.3 --z-min 0.003

# after change code
python examples/tactile/generate_grid_tactile_points.py --grid-spacing-h 0.003 --grid-spacing-v 0.005 --output finger1_tip.json --dense-samples 200000 --links finger1_tip_link --palm-threshold -0.3
python examples/tactile/generate_grid_tactile_points.py --grid-spacing-h 0.003 --grid-spacing-v 0.005 --output finger1_link4.json --dense-samples 200000 --links finger1_link4 --palm-threshold -0.3 --z-max 0.016 --z-min 0.001
python examples/tactile/generate_grid_tactile_points.py --grid-spacing-h 0.003 --grid-spacing-v 0.005 --output finger1_link3.json --dense-samples 200000 --links finger1_link3 --palm-threshold -0.3 --z-min 0.002 --z-max 0.026
python examples/tactile/generate_grid_tactile_points.py --grid-spacing-h 0.003 --grid-spacing-v 0.005 --output finger1_link2.json --dense-samples 200000 --links finger1_link2 --palm-threshold -0.3 --z-min 0.002 --z-max 0.032 --palm-facing-angle 70

python examples/tactile/generate_grid_tactile_points.py --grid-spacing-h 0.005 --grid-spacing-v 0.005 --output palm.json --dense-samples 200000 --links palm_link  --palm-threshold -0.8
```

## Tactile Point to Pixel Mapping Pipeline

### Step 1: Generate tactile points for each link
Use the commands above to generate grid tactile points for each finger link and palm.

### Step 2: Merge tactile grids
Merge all individual JSON files into one:
```bash
python examples/tactile/merge_tactile_grids.py \
    finger1_*.json fingertip.json finger_link*.json palm.json \
    --output full_hand_tactile.json
```

### Step 3: Create raw mapping (interactive)
Use the interactive tool to select tactile points and corresponding pixel regions:
```bash
python examples/tactile/create_tactile_mapping.py \
    --tactile-grid full_hand_tactile.json \
    --output tactile_to_image_mapping.json
```
Controls:
- Left drag: Select points/pixels
- Right drag: Deselect
- 'm': Create mapping from selection
- 's': Save mappings
- 'c': Clear selection

### Step 4: Compute optimal assignment
Convert raw mappings to point-to-pixel assignments:
```bash
python examples/tactile/compute_tactile_mapping.py \
    --raw-mapping tactile_to_image_mapping.json \
    --tactile-grid full_hand_tactile.json \
    --output tactile_pixel_mapping.json
```

### Step 5: Visualize and verify
Interactive visualization to check mapping correctness:
```bash
python examples/tactile/visualize_tactile_mapping.py \
    --mapping tactile_pixel_mapping.json \
    --tactile-grid full_hand_tactile.json
```
- Click tactile point → highlights corresponding pixel
- Click pixel → highlights corresponding tactile point(s)

### Coordinate mapping
- Row direction (image) ↔ Z direction (tactile, positive = up)
- Column direction (image) ↔ Y direction (tactile, flipped: positive Y = lower col)