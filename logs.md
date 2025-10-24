# Setup
```
export MESA_D3D12_DEFAULT_ADAPTER_NAME="NVIDIA"

```

# Generate Tactile Grids

```
python examples/tactile/generate_grid_tactile_points.py --hand wuji --grid-spacing 0.004 --output fingers1.json --links finger3_link2,finger3_link3,finger3_link4,finger3_tip_link,finger2_link2,finger2_link3,finger2_link4,finger2_tip_link,finger1_link3,finger1_link4,finger1_tip_link,finger4_link2,finger4_link3,finger4_link4,finger4_tip_link,finger5_link2,finger5_link3,finger5_link4,finger5_tip_link --dense-samples 200000

python examples/tactile/generate_grid_tactile_points.py --grid-spacing 0.004 --output fingers2.json  --dense-samples 200000 --links finger1_link2 --palm-facing-angle 90

python examples/tactile/generate_grid_tactile_points.py --grid-spacing 0.01 --output palm.json  --dense-samples 200000 --links palm_link

python examples/tactile/merge_tactile_grids.py fingers1.json fingers2.json palm.json -o merged_tactile_grid.json
```