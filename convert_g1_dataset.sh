#!/bin/bash

python scripts/convert_unitree_to_zarr.py \
      --input_dir /home/luke/MSMD/repos/unitree/xr_teleoperate/teleop/utils/data/empty_bucket_60ep_cut \
      --output_zarr Improved-3D-Diffusion-Policy/data/g1_empty_bucket_60ep_cut \
      --fx 607.7126 --fy 607.5232 --cx 319.4251 --cy 243.5345 \
      --z_near 0.1 --z_far 1