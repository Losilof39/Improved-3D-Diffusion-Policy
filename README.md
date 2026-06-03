# [Generalizable Humanoid Manipulation with 3D Diffusion Policies](https://humanoid-manipulation.github.io/)

Our project is **fully open-sourced**. We separate them into two repos: [Learning & Deployment of iDP3](https://github.com/YanjieZe/Improved-3D-Diffusion-Policy) and [Humanoid Teleoperation](https://github.com/YanjieZe/Humanoid-Teleoperation). This repo is for training and deployment of iDP3.


https://github.com/user-attachments/assets/97f6ff8c-45b3-497a-bb66-dd8b24e973b4

**Applications and extensions of iDP3 from the community**:

- [arXiv 2025.03](https://arxiv.org/abs/2503.07511), *PointVLA: Injecting the 3D World into Vision-Language-Action Models*, where iDP3 Encoder is used in training a powerful 3D VLA.

# News
- **2025-06-16** iDP3 is accepted to *IROS 2025*.
- **2024-11-04** Full data and checkpoints (all 3 tasks) are released in [Google Drive](https://drive.google.com/drive/folders/1f5Ln_d14OQ5eSjPDGnD7T4KQpacMhgCB?usp=sharing).
- **2024-10-13** Release the full code for learning/teleoperation. Have a try!


# Training & Deployment of iDP3

This repo is for training and deployment of iDP3. We provide the training data example in this [Google Drive](https://drive.google.com/file/d/1c-rDOe1CcJM8iUuT1ecXKjDYAn-afy2e/view?usp=sharing), so that you could try to train the model without collecting data. The full data and the checkpoints are available in this [Google Drive](https://drive.google.com/drive/folders/1f5Ln_d14OQ5eSjPDGnD7T4KQpacMhgCB?usp=sharing).

More info:
- For the training machine, we use a local computer with an Nvidia RTX 4090 (24G memory). 
- For the deployment machine, we use the cpu of the onboard computer in Fourier GR1.
- We use [RealSense L515](https://www.intelrealsense.com/lidar-camera-l515/) for the depth camera. *Note that RealSense D435 provides very imprecise depth data and is not recommended for training the 3D policy.*



iDP3 is a general 3D visuomotor policy for any robot. You could use iDP3 without camera calibration and point cloud segmentation. Please check our RealSense wrapper for the proposed egocentric 3D visual representation.




## Installation

Install conda env and packages for both learning and deployment machines:

    conda remove -n idp3 --all
    conda create -n idp3 python=3.8
    conda activate idp3
    
    # for cuda >= 12.1
    pip3 install torch==2.1.0 torchvision --index-url https://download.pytorch.org/whl/cu121
    # else, 
    # just install the torch version that matches your cuda version
    
    

    # install my visualizer
    cd third_party
    cd visualizer && pip install -e . && cd ..
    pip install kaleido plotly open3d tyro termcolor h5py
    cd ..


    # install 3d diffusion policy
    pip install --no-cache-dir wandb ipdb gpustat visdom notebook mediapy torch_geometric natsort scikit-video easydict pandas moviepy imageio imageio-ffmpeg termcolor av open3d dm_control dill==0.3.5.1 hydra-core==1.2.0 einops==0.4.1 diffusers==0.11.1 zarr==2.12.0 numba==0.56.4 pygame==2.1.2 shapely==1.8.4 tensorboard==2.10.1 tensorboardx==2.5.1 absl-py==0.13.0 pyparsing==2.4.7 jupyterlab==3.0.14 scikit-image yapf==0.31.0 opencv-python==4.5.3.56 psutil av matplotlib setuptools==59.5.0

    cd Improved-3D-Diffusion-Policy
    pip install -e .
    cd ..

    # install for diffusion policy if you want to use image-based policy
    pip install timm==0.9.7

    # install for r3m if you want to use image-based policy
    cd third_party/r3m
    pip install -e .
    cd ../..


[Install on Deployment Machine] Install realsense package for deploy:

    # first, install realsense driver
    # check this version for RealSenseL515: https://github.com/IntelRealSense/librealsense/releases/tag/v2.54.2

    # also install python api
    pip install pyrealsense2==2.54.2.5684

## Usage

We provide the training data example in [Google Drive](https://drive.google.com/file/d/1c-rDOe1CcJM8iUuT1ecXKjDYAn-afy2e/view?usp=sharing), so that you could try to train the model without collecting data. Download it and unzip it. Then specify the dataset path in `scripts/train_policy.sh`.

For example,  I put the dataset in `/home/ze/projects/Improved-3D-Diffusion-Policy/training_data_example`, and I set `dataset_path=/home/ze/projects/Improved-3D-Diffusion-Policy/training_data_example` in `scripts/train_policy.sh`.

Then you could train the policy and deploy it.

### Unitree G1: Converting xr_teleoperate Data

If you collected demonstrations with [xr_teleoperate](https://github.com/unitreerobotics/xr_teleoperate), use the provided conversion script to turn the raw episodes into the zarr format expected by iDP3.

**Recommended camera setup (RealSense D435):** 640×480 @ 15 Hz for both colour and depth. This is the D435's best-characterised mode and gives sufficient point density for 4096-point subsampling.

**Step 1 — Get your camera intrinsics.** Run this once with the camera at your target resolution:

    python -c "
    import pyrealsense2 as rs
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
    p = rs.pipeline(); p.start(cfg)
    i = p.get_active_profile().get_stream(rs.stream.color) \
          .as_video_stream_profile().get_intrinsics()
    print(f'--fx {i.fx:.4f} --fy {i.fy:.4f} --cx {i.ppx:.4f} --cy {i.ppy:.4f}')
    p.stop()"

Intrinsics are per-unit and resolution-specific — always read them from the device rather than using nominal values.

**Step 2 — Convert episodes to zarr:**

    python scripts/convert_unitree_to_zarr.py \
        --input_dir  /path/to/your/episodes \
        --output_zarr Improved-3D-Diffusion-Policy/data/g1_task_zarr \
        --fx <fx> --fy <fy> --cx <cx> --cy <cy> \
        --z_near 0.2 --z_far 1.2

The `--input_dir` should contain `episode_0000/`, `episode_0001/`, … subdirectories as written by xr_teleoperate's `EpisodeWriter`. Adjust `--z_far` to match your workspace depth (1.2 m works well for tabletop manipulation).

Optional arguments:

| Argument | Default | Description |
|---|---|---|
| `--color_key` | `color_0` | Which colour stream to use for the point cloud |
| `--depth_key` | `depth_0` | Which depth stream to use |
| `--depth_scale` | `1000.0` | Divide raw uint16 depth by this to get metres |
| `--z_near` | `0.1` | Minimum depth (metres) |
| `--z_far` | `1.5` | Maximum depth (metres) |
| `--num_points` | `4096` | Points per cloud |

**Deployment on G1 Jetson:** The repo targets Python 3.8, which matches the Jetson environment on the G1. Install PyTorch using NVIDIA's Jetson-specific wheel (JetPack) rather than the standard pip wheel — the standard `torch==2.1.0` build does not support Jetson CUDA.

**Step 3 — Verify the output:**

    python -c "
    import zarr
    z = zarr.open('Improved-3D-Diffusion-Policy/data/g1_task_zarr', 'r')
    print('state:      ', z['data/state'].shape)       # (T, 26)
    print('action:     ', z['data/action'].shape)      # (T, 26)
    print('point_cloud:', z['data/point_cloud'].shape) # (T, 4096, 6)
    print('episodes:   ', z['meta/episode_ends'][:])
    "

**Train.** The script to train policy:

    # 3d policy (GR1)
    bash scripts/train_policy.sh idp3 gr1_dex-3d 0913_example

    # 3d policy (Unitree G1)
    bash scripts/train_policy.sh idp3 g1_dex-3d g1_experiment

    # 2d policy
    bash scripts/train_policy.sh dp_224x224_r3m gr1_dex-image 0913_example

**Deploy.** After you have trained the policy, deploy the policy with the following command. For missing packages such as `communication.py`, see another [our repo](https://github.com/YanjieZe/Humanoid-Teleoperation/tree/main/humanoid_teleoperation/teleop-zenoh)

    # 3d policy
    bash scripts/deploy_policy.sh idp3 gr1_dex-3d 0913_example

    # 2d policy
    bash scripts/deploy_policy.sh dp_224x224_r3m gr1_dex-image 0913_example

Note that you may not run the deployment code without a robot (different robots have different API). The code we provide is more like an example to show how to deploy the policy. You could modify the code to fit your own robot (any robot with a camera is OK).

**Visualize.** You can visualize our training data example by running (remember to set the dataset path):

    bash scripts/vis_dataset.sh

You can specify `vis_cloud=1` to render the point cloud as in the paper.


## BibTeX

Please consider citing our paper if you find this repo useful:
```
@article{ze2024humanoid_manipulation,
  title   = {Generalizable Humanoid Manipulation with 3D Diffusion Policies},
  author  = {Yanjie Ze and Zixuan Chen and Wenhao Wang and Tianyi Chen and Xialin He and Ying Yuan and Xue Bin Peng and Jiajun Wu},
  year    = {2024},
  journal = {arXiv preprint arXiv:2410.10803}
}
```

## Acknowledgement

We thank the authors of the following repos for their great work: [3D Diffusion Policy](https://github.com/YanjieZe/3D-Diffusion-Policy), [Diffusion Policy](https://github.com/columbia-ai-robotics/diffusion_policy), [VisionProTeleop](https://github.com/Improbable-AI/VisionProTeleop), [Open-TeleVision](https://github.com/OpenTeleVision/TeleVision). 
