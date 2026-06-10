#!/bin/bash
#SBATCH --job-name=idp3_g1_bucket              # Job name
#SBATCH --account=es_coros                     # Account name (don't change)
#SBATCH --gpus=rtx_4090:1                      # 1x RTX 4090
#SBATCH --ntasks=1                             # Single-node training
#SBATCH --cpus-per-task=16                     # DataLoader workers (8) + headroom
#SBATCH --mem-per-cpu=4G                       # 16 x 4G = 64G total
#SBATCH --time=24:00:00                        # adjust down if it finishes sooner
#SBATCH --output=idp3_g1_train_%j.log          # Log file with job ID

echo "$(date) start ${SLURM_JOB_ID}"

# ── Modules ────────────────────────────────────────────────────────────────────
module purge
module load stack/2025-06 gcc/8.5.0

# ── Conda environment ──────────────────────────────────────────────────────────
# Source conda init so `conda activate` works inside the job shell.
# Adjust the path below if your conda lives somewhere other than ~/anaconda3
# (e.g. ~/miniconda3).
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate idp3

# ── Working directory ──────────────────────────────────────────────────────────
# Submit this job from the repo root (Improved-3D-Diffusion-Policy), e.g.:
#   sbatch scripts/train_g1_slurm.sh [addition_info]
cd "$SLURM_SUBMIT_DIR"

# ── Training ───────────────────────────────────────────────────────────────────
# Single-GPU run; train_policy.sh already pins CUDA_VISIBLE_DEVICES=0 and
# launches a plain `python train.py` (no accelerate/multi-GPU needed).
echo "Launching iDP3 training on $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ' ')"

addition_info=${1:-empty_bucket_v2_rgb_aug}

bash scripts/train_policy.sh idp3 g1_dex-3d "${addition_info}"

echo "$(date) done ${SLURM_JOB_ID}"
seff "${SLURM_JOB_ID}"
