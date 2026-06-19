#!/bin/bash
#SBATCH --job-name=crew-ppo
#SBATCH --output=crew_ppo_%j.out
#SBATCH --error=crew_ppo_%j.err
#SBATCH --account=gts-aa23         # Phoenix charge account (from pace-quota)
#SBATCH --qos=embers               # free/backfill: runs on idle GPUs, preemptible
# List several GPU partitions + a generic gres so the job lands on whichever GPU
# frees up first. Our model is tiny, so any of these (>=16GB) fits batch 8192,
# and the checkpoint is device-agnostic if chunks land on different GPU types.
#SBATCH --partition=gpu-rtx6000,gpu-l40s,gpu-v100,gpu-h200,gpu-a100
#SBATCH --gres=gpu:1               # one GPU of whatever type the partition has
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00            # embers walltime cap; resume spans multiple chunks
#SBATCH --requeue                  # preemption -> auto-requeue + resume from checkpoint
#
# PRIMARY training path: on-policy PPO self-play. Generates its own data each
# iteration on the GPU (via vec_engine) — NO separate data-gen step, NO dataset.
#
# embers is preemptible with a walltime cap, so to span ~24h of training just
# `sbatch ppo.sbatch` again when a chunk ends — --resume continues from
# ~/.cache/crewbot/ppo_state.pt (atomic checkpoint every --ckpt-every iters).
# Preemption mid-chunk is handled automatically by --requeue + --resume.
#
# For ONE uninterrupted run instead (charges gts-aa23, not preempted), switch:
#   --qos=inferno   and   --time=24:00:00

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate crewbot

python -c "import torch; assert torch.cuda.is_available(), 'no CUDA GPU visible'"
python test_vec_consistency.py        # sanity-check the vectorized engine first
# --minutes 460 leaves a buffer before the 8h cap so the final eval+checkpoint save
python ppo.py --minutes 460 --batch 8192 \
    --eval-games 100 --eval-every 50 --ckpt-every 25 --resume
echo "PPO chunk complete -> best model at ~/.cache/crewbot/ppo_model.pt"
