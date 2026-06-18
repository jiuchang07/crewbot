#!/bin/bash
#SBATCH --job-name=crewbot
#SBATCH --output=crewbot_%j.out
#SBATCH --error=crewbot_%j.err
#SBATCH --time=30:00:00            # 24h self-play + training + eval headroom
#SBATCH --gres=gpu:1               # one idle GPU (any type)
#SBATCH --cpus-per-task=24         # self-play is CPU-bound; grab many cores
#SBATCH --mem=64G
# GaTech PACE: submit with `sbatch slurm_run.sbatch`. Adjust the partition/QOS
# to your allocation, e.g.:  #SBATCH --partition=<your-partition>
#
# This runs the full pipeline on a node with one spare GPU:
#   1. self-play for 24h across all CPU cores -> ~/.cache/crewbot/data
#   2. train the policy/value net on that data (uses the GPU)
#   3. final win_rate evaluation across all 50 missions
#
# Tip: to keep the GPU busy during the 24h of CPU self-play, run self-play and a
# first training pass as separate jobs, or split self-play to a CPU-only job
# and training to a short GPU job. See README "Running on PACE".

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# --- Environment ------------------------------------------------------------
# Option A: cluster modules (preferred on PACE)
# module load anaconda3 cuda
# conda activate crewbot
#
# Option B: uv-managed venv
# uv sync && source .venv/bin/activate
#
# Make sure `python -c "import torch; print(torch.cuda.is_available())"` is True.

NPROC="${SLURM_CPUS_PER_TASK:-$(nproc)}"

# --- 1. Self-play: 24h of play data ----------------------------------------
python selfplay.py --hours 24 --workers "$NPROC"

# --- 2. Train on the collected data ----------------------------------------
python train.py --minutes 120 --eval-games 200

# --- 3. (optional) one round of self-play iteration ------------------------
# python selfplay.py --hours 6 --workers "$NPROC" \
#     --policy-ckpt "$HOME/.cache/crewbot/model.pt" --epsilon 0.2
# python train.py --minutes 120 --eval-games 200

echo "crewbot pipeline complete."
