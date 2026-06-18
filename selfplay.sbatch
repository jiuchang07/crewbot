#!/bin/bash
#SBATCH --job-name=crew-selfplay
#SBATCH --output=crew_selfplay_%j.out
#SBATCH --error=crew_selfplay_%j.err
#SBATCH --time=25:00:00            # 24h of data + headroom
#SBATCH --cpus-per-task=24         # self-play is CPU-bound: grab many cores
#SBATCH --mem=64G
# NOTE: no --gres=gpu here on purpose. CPU self-play does not use a GPU, so
# don't reserve one for 24h. (See README "Why self-play isn't on the GPU".)
# Set your allocation:  #SBATCH --partition=<your-partition>

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# --- Environment (pick one) ---
# module load anaconda3 cuda && conda activate crewbot
# uv sync && source .venv/bin/activate

NPROC="${SLURM_CPUS_PER_TASK:-$(nproc)}"
python selfplay.py --hours 24 --workers "$NPROC"
echo "self-play complete -> ~/.cache/crewbot/data"
