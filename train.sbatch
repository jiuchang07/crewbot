#!/bin/bash
#SBATCH --job-name=crew-train
#SBATCH --output=crew_train_%j.out
#SBATCH --error=crew_train_%j.err
#SBATCH --time=04:00:00            # training + eval; short GPU job
#SBATCH --gres=gpu:1              # one idle GPU (any type)
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
# Set your allocation:  #SBATCH --partition=<your-gpu-partition>
# Run this AFTER selfplay.sbatch has produced shards in ~/.cache/crewbot/data.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# --- Environment (pick one) ---
# module load anaconda3 cuda && conda activate crewbot
# uv sync && source .venv/bin/activate

python -c "import torch; assert torch.cuda.is_available(), 'no CUDA GPU visible'"
python train.py --minutes 120 --eval-games 200
echo "training complete -> ~/.cache/crewbot/model.pt"
