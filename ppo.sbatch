#!/bin/bash
#SBATCH --job-name=crew-ppo
#SBATCH --output=crew_ppo_%j.out
#SBATCH --error=crew_ppo_%j.err
#SBATCH --time=24:00:00            # on-policy PPO trains for the full budget
#SBATCH --gres=gpu:1              # one idle GPU (any type)
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --requeue                 # backfill/idle GPUs get preempted: allow requeue
# Set your allocation:  #SBATCH --partition=<your-gpu-partition>
# If your site has a preemptible/backfill QOS for idle GPUs, set it too, e.g.:
#   #SBATCH --qos=<backfill-or-preemptible-qos>
#
# PRIMARY training path: on-policy PPO self-play. It generates its own data each
# iteration on the GPU (via vec_engine), so there is NO separate data-gen step
# and NO fixed dataset — just run this. (The CPU selfplay.sbatch + offline
# train.sbatch are only for the optional behavior-cloning bootstrap.)
#
# Preemption-safe: ppo.py checkpoints to ~/.cache/crewbot/ppo_state.pt every
# --ckpt-every iters (atomic write) and --resume continues from it. If this job
# is preempted and requeued, it picks up where it left off instead of restarting.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# --- Environment (pick one) ---
# module load anaconda3 cuda && conda activate crewbot
# uv sync && source .venv/bin/activate

python -c "import torch; assert torch.cuda.is_available(), 'no CUDA GPU visible'"
python test_vec_consistency.py        # sanity-check the vectorized engine first
python ppo.py --minutes 1380 --batch 8192 \
    --eval-games 100 --eval-every 50 --ckpt-every 25 --resume
echo "PPO complete -> best checkpoint at ~/.cache/crewbot/ppo_model.pt"
