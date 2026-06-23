#!/usr/bin/env bash
# Self-healing launcher for the "push higher" run: bigger net (768x6) on solvable
# missions. Writes to ppo_big_* so it never disturbs the banked playable model.
set -u
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=0

TOTAL_HOURS="${1:-23}"
END=$(( $(date +%s) + $(python -c "print(int($TOTAL_HOURS*3600))") ))
LOG=ppo_big.log
OUT="$HOME/.cache/crewbot/ppo_big_model.pt"
STATE="$HOME/.cache/crewbot/ppo_big_state.pt"

while :; do
    NOW=$(date +%s); REM_MIN=$(( (END - NOW) / 60 ))
    if [ "$REM_MIN" -lt 1 ]; then
        echo "[wrapper] global deadline reached; stopping." >> "$LOG"; break
    fi
    echo "[wrapper] launching big-net ppo.py --resume with ${REM_MIN} min remaining" >> "$LOG"
    python ppo.py --minutes "$REM_MIN" --batch 4096 --hidden 768 --n-blocks 6 \
        --eval-games 100 --eval-every 50 --ckpt-every 25 --resume --solvable \
        --out "$OUT" --state "$STATE" >> "$LOG" 2>&1
    if grep -q "^win_rate:" "$LOG"; then
        echo "[wrapper] trainer completed its budget; done." >> "$LOG"; break
    fi
    echo "[wrapper] trainer exited early; resuming in 5s..." >> "$LOG"
    sleep 5
done
