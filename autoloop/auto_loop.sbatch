#!/bin/bash
#SBATCH --job-name=hyb-autoloop
#SBATCH --account=engr-lab-chongjie
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=runs/slurm/%x-%j.log
# Code-as-policy loop on the clean cell. Submit from the repository root:
#   mkdir -p runs/slurm; sbatch --export=ALL,TASK=P4 autoloop/auto_loop.sbatch        (RESUME=1 continues an existing run)
# Claude auth: a long-lived OAuth token in $CLAUDE_TOKEN_FILE and a config dir of its own (no account rotation).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=.:benchmark:autoloop MUJOCO_GL=egl CAP_CLAUDE_BIN="${CAP_CLAUDE_BIN:-$HOME/.local/bin/claude}"
export CLAUDE_CONFIG_DIR="${CLAUDE_CONFIG_DIR_LOOP:-/engrfs/project/chongjie/buzhao/claude/config-hybridrsi}"
export CLAUDE_CODE_OAUTH_TOKEN="$(cat "${CLAUDE_TOKEN_FILE:-/engrfs/project/chongjie/buzhao/claude/tokens/buzhao.token}")"
export CLAUDE_CODE_DISABLE_CLAUDE_MDS=1
mkdir -p "$CLAUDE_CONFIG_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
exec "${PYTHON:-.venv/bin/python}" -u autoloop/loop.py --task "${TASK:-P4}" --schedule "${SCHEDULE:-2,5,10,15,20}" \
    --rounds "${ROUNDS:-25}" --steps "${STEPS:-6000}" \
    --model "${MODEL:-claude-opus-5-5[1m]}" --audit-model "${AUDIT_MODEL:-claude-opus-5-5[1m]}" \
    --max-experiments "${MAX_EXPERIMENTS:-3}" ${OUT:+--out "$OUT"} ${RESUME:+--resume} ${SMOKE:+--smoke}
