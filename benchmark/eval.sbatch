#!/bin/bash
#SBATCH --job-name=nist-eval
#SBATCH --account=engr-lab-chongjie
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=eval-%j.log
# Submit from the repository root:
#   sbatch --export=ALL,TASK=R8,POLICY=examples/noop_policy.py benchmark/eval.sbatch        (N, SEED, HORIZON, OUT optional)
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=.:benchmark MUJOCO_GL=egl
"${PYTHON:-.venv/bin/python}" -u benchmark/eval_protocol.py --task "${TASK:-R8}" --policy "$POLICY" \
    --n "${N:-51}" --seed "${SEED:-0}" --horizon "${HORIZON:-40}" ${OUT:+--out "$OUT"}
