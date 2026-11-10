#!/bin/bash
#SBATCH --job-name=drqv2-near
#SBATCH --account=engr-lab-chongjie
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=runs/slurm/%x-%j.log
# DrQ-v2 on the cell. Submit from the repository root:
#   mkdir -p runs/slurm; sbatch --export=ALL,TASK=P4,SEED=1 rl/train_drqv2.sbatch        (ARGS="key=value ..." for overrides)
# Resumes from runs/drqv2/<run>/snapshot.pt when RUN_DIR points at an existing run.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=.:benchmark:third_party/drqv2:rl MUJOCO_GL=egl
TASK=${TASK:-P4}; SEED=${SEED:-1}
RUN_DIR=${RUN_DIR:-runs/drqv2/${TASK}_near_s${SEED}_$SLURM_JOB_ID}
BUF=/tmp/hybridrsi_$SLURM_JOB_ID/buffer; mkdir -p "$BUF"
trap 'rm -rf /tmp/hybridrsi_'"$SLURM_JOB_ID" EXIT
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
exec "${PYTHON:-.venv/bin/python}" -u rl/train_drqv2.py task="$TASK" seed="$SEED" buffer_dir="$BUF" run_dir="$RUN_DIR" ${ARGS:-}
