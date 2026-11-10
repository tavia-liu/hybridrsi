#!/bin/bash
#SBATCH --job-name=resnet-ppo
#SBATCH --account=engr-lab-chongjie
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=runs/slurm/%x-%j.log
# Symmetric ResNet-18 pixel PPO on the cell (rl/ppo/train_resnet.py). Submit from the repository root:
#   mkdir -p runs/slurm; sbatch --export=ALL,TASK=P4,SEED=0 rl/ppo/train_resnet.sbatch      (ARGS="--flag value ..." for overrides)
# Resumes from $RUN_DIR/checkpoint.pt: resubmit with RUN_DIR=<existing run dir> to continue.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=.:benchmark:rl/ppo MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
TASK=${TASK:-P4}; SEED=${SEED:-0}
RUN_DIR=${RUN_DIR:-runs/ppo/${TASK}_near_s${SEED}_$SLURM_JOB_ID}
export WANDB_PROJECT=${WANDB_PROJECT:-assembly}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
exec "${PYTHON:-.venv/bin/python}" -u rl/ppo/train_resnet.py --task "$TASK" --seed "$SEED" --run-dir "$RUN_DIR" \
    --nthread "$SLURM_CPUS_PER_TASK" --max-hours 23.5 ${ARGS:-}
