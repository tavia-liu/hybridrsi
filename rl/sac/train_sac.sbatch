#!/bin/bash
#SBATCH --job-name=resnet-sac
#SBATCH --account=engr-lab-chongjie
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=runs/slurm/%x-%j.log
# SAC on frozen ResNet-18 features (rl/sac/train_sac.py). Submit from the repository root:
#   mkdir -p runs/slurm; sbatch --export=ALL,TASK=P4,SEED=0 rl/sac/train_sac.sbatch      (ARGS="--flag value ..." for overrides)
# Resumes from $RUN_DIR/checkpoint.pt (model + optimisers; the replay buffer restarts empty): RUN_DIR=<existing run dir>.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=.:benchmark:rl/ppo:rl/sac MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
TASK=${TASK:-P4}; SEED=${SEED:-0}; FAMILY=${FAMILY:-near_up}
RUN_DIR=${RUN_DIR:-runs/sac/${TASK}_${FAMILY}_s${SEED}_$SLURM_JOB_ID}
export WANDB_PROJECT=${WANDB_PROJECT:-assembly}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
exec "${PYTHON:-.venv/bin/python}" -u rl/sac/train_sac.py --task "$TASK" --seed "$SEED" --family "$FAMILY" --run-dir "$RUN_DIR" \
    --nthread "$SLURM_CPUS_PER_TASK" --max-hours 23.5 ${ARGS:-}
