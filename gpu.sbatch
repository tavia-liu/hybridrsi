#!/bin/bash
#SBATCH --job-name=nist-gpu
#SBATCH --account=engr-lab-chongjie
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.log
# GPU job template. Submit from the repository root with the command to run in $CMD:
#
#   sbatch --export=ALL,CMD="benchmark/eval_protocol.py --task R8 --policy examples/noop_policy.py --n 10 --seed 0" gpu.sbatch
#   sbatch --export=ALL,CMD="benchmark/multi_board.py --tasks board5 --policy my_policy.py --n-seeds 2" gpu.sbatch
#
# Rendering (cameras, videos) needs a GPU with EGL; never run these on the login node. Interactive alternative:
#   srun -p general-gpu -A engr-lab-chongjie --gres=gpu:a40:1 --cpus-per-task=4 --mem=24G --time=2:00:00 --pty bash
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=.:benchmark MUJOCO_GL=egl
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"${PYTHON:-.venv/bin/python}" -u $CMD
