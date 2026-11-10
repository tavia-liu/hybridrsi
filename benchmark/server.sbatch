#!/bin/bash
#SBATCH --job-name=cell-server
#SBATCH --account=engr-lab-chongjie
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=52:00:00
#SBATCH --output=runs/server/slurm-%j.log
# The robot cell as a service for a sandboxed agent. Submit from the repository root:
#   mkdir -p runs/server; SERVER_TOKEN=<secret> sbatch --export=ALL benchmark/server.sbatch
# Writes runs/server/<jobid>/server.addr (host:port) for the sandbox's ROBOT_ADDR.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=.:benchmark MUJOCO_GL=egl
export PORT=$((8700 + SLURM_JOB_ID % 300))
OUT=runs/server/$SLURM_JOB_ID; mkdir -p $OUT
exec "${PYTHON:-.venv/bin/python}" -u benchmark/server.py --port $PORT --out $OUT
