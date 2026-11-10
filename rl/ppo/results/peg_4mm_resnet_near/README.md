# peg_4mm_resnet_near (imported from assembly-nist)

Reference pixel-PPO run on P4 (peg 4 mm) from near starts, trained in `autodeco/assembly-nist` with
`cap/train_resnet.py` before this repository existed. Slurm jobs 367177 -> 367180 -> 367181 (resumed), run dir
`assembly-nist/runs/peg_4mm_resnet_near`, wandb `WashU-MARL/assemble/peg_4mm_resnet_near_ppo`.
Copied here: `config.json`, `log.jsonl` (one record per iteration), `bank_P4_near_train.npz` (the 640 near starts).
Checkpoints (`checkpoint.pt`, `model_0133*.pt`, `sampler.pt`) and the slurm logs are on engr2 under
`hybridrsi/runs/ppo/peg_4mm_resnet_near_assemblynist/` (not in git, 108 MB).

Setup (identical to `rl/ppo/train_resnet.py` unless noted): 128 envs x 32 steps per iteration, PPO 5 epochs x 4
minibatches, lr 1e-4 adaptive-KL (target 0.01), gamma 0.995, lambda 0.95, clip 0.2, entropy 0.005, uniform reset
sampler over the 640-start bank, action-rate term from iteration 5000, ResNet-18 trunk with lr x0.1, 128x128 x 2
cameras, 150-d proprio history. Differences from this port:

* cameras were assembly-nist's `global` (front-right, fovy 34) + `wrist`; the port renders the benchmark's
  `agentview` + `eye_in_hand` (what `api.camera()` shows, part orange / fixture blue);
* the random-shift augmentation was gated on `model.training`, which is False while collecting, so this run was
  effectively trained without augmentation; the port shifts every collected frame.

Learning curve (`log.jsonl`; success / safety are fractions of the episodes finished in that iteration):

| iteration | transitions | success | safety stop | episode length (steps) | policy std |
|---|---|---|---|---|---|
| 1-50 | 0-0.2 M | 0.00 | 0.00 | 375 (all time out) | 1.00 |
| 100 | 0.4 M | 0.15 | 0.08 | 332 | 0.98 |
| 300 | 1.2 M | 0.00 | 0.65 | 194 | 0.95 |
| 500 | 2.1 M | 0.09 | 0.91 | 34 | 0.94 |
| 1000 | 4.1 M | 0.13 | 0.81 | 94 | 0.92 |
| 2000 | 8.2 M | 0.28 | 0.67 | 72 | 0.90 |
| 5000 | 20.5 M | 0.38 | 0.57 | 65 | 0.86 |
| 8000 | 32.8 M | 0.51 | 0.49 | 15 | 0.71 |
| 10000 | 41.0 M | 0.60 | 0.40 | 16 | 0.66 |
| 13000 | 53.3 M | 0.70 | 0.30 | 13 | 0.61 |

Reading: the first 1.6 M transitions produce no success at all; the policy then learns to move fast enough that
episodes end within ~15 steps, and success + joint-velocity safety stops sum to ~1 from iteration 500 on. The
remaining failures are velocity-limit stops, not time-outs.

Behaviour of the final policy (checkpoint at iteration 13350, deterministic actions, 64 bank starts, measured
2026-11-10): 44/64 success, 19 safety stops, 1 time-out. In 31 of the 44 successes the policy opened the gripper
before success, at a median assembly distance of 4.7 mm (p10 1.6 mm, p90 8.6 mm): it carries the peg to the hole
entrance and drops it in rather than pushing it to the seated pose.
