# Bench2Drive Evaluation

This document covers everything needed to train our trajectory prediction
models and evaluate them on the [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive)
benchmark running inside CARLA 0.9.15.

---

## Overview

Bench2Drive is a closed-loop evaluation benchmark built on CARLA 0.9.15.
It runs 220 routes across 12 towns, 44 scenario types, and 23 weather
conditions.  Each route is ~150 m and tests an isolated driving skill
(cut-in, overtaking, emergency braking, etc.).

**Metrics reported:**

| Metric | Description |
|--------|-------------|
| Driving Score (DS) | Route completion × infraction penalty |
| Success Rate (SR) | Fraction of routes completed without critical failure |
| Efficiency | Average speed relative to nearby vehicles |
| Comfortness | Smoothness of acceleration, jerk, and yaw rate |

Our agent predicts 50 future waypoints (5 s @ 10 Hz) in ego frame and
converts them to low-level vehicle controls with a trajectory controller.

---

## Repo layout

```
drive_e2e/
├── bench2drive_agent.py      # CARLA leaderboard agent (TEAM_AGENT)
├── train.py                  # Training CLI
├── module.py                 # PyTorch Lightning training modules
├── dataset.py                # Dataset + coordinate helpers
├── models/                   # Model architectures
├── agent_configs/
│   ├── template.yaml         # Full config schema with comments
│   ├── mlp.yaml              # MLP baseline (trajectory-only)
│   └── ...
└── benchmarking/
    ├── README.md             # This file
    ├── dry_run.py            # CARLA-free agent smoke test
    └── results/              # Evaluation JSON outputs (git-ignored)
```

**Bench2Drive repo** (separate clone):
```
/home/farm/Documents/Projects/Bench2Drive/
├── carla/                    # CARLA 0.9.15 binary + maps
├── leaderboard/
│   ├── team_code/            # Symlinks to our agent (set up below)
│   │   ├── e2e_agent.py  → drive_e2e/bench2drive_agent.py
│   │   ├── agent_configs → drive_e2e/agent_configs/
│   │   └── drive_e2e     → drive_e2e/
│   └── scripts/
│       ├── run_evaluation.sh              # Core eval runner (configured)
│       ├── run_evaluation_e2e_debug.sh    # 1-route debug run
│       └── run_evaluation_multi_e2e.sh   # Multi-GPU full run
├── scenario_runner/
└── tools/
```

---

## 1 — Environment

Everything runs in the **`b2d`** conda environment (Python 3.8):

```bash
conda activate b2d
```

The `b2d` environment is already set up with:
- Python 3.8, PyTorch 2.4.1+cu124
- CARLA 0.9.15 Python bindings (py3.7 egg via `.pth`)
- Leaderboard deps: `pygame`, `transforms3d`, `py_trees`, `six`, `opencv-python`, etc.

For training, use the **`drive_e2e`** environment (Python 3.11, PyTorch 2.10):

```bash
conda activate drive_e2e
```

---

## 2 — Training

Training uses PyTorch Lightning and logs to TensorBoard.

### MLP baseline (trajectory-only, no camera)

```bash
conda activate drive_e2e
cd /home/farm/Documents/Projects/drive_e2e

python train.py \
    --model mlp \
    --data_root /home/farm/data/bench2drive \
    --val_data_root "" \
    --hidden_dim 256 \
    --num_layers 4 \
    --batch_size 256 \
    --epochs 30 \
    --lr 1e-3 \
    --name mlp
```

The best checkpoint (by `val/avg_l2`) is saved to:
```
checkpoints/mlp-epoch=NNN-val/avg_l2=X.XXXX.ckpt
```

### Other models

```bash
# Transformer (kinematics-only, ~1.4M params)
python train.py --model transformer --batch_size 256 --epochs 30 --name transformer

# ResNet-50 front camera planner (~33M params, slower)
python train.py --model resnet --batch_size 64 --epochs 30 --name resnet

# TinyViT front camera planner (~30M params)
python train.py --model front_cam --batch_size 64 --epochs 30 --name front_cam
```

### Monitor training

```bash
tensorboard --logdir /home/farm/Documents/Projects/drive_e2e/logs/ --port 6006
```

---

## 3 — Verifying the agent (no CARLA required)

Before running the full CARLA benchmark, verify the agent pipeline with
the CARLA-free smoke test:

```bash
conda activate b2d
cd /home/farm/Documents/Projects/drive_e2e

python benchmarking/dry_run.py --config agent_configs/mlp.yaml --ticks 60
```

Expected output: `RESULT: ALL CHECKS PASSED`

**Note:** Update `ckpt_path` in `agent_configs/mlp.yaml` to point to your
trained checkpoint before running this or the CARLA benchmark.

---

## 4 — Running the CARLA benchmark

All evaluation commands run from the **Bench2Drive root**:

```bash
conda activate b2d
cd /home/farm/Documents/Projects/Bench2Drive
```

### 4a — Debug mode (1 route, ~5 min)

Use this to confirm CARLA starts and the agent runs correctly before
committing to the full 220-route evaluation.

```bash
bash leaderboard/scripts/run_evaluation_e2e_debug.sh
```

This launches CARLA on GPU 0, runs 1 route, and writes the result to
`benchmarking/results/debug_run.json`.

**To use a different model or config**, edit the `TEAM_CONFIG` line in
`run_evaluation_e2e_debug.sh`:

```bash
TEAM_CONFIG=leaderboard/team_code/agent_configs/mlp.yaml
```

### 4b — Full 220-route evaluation (multi-GPU)

Edit `leaderboard/scripts/run_evaluation_multi_e2e.sh` to set:
- `TEAM_CONFIG`: path to your agent config (with a trained checkpoint)
- `TASK_NUM`: number of parallel workers (= number of GPUs)
- `GPU_RANK_LIST` and `TASK_LIST`: GPU indices (one per task)

Then run:

```bash
bash leaderboard/scripts/run_evaluation_multi_e2e.sh
```

Each task logs to `e2e_b2d_only_traj/task_N.log`. CARLA can crash on
individual routes — this is normal. The evaluator auto-resumes.

### 4c — Compute metrics

After all 220 routes complete (or fail/timeout):

```bash
cd /home/farm/Documents/Projects/Bench2Drive

# Merge per-route JSON files into one
python tools/merge_route_json.py -f e2e_b2d_only_traj/

# Driving Score and Success Rate
python tools/ability_benchmark.py -r merge.json

# Efficiency and Smoothness
python tools/efficiency_smoothness_benchmark.py -f merge.json -m e2e_b2d_only_traj/
```

> **Important:** `merge_route_json.py` expects exactly 220 route results.
> Failed/crashed routes count as 0 score — do not exclude them.

---

## 5 — Agent config schema

Copy `agent_configs/template.yaml` as a starting point:

```yaml
model:
  type: mlp              # mlp | transformer | front_cam | resnet | vision_transformer
  hidden_dim: 256        # [mlp] hidden width
  num_layers: 4          # [mlp] depth

# Absolute path to a PyTorch Lightning .ckpt file
ckpt_path: /home/farm/Documents/Projects/drive_e2e/checkpoints/mlp-epoch=NNN-val/avg_l2=X.XXXX.ckpt

device: cuda

# Trajectory controller — tune on a 1-route debug run first
max_speed:        6.0    # m/s cap (lower = safer in dense traffic)
steer_lookahead:  1.0    # seconds ahead for steering target
speed_lookahead:  2.0    # seconds ahead for speed target
k_steer:          1.0    # angle → steer gain
k_throttle:       0.5    # speed error → throttle gain
k_brake:          1.5    # overspeed → brake gain
```

---

## 6 — Coordinate and control conventions

### Coordinate frames

| Frame | x | y |
|-------|---|---|
| World (CARLA) | East (+X) | South (+Y) |
| Ego | Forward along heading | Left |

**Heading** `theta`: radians, `0 = North`, `π/2 = East` (standard compass).

Derived from CARLA yaw at inference:
```python
theta = math.radians(carla_yaw_degrees) + math.pi / 2
```

### Trajectory controller

The predicted 50-waypoint trajectory is converted to vehicle commands:
```
steer     ← atan2(y_left, x_fwd)   at waypoint t = steer_lookahead
throttle  ← k_throttle × max(0, desired_speed − current_speed)
brake     ← k_brake    × max(0, current_speed − desired_speed)

desired_speed ← distance_to_wp(t = speed_lookahead) / speed_lookahead
```

**Tuning tips:**
- `max_speed 6.0` works well for MLP/Transformer in dense urban scenarios
- `k_steer 0.7` if the car oscillates; `k_steer 1.2` if it understeers
- `steer_lookahead 0.5–0.8` for tighter corners

---

## 7 — Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Agent drives straight into walls | `theta` sign wrong | Check CARLA yaw convention |
| Car too slow / stops early | Speed underestimated | Increase `max_speed` or `k_throttle` |
| Car weaves / oscillates | `k_steer` too high | Reduce to 0.5–0.7 |
| `ImportError: carla` | Wrong conda env | `conda activate b2d` |
| `ImportError: libtiff.so.5` | Library path | `LD_LIBRARY_PATH` set in `run_evaluation.sh` |
| CARLA crashes silently | Wrong GPU / Vulkan | Use `-graphicsadapter=N`, check `vulkaninfo` |
| `No module named 'agents'` | PYTHONPATH missing PythonAPI | Check `run_evaluation.sh` |
| CARLA shows only 2 log lines and exits | Vulkan issue | `sudo apt install vulkan-tools; vulkaninfo \| head -5` |

### Kill stale CARLA processes

```bash
bash /home/farm/Documents/Projects/Bench2Drive/leaderboard/scripts/../../../tools/clean_carla.sh
# or:
pkill -f CarlaUE4
```

---

## 8 — Adding a new model

1. Train with `train.py --model your_model_type` and note the best checkpoint.
2. Add a config to `agent_configs/`:
   ```yaml
   model:
     type: your_model_type
     # architecture params
   ckpt_path: /path/to/checkpoint.ckpt
   ```
3. Add a branch in `_build_model_from_cfg()` in `bench2drive_agent.py`.
4. Verify with `python benchmarking/dry_run.py --config agent_configs/your_config.yaml`.
5. Run a 1-route debug evaluation before the full 220-route run.

---

## 9 — Key paths (this machine)

| Resource | Path |
|----------|------|
| Training data | `/home/farm/data/bench2drive/` (609 scenarios, ~92K samples) |
| CARLA 0.9.15 | `/home/farm/Documents/Projects/Bench2Drive/carla/` |
| Bench2Drive repo | `/home/farm/Documents/Projects/Bench2Drive/` |
| drive_e2e repo | `/home/farm/Documents/Projects/drive_e2e/` |
| Checkpoints | `/home/farm/Documents/Projects/drive_e2e/checkpoints/` |
| Eval results | `/home/farm/Documents/Projects/Bench2Drive/e2e_b2d_only_traj/` |
| b2d conda env | `/home/farm/miniconda3/envs/b2d/` (Python 3.8, eval) |
| drive_e2e conda env | `/home/farm/miniconda3/envs/drive_e2e/` (Python 3.11, training) |
