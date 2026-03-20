# Bench2Drive Evaluation

This document covers everything needed to evaluate our trajectory prediction
models on the [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive)
benchmark and submit to the leaderboard.

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
| Efficiency | Average speed relative to nearby vehicles (20 checkpoints/route) |
| Comfortness | Smoothness of acceleration, jerk, and yaw rate at 20 Hz |

Our agent predicts 50 future waypoints (5 s @ 10 Hz) in ego frame and
converts them to low-level vehicle controls with a trajectory controller.

---

## Repo layout

```
e2e/
├── bench2drive_agent.py      # CARLA leaderboard agent (TEAM_AGENT)
├── agent_configs/
│   ├── template.yaml         # Full config schema with comments
│   ├── mlp.yaml              # MLP baseline (trajectory-only)
│   ├── resnet.yaml           # ResNet front-camera planner
│   └── front_cam.yaml        # TinyViT front-camera planner
├── benchmarking/
│   ├── README.md             # This file
│   ├── dry_run.py            # Unit test — exercises agent without CARLA
│   └── results/              # Evaluation JSON outputs (git-ignored)
├── module.py                 # Lightning training modules
├── dataset.py                # Dataset + coordinate helpers
└── models/                   # Model architectures
```

---

## 1 — Prerequisites

### Python environment

The leaderboard requires **Python 3.8** and **CARLA 0.9.15**.  Create a
dedicated conda environment to avoid conflicts with the training environment:

```bash
conda create -n b2d python=3.8
conda activate b2d

# PyTorch — match your CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Bench2Drive leaderboard
git clone https://github.com/Thinklab-SJTU/Bench2Drive.git
cd Bench2Drive
pip install ninja packaging
pip install -v -e .

# Our extra deps
pip install pyyaml pillow torchvision
```

### CARLA 0.9.15

```bash
mkdir ~/carla && cd ~/carla
wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz
tar -xvf CARLA_0.9.15.tar.gz

# Register the CARLA Python egg for this conda environment
echo "$(pwd)/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg" \
    >> $(conda info --base)/envs/b2d/lib/python3.8/site-packages/carla.pth

# Import extra maps
bash ImportAssets.sh

export CARLA_ROOT=~/carla
```

> **Driver note:** NVIDIA driver 470 is recommended.  Driver 550 has known
> bugs with CARLA's Vulkan renderer.  Check with `nvidia-smi`.

---

## 2 — Linking our code into Bench2Drive

```bash
cd Bench2Drive/leaderboard
mkdir -p team_code

# Symlink agent and configs (keeps Bench2Drive repo clean)
ln -s /workspace/e2e/bench2drive_agent.py  team_code/e2e_agent.py
ln -s /workspace/e2e/agent_configs         team_code/agent_configs
ln -s /workspace/e2e                       team_code/e2e
```

---

## 3 — Choosing and configuring a checkpoint

### Available checkpoints

| Model | Best val/avg_l2 | Checkpoint |
|-------|-----------------|------------|
| MLP (baseline) | 2.639 m | `checkpoints/mlp-epoch=008-val/avg_l2=2.6390.ckpt` |
| Transformer | — | `checkpoints/transformer-epoch=*/avg_l2=*.ckpt` |
| ResNet-50 | — | `checkpoints/resnet-epoch=*/avg_l2=*.ckpt` |
| FrontCam (TinyViT) | — | `checkpoints/front_cam-epoch=*/avg_l2=*.ckpt` |

Copy the relevant config from `agent_configs/` and set `ckpt_path`:

```bash
cp agent_configs/mlp.yaml agent_configs/mlp_run1.yaml
# Edit ckpt_path, and tune control parameters if needed
```

### Config schema (key fields)

```yaml
model:
  type: mlp              # mlp | transformer | front_cam | resnet
  hidden_dim: 256        # [mlp] hidden layer width
  num_layers: 4          # [mlp] depth

ckpt_path: /workspace/e2e/checkpoints/mlp-epoch=008-val/avg_l2=2.6390.ckpt

device: cuda

# Trajectory controller gains — tune these on a debug route first
max_speed:       8.0     # m/s cap
steer_lookahead: 1.0     # seconds ahead for steering target
speed_lookahead: 2.0     # seconds ahead for speed target
k_steer:         1.0     # angle_rad → steer [-1,1]
k_throttle:      0.5     # speed_error → throttle
k_brake:         1.5     # overspeed  → brake
```

---

## 4 — Running evaluation

### Debug mode (1 route, fast feedback)

```bash
export CARLA_ROOT=~/carla
export TEAM_AGENT=$(pwd)/team_code/e2e_agent.py
export TEAM_CONFIG=$(pwd)/team_code/agent_configs/mlp.yaml

# Launch CARLA server on GPU 0
${CARLA_ROOT}/CarlaUE4.sh -graphicsadapter=0 -RenderOffScreen &
sleep 10

# Run one debug route
python leaderboard/leaderboard_evaluator.py \
    --agent=${TEAM_AGENT} \
    --agent-config=${TEAM_CONFIG} \
    --routes=leaderboard/data/bench2drive220.xml \
    --routes-subset=0 \
    --checkpoint=benchmarking/results/debug_run.json \
    --debug=1
```

### Full 220-route evaluation (multi-GPU)

```bash
# Edit GPU_RANK_LIST, TASK_NUM, TEAM_AGENT, TEAM_CONFIG in the script
bash leaderboard/scripts/run_evaluation.sh
```

After all routes finish:

```bash
# Merge per-route JSONs and compute metrics
python leaderboard/scripts/merge_results.py \
    --results-folder benchmarking/results/ \
    --output benchmarking/results/final.json

python leaderboard/scripts/compute_metrics.py \
    --results benchmarking/results/final.json
```

---

## 5 — Coordinate and control conventions

### Coordinate system

The training data and agent share the same conventions:

| Frame | x | y |
|-------|---|---|
| World | East (CARLA +X) | South (CARLA +Y) |
| Ego | Forward along heading | Left |

**Heading:** `theta` in radians, `0 = North`, `π/2 = East` (standard compass).

Derived at inference from the CARLA vehicle transform:
```python
theta = math.radians(carla_yaw_degrees) + math.pi / 2
```

Verified empirically against bench2drive_mini annotations: a vehicle with
`theta ≈ π/2` moves primarily in the `+X` world direction (east), and
`world_to_ego` maps that to `x_ego ≈ 1` (forward). ✓

### Trajectory controller

The predicted trajectory is 50 waypoints at 10 Hz (5 s ahead) in ego frame.
The `TrajectoryController` converts it to vehicle commands:

```
steer     ← atan2(y_left, x_fwd) at waypoint t=steer_lookahead
throttle  ← k_throttle × max(0, desired_speed − current_speed)
brake     ← k_brake    × max(0, current_speed − desired_speed)

desired_speed ← distance_to_wp(t=speed_lookahead) / speed_lookahead
```

**Tuning tips:**
- `max_speed`: reduce to 6 m/s in complex urban scenarios
- `k_steer`: increase if the car understeers, decrease if it oscillates
- `steer_lookahead`: shorter = more reactive; try 0.5–1.5 s

---

## 6 — Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Agent drives straight into walls | theta sign wrong | Check CARLA yaw convention for your map |
| Agent too slow / stops early | `desired_speed` underestimated | Increase `max_speed` or `k_throttle` |
| Agent weaves / oscillates | `k_steer` too high | Reduce to 0.5–0.7 |
| `ImportError: carla` | Not in b2d conda env | `conda activate b2d` |
| CARLA crashes silently | Wrong GPU selected | Use `-graphicsadapter=N`, not `CUDA_VISIBLE_DEVICES` |
| Poor DS despite good val/avg_l2 | Closed-loop compounding errors | Expected for imitation learning baselines |

### Killing stale CARLA processes

```bash
pkill -f CarlaUE4
# or use Bench2Drive's helper:
bash leaderboard/scripts/clean_carla.sh
```

---

## 7 — Adding a new model to the benchmark

1. Train the model with `train.py` and note the best checkpoint.
2. Add a config file to `agent_configs/`:
   ```yaml
   model:
     type: your_model_type   # must match _build_model_from_cfg() in bench2drive_agent.py
     # ... architecture params
   ckpt_path: /workspace/e2e/checkpoints/...
   ```
3. If it's a new architecture, add a branch in `_build_model_from_cfg()`
   inside `bench2drive_agent.py`.
4. Verify with the dry-run test:
   ```bash
   python benchmarking/dry_run.py --config agent_configs/your_config.yaml
   ```
5. Run a 1-route debug evaluation before the full 220-route run.

---

## 8 — Leaderboard submission

The Bench2Drive leaderboard is hosted on the
[NeurIPS 2024 competition page](https://openreview.net/group?id=NeurIPS.cc/2024/Competition/Bench2Drive).
Submit the `final.json` produced by `compute_metrics.py`.

Before submitting, confirm:
- All 220 routes completed (no crashes / timeouts)
- Metrics were computed on the **full** test set, not just a subset
- Checkpoint and config committed to a tagged git release for reproducibility
