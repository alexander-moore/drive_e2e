"""
diagnose_predictions.py — Inspect what the model actually predicts and what
speed the controller derives from those predictions.

Usage:
    python benchmarking/diagnose_predictions.py --config agent_configs/resnet18.yaml
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from bench2drive_agent import TrajectoryController, _build_model_from_cfg, COMMAND_MAP
import yaml


def load_model(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model = _build_model_from_cfg(cfg["model"])
    ckpt = torch.load(cfg["ckpt_path"], map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    # strip "model." prefix added by Lightning
    state = {k[len("model."):] if k.startswith("model.") else k: v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model, cfg


def make_batch(speed_mps, command=3):
    """Simulate a vehicle travelling at constant speed for 41 frames."""
    # past_traj: vehicle moving forward at constant speed in ego frame
    dt = 0.1
    past = np.array([[i * dt * speed_mps, 0.0] for i in range(-40, 1)], dtype=np.float32)
    speed = np.full(41, speed_mps, dtype=np.float32)
    accel = np.zeros((41, 3), dtype=np.float32)
    accel[:, 2] = 9.81  # gravity

    return {
        "past_traj":    torch.from_numpy(past).unsqueeze(0),
        "speed":        torch.from_numpy(speed).unsqueeze(0),
        "acceleration": torch.from_numpy(accel).unsqueeze(0),
        "command":      torch.tensor([command], dtype=torch.long),
        "images":       torch.zeros(1, 1, 3, 224, 224),  # blank image
    }


def analyse(model, cfg, speed_mps, command=3, label=""):
    controller = TrajectoryController(
        max_speed       = cfg.get("max_speed", 8.0),
        steer_lookahead = cfg.get("steer_lookahead", 1.0),
        speed_lookahead = cfg.get("speed_lookahead", 2.0),
        k_steer         = cfg.get("k_steer", 1.0),
        k_throttle      = cfg.get("k_throttle", 0.5),
        k_brake         = cfg.get("k_brake", 1.5),
    )

    batch = make_batch(speed_mps, command)
    with torch.no_grad():
        pred = model(batch)[0].numpy()  # (50, 2)

    thr, steer, brk = controller.control(pred, speed_mps)

    # Key waypoints
    wp2s  = pred[19]  # 2s ahead (speed_idx)
    wp1s  = pred[9]   # 1s ahead (steer_idx)
    dist2 = math.sqrt(wp2s[0]**2 + wp2s[1]**2)
    desired_speed = dist2 / 2.0  # same formula as controller

    print(f"\n{'─'*60}")
    print(f"  {label or f'speed={speed_mps:.1f} m/s, cmd={command}'}")
    print(f"{'─'*60}")
    print(f"  Predicted waypoints (x=fwd, y=left):")
    for t in [0, 4, 9, 19, 29, 49]:
        print(f"    t+{(t+1)*0.1:.1f}s  x={pred[t,0]:+.2f}  y={pred[t,1]:+.2f}  "
              f"dist={math.sqrt(pred[t,0]**2+pred[t,1]**2):.2f} m")
    print(f"  wp@1s (steer):  x={wp1s[0]:+.2f}  y={wp1s[1]:+.2f}")
    print(f"  wp@2s (speed):  x={wp2s[0]:+.2f}  y={wp2s[1]:+.2f}  dist={dist2:.2f} m")
    print(f"  Desired speed:  {desired_speed:.2f} m/s  (current: {speed_mps:.1f} m/s)")
    print(f"  Controller out: throttle={thr:.3f}  steer={steer:.3f}  brake={brk:.3f}")

    return pred, thr, steer, brk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="agent_configs/resnet18.yaml")
    args = parser.parse_args()

    config_path = args.config
    if not Path(config_path).is_absolute():
        config_path = str(Path(__file__).parent.parent / config_path)

    print(f"Loading model from {config_path} ...")
    model, cfg = load_model(config_path)
    print("Done.")

    print("\n" + "="*60)
    print("  PREDICTION DIAGNOSTICS")
    print("="*60)

    # Simulate different driving speeds, all with LANEFOLLOW command
    for speed in [0.0, 1.0, 3.0, 6.0]:
        analyse(model, cfg, speed_mps=speed, command=3,
                label=f"LANEFOLLOW  speed={speed:.1f} m/s")

    # Stationary with different commands
    for cmd_name, cmd_idx in [("LEFT", 0), ("RIGHT", 1), ("STRAIGHT", 2)]:
        analyse(model, cfg, speed_mps=0.0, command=cmd_idx,
                label=f"{cmd_name}  speed=0.0 m/s")


if __name__ == "__main__":
    main()
