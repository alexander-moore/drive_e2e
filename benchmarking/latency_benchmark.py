"""
latency_benchmark.py — Measure per-tick inference latency across optimization configs.

Runs the agent in dry-run mode (no CARLA) and times only the model forward pass
ticks (after the history buffer is full), reporting mean ± std ms/tick.

Usage:
    python benchmarking/latency_benchmark.py
    python benchmarking/latency_benchmark.py --config agent_configs/resnet18.yaml
    python benchmarking/latency_benchmark.py --config agent_configs/mlp.yaml --ticks 200
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Fake CARLA stubs (mirrors dry_run.py) ────────────────────────────────────
import types as _types
_carla_mod = _types.ModuleType("carla")

class _Vec3:
    def __init__(self, x=0.0, y=0.0, z=0.0): self.x, self.y, self.z = x, y, z
class _Rotation:
    def __init__(self, roll=0.0, pitch=0.0, yaw=0.0): self.roll, self.pitch, self.yaw = roll, pitch, yaw
class _Location(_Vec3): pass
class _Transform:
    def __init__(self, location=None, rotation=None):
        self.location = location or _Location()
        self.rotation = rotation or _Rotation()
    def get_forward_vector(self):
        yr = math.radians(self.rotation.yaw)
        return _Vec3(math.cos(yr), math.sin(yr), 0.0)
class _VehicleControl:
    def __init__(self): self.throttle = self.steer = self.brake = 0.0; self.hand_brake = False

_carla_mod.VehicleControl = _VehicleControl
sys.modules["carla"] = _carla_mod

from bench2drive_agent import (
    E2EBench2DriveAgent, carla_yaw_to_theta, PAST_STEPS_TOTAL
)


class _FakeHero:
    def __init__(self): self._x = self._y = self._speed = self._yaw = 0.0
    def step(self, thr, st, brk, dt=0.1):
        self._speed = max(0.0, self._speed + (3.0*thr - 4.0*brk)*dt)
        self._yaw  += st * 30.0 * dt * (self._speed / (self._speed + 1.0))
        theta = carla_yaw_to_theta(self._yaw)
        self._x += self._speed * dt * math.sin(theta)
        self._y -= self._speed * dt * math.cos(theta)
    def get_transform(self):
        return _Transform(_Location(self._x, self._y, 0.0), _Rotation(0., 0., self._yaw))
    def get_velocity(self):
        t = carla_yaw_to_theta(self._yaw)
        return _Vec3(self._speed*math.sin(t), -self._speed*math.cos(t), 0.0)
    def get_acceleration(self): return _Vec3(0., 0., 9.81)
    def get_angular_velocity(self): return _Vec3()


def _make_sensor_data(hero: _FakeHero, needs_images: bool) -> dict:
    compass_deg = hero._yaw + 90.0
    imu = np.array([0., 0., 9.81, 0., 0., 0., compass_deg], dtype=np.float32)
    data = {"IMU": (0, imu), "SPEED": (0, {"speed": hero._speed})}
    if needs_images:
        data["CAM_FRONT"] = (0, np.random.randint(0, 256, (900, 1600, 4), dtype=np.uint8))
    return data


# ── Benchmark configurations to sweep ────────────────────────────────────────

CONFIGS = [
    # label                     inference_mode   quantization
    ("pytorch  / float32",      "pytorch",       "none"),
    ("compile  / float32",      "compile",       "none"),
    ("compile  / bf16  ★",      "compile",       "bf16"),
    ("compile  / fp16",         "compile",       "fp16"),
    ("torchscript / float32",   "torchscript",   "none"),
    ("torchscript / bf16",      "torchscript",   "bf16"),
]


def _run_config(
    base_cfg: dict,
    inference_mode: str,
    quantization: str,
    num_ticks: int,
) -> Tuple[float, float]:
    """
    Run one configuration and return (mean_ms, std_ms) over inference ticks only.
    Returns (nan, nan) if the run fails.
    """
    cfg = copy.deepcopy(base_cfg)
    cfg["inference_mode"] = inference_mode
    cfg["quantization"]   = quantization

    # Write a temp YAML so the agent can load it normally
    tmp_path = Path(__file__).parent / "_tmp_bench_cfg.yaml"
    with open(tmp_path, "w") as f:
        yaml.dump(cfg, f)

    try:
        agent = E2EBench2DriveAgent()
        agent.setup(str(tmp_path))
        agent._hero = _FakeHero()
        agent.set_global_plan([], [])

        tick_times_ms: List[float] = []

        for tick in range(num_ticks):
            sensor_data = _make_sensor_data(agent._hero, agent._needs_images)
            t0 = time.perf_counter()
            control = agent.run_step(sensor_data, tick * 0.1)
            t1 = time.perf_counter()

            agent._hero.step(control.throttle, control.steer, control.brake)

            # Only record post-buffer inference ticks
            if tick >= PAST_STEPS_TOTAL:
                tick_times_ms.append((t1 - t0) * 1000.0)

        if not tick_times_ms:
            return float("nan"), float("nan")

        return float(np.mean(tick_times_ms)), float(np.std(tick_times_ms))

    except Exception as exc:
        print(f"    ERROR: {exc}")
        return float("nan"), float("nan")
    finally:
        tmp_path.unlink(missing_ok=True)


def run_benchmark(config_path: str, num_ticks: int = 200) -> None:
    with open(config_path) as f:
        base_cfg = yaml.safe_load(f)

    model_type = base_cfg.get("model", {}).get("type", "unknown")
    device     = base_cfg.get("device", "cuda")

    print()
    print("=" * 68)
    print(f"  Inference Latency Benchmark")
    print(f"  model  : {model_type}")
    print(f"  config : {config_path}")
    print(f"  device : {device}")
    print(f"  ticks  : {num_ticks}  (first {PAST_STEPS_TOTAL} excluded — buffer fill)")
    print("=" * 68)

    results: List[Tuple[str, float, float]] = []

    for label, inf_mode, quant in CONFIGS:
        print(f"\n  [{label}]")
        mean_ms, std_ms = _run_config(base_cfg, inf_mode, quant, num_ticks)
        results.append((label, mean_ms, std_ms))
        if not math.isnan(mean_ms):
            print(f"    → {mean_ms:.2f} ± {std_ms:.2f} ms/tick")

    # ── Results table ─────────────────────────────────────────────────────
    baseline_ms = next(
        (m for l, m, _ in results if "pytorch" in l and not math.isnan(m)), None
    )

    print()
    print("=" * 68)
    print(f"  Results — {model_type} on {device}")
    print("=" * 68)
    header = f"  {'Configuration':<28}  {'Mean (ms)':>10}  {'Std (ms)':>9}  {'Speedup':>8}"
    print(header)
    print("  " + "-" * 64)
    for label, mean_ms, std_ms in results:
        if math.isnan(mean_ms):
            row = f"  {label:<28}  {'FAILED':>10}  {'':>9}  {'':>8}"
        else:
            speedup = f"{baseline_ms/mean_ms:.2f}×" if baseline_ms else "—"
            row = f"  {label:<28}  {mean_ms:>10.2f}  {std_ms:>9.2f}  {speedup:>8}"
        print(row)
    print()

    # ── Markdown table for copy-paste into README ─────────────────────────
    print("  Markdown table (copy into README):")
    print()
    print(f"  | Configuration | Mean ms/tick | Std | Speedup vs pytorch/f32 |")
    print(f"  |---|---|---|---|")
    for label, mean_ms, std_ms in results:
        label_md = label.replace("★", "\\★").strip()
        if math.isnan(mean_ms):
            print(f"  | {label_md} | FAILED | — | — |")
        else:
            speedup = f"{baseline_ms/mean_ms:.2f}×" if baseline_ms else "—"
            print(f"  | {label_md} | {mean_ms:.2f} | ±{std_ms:.2f} | {speedup} |")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="agent_configs/resnet18.yaml",
        help="Base agent YAML config to benchmark (default: agent_configs/resnet18.yaml)",
    )
    parser.add_argument(
        "--ticks", type=int, default=200,
        help="Total ticks to run per config (default: 200; first 41 excluded)",
    )
    args = parser.parse_args()

    config_path = args.config
    if not Path(config_path).is_absolute():
        config_path = str(Path(__file__).parent.parent / config_path)

    run_benchmark(config_path, args.ticks)


if __name__ == "__main__":
    main()
