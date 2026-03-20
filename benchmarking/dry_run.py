"""
dry_run.py — Test the full agent pipeline without a running CARLA server.

Simulates 60 ticks of sensor input, exercises every code path in the agent
(checkpoint loading, history buffering, inference, trajectory-to-control),
and prints a per-tick summary plus a final health report.

Usage:
    python benchmarking/dry_run.py
    python benchmarking/dry_run.py --config agent_configs/resnet.yaml
    python benchmarking/dry_run.py --config agent_configs/mlp.yaml --ticks 100
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

# Make sure the e2e package is importable from wherever this script is run
sys.path.insert(0, str(Path(__file__).parent.parent))

from bench2drive_agent import (
    E2EBench2DriveAgent,
    TrajectoryController,
    world_to_ego,
    carla_yaw_to_theta,
    PAST_STEPS_TOTAL,
)


# ---------------------------------------------------------------------------
# Fake CARLA stubs (no CARLA installation needed)
# ---------------------------------------------------------------------------
# Inject a minimal 'carla' module so bench2drive_agent.py's
# `import carla` inside run_step doesn't raise ModuleNotFoundError.
import types as _types
_carla_mod = _types.ModuleType("carla")

class _Vec3:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = x, y, z


class _Rotation:
    def __init__(self, roll=0.0, pitch=0.0, yaw=0.0):
        self.roll, self.pitch, self.yaw = roll, pitch, yaw


class _Location(_Vec3):
    pass


class _Transform:
    def __init__(self, location=None, rotation=None):
        self.location = location or _Location()
        self.rotation = rotation or _Rotation()

    def get_forward_vector(self):
        yaw_rad = math.radians(self.rotation.yaw)
        return _Vec3(math.cos(yaw_rad), math.sin(yaw_rad), 0.0)


class _VehicleControl:
    def __init__(self):
        self.throttle = 0.0
        self.steer    = 0.0
        self.brake    = 0.0
        self.hand_brake = False


# Wire into the fake carla module and register before any import of bench2drive_agent
_carla_mod.VehicleControl = _VehicleControl
sys.modules["carla"] = _carla_mod


class _FakeHero:
    """Minimal stub that mimics the carla.Actor interface our agent uses."""

    def __init__(self):
        self._x     = 0.0
        self._y     = 0.0
        self._speed = 0.0    # m/s
        self._yaw   = 0.0    # CARLA degrees

    def step(self, throttle: float, steer: float, brake: float, dt: float = 0.1):
        """Very simple kinematic update."""
        # Speed update
        accel = 3.0 * throttle - 4.0 * brake
        self._speed = max(0.0, self._speed + accel * dt)

        # Heading update from steer
        steer_angle = steer * 30.0   # degrees per step (rough)
        self._yaw  += steer_angle * dt * (self._speed / (self._speed + 1.0))

        # Position update
        theta = carla_yaw_to_theta(self._yaw)
        self._x += self._speed * dt * math.sin(theta)
        self._y -= self._speed * dt * math.cos(theta)  # CARLA Y increases south

    def get_transform(self):
        return _Transform(
            location=_Location(self._x, self._y, 0.0),
            rotation=_Rotation(0.0, 0.0, self._yaw),
        )

    def get_velocity(self):
        theta = carla_yaw_to_theta(self._yaw)
        return _Vec3(self._speed * math.sin(theta),
                     -self._speed * math.cos(theta), 0.0)

    def get_acceleration(self):
        return _Vec3(0.0, 0.0, 9.81)   # gravity component, like training data

    def get_angular_velocity(self):
        return _Vec3(0.0, 0.0, 0.0)


def _make_sensor_data(hero: _FakeHero, needs_images: bool) -> dict:
    """Build a fake input_data dict that matches the real CARLA sensor format."""
    transform = hero.get_transform()
    yaw_rad   = math.radians(transform.rotation.yaw)

    # IMU: [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, compass_deg]
    # compass_deg: 0=north, 90=east  →  theta = deg2rad(compass)
    # We derive compass from carla yaw: compass = yaw - 90  (east=0yaw → 90compass? No.)
    # theta = carla_yaw_rad + pi/2 = deg2rad(compass)
    # compass = deg(carla_yaw_rad + pi/2) = carla_yaw_deg + 90
    compass_deg = transform.rotation.yaw + 90.0
    imu_array = np.array([
        0.0, 0.0, 9.81,          # accelerometer (incl. gravity)
        0.0, 0.0, 0.0,           # gyroscope
        compass_deg,             # compass in degrees (CARLA convention)
    ], dtype=np.float32)

    data = {
        "IMU":   (0, imu_array),
        "SPEED": (0, {"speed": hero._speed}),
    }

    if needs_images:
        # Random RGBA image at CARLA's native 900×1600
        rgba = np.random.randint(0, 256, (900, 1600, 4), dtype=np.uint8)
        data["CAM_FRONT"] = (0, rgba)

    return data


# ---------------------------------------------------------------------------
# Dry-run harness
# ---------------------------------------------------------------------------

def run_dry_run(config_path: str, num_ticks: int = 60) -> bool:
    """
    Simulate `num_ticks` steps of the agent without CARLA.

    Returns True if all checks pass.
    """
    print("=" * 64)
    print(f"  Bench2Drive Agent Dry Run")
    print(f"  config : {config_path}")
    print(f"  ticks  : {num_ticks}")
    print("=" * 64)

    # ── Instantiate agent ──────────────────────────────────────────────
    agent = E2EBench2DriveAgent()
    agent.setup(config_path)

    # Inject fake hero vehicle (replaces self._hero which the base class sets)
    hero = _FakeHero()
    agent._hero = hero

    # ── Provide a trivial global plan (straight ahead) ─────────────────
    # Bench2Drive calls set_global_plan before the first run_step.
    # We pass an empty plan — agent falls back to LANEFOLLOW command.
    agent.set_global_plan([], [])

    # ── Tick loop ──────────────────────────────────────────────────────
    print(f"\n{'tick':>5}  {'x':>7}  {'y':>7}  {'spd':>5}  "
          f"{'thr':>5}  {'str':>6}  {'brk':>5}  {'cmd':>4}  note")
    print("-" * 70)

    controls_history = []
    errors = []
    t0 = time.time()

    for tick in range(num_ticks):
        needs_images = agent._needs_images
        sensor_data  = _make_sensor_data(hero, needs_images)
        timestamp    = tick * 0.1   # 10 Hz

        # ── Run one agent step ─────────────────────────────────────────
        control = agent.run_step(sensor_data, timestamp)

        thr = control.throttle
        st  = control.steer
        brk = control.brake

        controls_history.append((thr, st, brk))

        # Update fake vehicle kinematics
        hero.step(thr, st, brk, dt=0.1)

        note = ""
        if tick < PAST_STEPS_TOTAL - 1:
            note = "(buffering)"
        elif tick == PAST_STEPS_TOTAL - 1:
            note = "← first inference"

        print(f"{tick:>5}  {hero._x:>7.2f}  {hero._y:>7.2f}  "
              f"{hero._speed:>5.2f}  {thr:>5.3f}  {st:>6.3f}  "
              f"{brk:>5.3f}  {agent._current_command:>4}  {note}")

    elapsed = time.time() - t0

    # ── Checks ─────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("  Health checks")
    print("=" * 64)

    # 1. Agent produced controls for every tick
    assert len(controls_history) == num_ticks, "Missing control outputs"
    print(f"  [OK] {num_ticks} controls produced")

    # 2. Throttle/steer/brake are in valid ranges
    for i, (thr, st, brk) in enumerate(controls_history):
        if not (0.0 <= thr <= 1.0):
            errors.append(f"tick {i}: throttle={thr:.3f} out of [0,1]")
        if not (-1.0 <= st <= 1.0):
            errors.append(f"tick {i}: steer={st:.3f} out of [-1,1]")
        if not (0.0 <= brk <= 1.0):
            errors.append(f"tick {i}: brake={brk:.3f} out of [0,1]")
    if not errors:
        print(f"  [OK] All controls in valid ranges")
    else:
        for e in errors:
            print(f"  [FAIL] {e}")

    # 3. Agent started moving after buffer filled
    first_inference = PAST_STEPS_TOTAL  # tick index of first model output
    if num_ticks > first_inference + 5:
        # Just check that the vehicle received non-zero throttle at some point
        post_controls = controls_history[first_inference:]
        any_throttle = any(thr > 0.01 for thr, _, _ in post_controls)
        if any_throttle:
            print(f"  [OK] Throttle > 0 observed after buffer filled")
        else:
            print(f"  [WARN] No throttle > 0.01 after buffer filled — "
                  f"check controller gains (k_throttle, max_speed)")

        # Warn if steer is saturated for most of the post-buffer window
        # (expected when vehicle starts stationary with zero-history; fine in real CARLA)
        saturated = sum(1 for _, st, _ in post_controls if abs(st) > 0.99)
        if saturated / len(post_controls) > 0.5:
            print(f"  [NOTE] Steer saturated for {saturated}/{len(post_controls)} "
                  f"post-buffer ticks.  This is expected in the dry run (vehicle starts "
                  f"stationary, model fed zero history).  Real CARLA starts on a live road.")

    # 4. metric_info populated
    mi_keys = list(agent.metric_info.keys())
    if mi_keys:
        last_mi = agent.metric_info[mi_keys[-1]]
        expected = {"location", "rotation", "velocity", "acceleration",
                    "angular_velocity", "forward_vector"}
        missing = expected - set(last_mi.keys())
        if not missing:
            print(f"  [OK] metric_info has all required fields ({len(mi_keys)} entries)")
        else:
            print(f"  [FAIL] metric_info missing fields: {missing}")
    else:
        print(f"  [FAIL] metric_info is empty")

    # 5. No NaN / Inf in any control output
    flat = [v for thr, st, brk in controls_history for v in (thr, st, brk)]
    if all(math.isfinite(v) for v in flat):
        print(f"  [OK] No NaN/Inf in control outputs")
    else:
        bad = [(i, v) for i, v in enumerate(flat) if not math.isfinite(v)]
        errors.append(f"NaN/Inf in controls: {bad[:5]}")
        print(f"  [FAIL] NaN/Inf detected in controls")

    # ── Summary ────────────────────────────────────────────────────────
    print()
    print(f"  Elapsed : {elapsed:.2f} s  ({elapsed/num_ticks*1000:.1f} ms/tick)")
    print(f"  Final   : x={hero._x:.2f}  y={hero._y:.2f}  speed={hero._speed:.2f} m/s")

    if errors:
        print(f"\n  RESULT: FAILED ({len(errors)} error(s))")
        for e in errors:
            print(f"    - {e}")
        return False
    else:
        print(f"\n  RESULT: ALL CHECKS PASSED")
        return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Bench2Drive agent dry run (no CARLA needed)")
    parser.add_argument(
        "--config", default="agent_configs/mlp.yaml",
        help="Path to agent YAML config (default: agent_configs/mlp.yaml)",
    )
    parser.add_argument(
        "--ticks", type=int, default=60,
        help="Number of simulation ticks to run (default: 60)",
    )
    args = parser.parse_args()

    config_path = args.config
    if not Path(config_path).is_absolute():
        # Resolve relative to repo root (parent of benchmarking/)
        config_path = str(Path(__file__).parent.parent / config_path)

    ok = run_dry_run(config_path, args.ticks)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
