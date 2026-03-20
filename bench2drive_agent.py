"""
bench2drive_agent.py — Bench2Drive / CARLA leaderboard agent for our E2E trajectory models.

USAGE
-----
Set in your evaluation script (e.g. run_evaluation.sh):
    TEAM_AGENT  = /workspace/e2e/bench2drive_agent.py
    TEAM_CONFIG = /workspace/e2e/agent_configs/resnet.yaml

See agent_configs/template.yaml for the full config schema.

MODEL LOADING
-------------
Loads the inner torch model from a PyTorch Lightning checkpoint.  The Lightning
wrapper (E2EDrivingModule / MultiTaskE2EModule) is NOT required at inference;
only the underlying nn.Module weights are extracted from the checkpoint.

COORDINATE CONVENTION
---------------------
Training data stores positions in CARLA world coordinates (x ≈ east, y ≈ south).
theta (heading) is stored in radians with the standard compass convention:
    0     = north  (vehicle faces world -Y)
    π/2   = east   (vehicle faces world +X)
At inference we derive theta from CARLA's rotation.yaw:
    theta = math.radians(carla_yaw_deg) + math.pi / 2
world_to_ego applies the same rotation used at training time (from dataset.py):
    x_ego = dx * sin(theta) - dy * cos(theta)   # forward
    y_ego = dx * cos(theta) + dy * sin(theta)   # left
"""

from __future__ import annotations

import math
import os
import sys
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# carla is only importable inside the CARLA/leaderboard environment.
# We import it lazily inside methods that need it so the file can be imported
# and tested outside of CARLA without error.

# ---------------------------------------------------------------------------
# Constants — must match dataset.py
# ---------------------------------------------------------------------------

PAST_STEPS_TOTAL = 41    # 40 history frames + 1 anchor frame (current)
FUTURE_STEPS     = 50    # 5 s @ 10 Hz
IMAGE_H, IMAGE_W = 224, 224

# CARLA RoadOption → our 0-indexed command (same mapping as dataset.py)
COMMAND_MAP = {2: 0, 3: 1, 4: 2, 5: 3}   # LEFT, RIGHT, STRAIGHT, LANEFOLLOW

# Sensor IDs (must match sensors() list below)
CAM_FRONT_ID = "CAM_FRONT"
IMU_ID       = "IMU"
SPEED_ID     = "SPEED"


# ---------------------------------------------------------------------------
# Coordinate helpers (mirrors dataset.py exactly)
# ---------------------------------------------------------------------------

def world_to_ego(points_world: np.ndarray, x0: float, y0: float,
                 theta: float) -> np.ndarray:
    """Transform (N, 2) world-frame points to ego frame.

    theta convention:
        0 = north (vehicle faces world -Y), π/2 = east (vehicle faces world +X).
    Ego frame:
        col-0 = x = forward along heading
        col-1 = y = left
    """
    dx = points_world[:, 0] - x0
    dy = points_world[:, 1] - y0
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    x_ego =  dx * sin_t - dy * cos_t   # forward
    y_ego =  dx * cos_t + dy * sin_t   # left
    return np.stack([x_ego, y_ego], axis=1)


def carla_yaw_to_theta(yaw_deg: float) -> float:
    """Convert CARLA rotation.yaw (degrees, 0=east, CW positive) to theta."""
    return math.radians(yaw_deg) + math.pi / 2


# ---------------------------------------------------------------------------
# Trajectory-to-control converter
# ---------------------------------------------------------------------------

class TrajectoryController:
    """
    Converts a predicted ego-frame trajectory to (throttle, steer, brake).

    The trajectory is (T, 2) at 10 Hz in ego frame (x=forward, y=left).

    Steering: signed angle to the waypoint at `steer_lookahead` seconds ahead,
              clipped to [-1, 1] and scaled by k_steer.

    Speed:    desired speed estimated from trajectory displacement over 2 s
              ahead, capped by max_speed and clipped by a minimum.
              PID-style: error drives throttle; braking if overspeed.
    """

    def __init__(
        self,
        max_speed: float = 8.0,     # m/s — absolute speed cap
        steer_lookahead: float = 1.0,   # seconds ahead for steering target
        speed_lookahead: float = 2.0,   # seconds ahead for speed target
        k_steer: float = 1.0,       # gain on steer angle (rad → normalised)
        k_throttle: float = 0.5,    # throttle gain (speed error → throttle)
        k_brake: float = 1.5,       # brake gain (overspeed → brake)
        max_throttle: float = 0.75,
        max_brake: float = 1.0,
    ):
        self.max_speed = max_speed
        self.steer_idx = max(0, round(steer_lookahead * 10) - 1)  # 0-based
        self.speed_idx = max(0, round(speed_lookahead * 10) - 1)
        self.k_steer   = k_steer
        self.k_throttle = k_throttle
        self.k_brake    = k_brake
        self.max_throttle = max_throttle
        self.max_brake    = max_brake

    def control(self, traj: np.ndarray, current_speed: float
                ) -> Tuple[float, float, float]:
        """
        Args:
            traj:          (T, 2) ego-frame waypoints (x=fwd, y=left)
            current_speed: current scalar speed in m/s

        Returns:
            (throttle, steer, brake) all floats
        """
        T = traj.shape[0]

        # ── Steer ─────────────────────────────────────────────────────────
        wp_steer = traj[min(self.steer_idx, T - 1)]        # (2,)
        angle    = math.atan2(float(wp_steer[1]),           # y = left
                              float(wp_steer[0]) + 1e-6)   # x = forward
        steer = float(np.clip(self.k_steer * angle, -1.0, 1.0))

        # ── Speed target from trajectory displacement ─────────────────────
        # Estimate desired speed as: (distance from origin to wp@speed_idx) / time
        wp_speed = traj[min(self.speed_idx, T - 1)]
        dist = math.sqrt(float(wp_speed[0]) ** 2 + float(wp_speed[1]) ** 2)
        time_horizon = (self.speed_idx + 1) / 10.0         # seconds
        desired_speed = float(np.clip(dist / (time_horizon + 1e-6),
                                      0.0, self.max_speed))

        # ── Throttle / brake ──────────────────────────────────────────────
        speed_error = desired_speed - current_speed
        if speed_error >= 0:
            throttle = float(np.clip(self.k_throttle * speed_error,
                                     0.0, self.max_throttle))
            brake = 0.0
        else:
            throttle = 0.0
            brake    = float(np.clip(-self.k_brake * speed_error,
                                     0.0, self.max_brake))

        return throttle, steer, brake


# ---------------------------------------------------------------------------
# Model builder  (mirrors train.py — add new model types here)
# ---------------------------------------------------------------------------

def _build_model_from_cfg(cfg: dict) -> nn.Module:
    """Instantiate the torch model from a config dict (no checkpoint loaded)."""
    # Add the e2e repo to path so model imports resolve regardless of CWD.
    repo_root = str(Path(__file__).parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    model_type   = cfg["type"]
    future_steps = cfg.get("future_steps", FUTURE_STEPS)

    if model_type == "mlp":
        from models.mlp_planner import MLPPlanner
        return MLPPlanner(
            past_steps=PAST_STEPS_TOTAL,
            future_steps=future_steps,
            hidden_dim=cfg.get("hidden_dim", 256),
            num_layers=cfg.get("num_layers", 4),
            dropout=cfg.get("dropout", 0.0),
        )
    elif model_type == "transformer":
        from models.transformer_planner import TransformerPlanner
        return TransformerPlanner(
            past_steps=PAST_STEPS_TOTAL,
            future_steps=future_steps,
            d_model=cfg.get("d_model", 128),
            nhead=cfg.get("nhead", 4),
            enc_layers=cfg.get("enc_layers", 3),
            dec_layers=cfg.get("dec_layers", 3),
            dim_feedforward=cfg.get("dim_feedforward", 512),
            dropout=cfg.get("dropout", 0.0),
        )
    elif model_type == "front_cam":
        from models.front_cam_planner import FrontCamPlanner
        return FrontCamPlanner(
            token_dim=cfg.get("token_dim", 256),
            num_heads=cfg.get("nhead", 4),
            enc_layers=cfg.get("enc_layers", 2),
            dec_layers=cfg.get("dec_layers", 3),
            multiscale=cfg.get("multiscale", False),
        )
    elif model_type == "front_cam_depth":
        from models.front_cam_depth_planner import FrontCamDepthPlanner
        return FrontCamDepthPlanner(
            token_dim=cfg.get("token_dim", 256),
            num_heads=cfg.get("nhead", 4),
            enc_layers=cfg.get("enc_layers", 2),
            dec_layers=cfg.get("dec_layers", 3),
            multiscale=cfg.get("multiscale", False),
        )
    elif model_type == "resnet":
        from models.resnet_planner import ResNetPlanner
        return ResNetPlanner(
            token_dim=cfg.get("token_dim", 128),
            num_heads=cfg.get("nhead", 4),
            enc_layers=cfg.get("enc_layers", 2),
            dec_layers=cfg.get("dec_layers", 3),
            multiscale=cfg.get("multiscale", False),
            backbone=cfg.get("backbone", "resnet50"),
            frozen=cfg.get("frozen_backbone", True),
        )
    else:
        raise ValueError(
            f"Unknown model type {model_type!r}. "
            "Supported: mlp, transformer, front_cam, front_cam_depth, resnet"
        )


def _load_weights_from_lightning_ckpt(model: nn.Module, ckpt_path: str,
                                      device: torch.device) -> nn.Module:
    """
    Load inner model weights from a PyTorch Lightning checkpoint.

    Lightning saves state_dict keys prefixed with 'model.' (e.g.
    'model.drive_head.weight').  We strip the prefix and load into the
    raw nn.Module.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["state_dict"]

    # Strip 'model.' prefix
    inner_state = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            inner_state[k[len("model."):]] = v
        else:
            inner_state[k] = v  # fall back: load as-is

    model.load_state_dict(inner_state, strict=True)
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Vision-model image preprocessor
# ---------------------------------------------------------------------------

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def preprocess_image(rgba: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Convert a CARLA camera RGBA numpy array (H, W, 4) uint8
    to a normalised RGB tensor (1, 1, 3, IMAGE_H, IMAGE_W).
    """
    import torchvision.transforms.functional as TF
    from PIL import Image

    rgb = rgba[:, :, :3]                       # drop alpha → (H, W, 3)
    pil = Image.fromarray(rgb, mode="RGB")
    pil = pil.resize((IMAGE_W, IMAGE_H), Image.BILINEAR)
    t   = TF.to_tensor(pil)                    # (3, H, W) float32 in [0,1]
    t   = (t - _IMAGENET_MEAN) / _IMAGENET_STD
    return t.unsqueeze(0).unsqueeze(0).to(device)   # (1, 1, 3, H, W)


# ---------------------------------------------------------------------------
# The Agent
# ---------------------------------------------------------------------------

class E2EBench2DriveAgent:
    """
    Bench2Drive / CARLA leaderboard agent wrapping our trajectory-prediction
    models.

    This class is intentionally not a subclass of AutonomousAgent at the
    module level so that the file can be imported without CARLA being present.
    Inside the CARLA environment the leaderboard injects the correct base
    class at import time.  If you need to subclass explicitly (e.g. for
    offline testing), see the end of this file.

    Config keys (YAML):
        model.type          : mlp | transformer | front_cam | resnet | ...
        model.*             : architecture hyper-parameters (see _build_model_from_cfg)
        ckpt_path           : path to Lightning checkpoint (.ckpt)
        device              : "cuda" | "cpu" (default: "cuda" if available)
        max_speed           : m/s speed cap (default 8.0)
        steer_lookahead     : seconds ahead for steering (default 1.0)
        speed_lookahead     : seconds ahead for speed target (default 2.0)
        k_steer             : steering gain (default 1.0)
        k_throttle          : throttle gain (default 0.5)
        k_brake             : brake gain (default 1.5)
    """

    # Set to True for models that take images (front_cam, resnet, vision_transformer)
    _VISION_MODELS = {"front_cam", "front_cam_depth", "resnet", "vision_transformer"}

    def setup(self, path_to_conf_file: str) -> None:
        """Called once before evaluation starts."""
        import yaml

        with open(path_to_conf_file) as f:
            cfg = yaml.safe_load(f)

        # ── Device ────────────────────────────────────────────────────────
        device_str = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self._device = torch.device(device_str)

        # ── Model ─────────────────────────────────────────────────────────
        model_cfg  = cfg["model"]
        model_type = model_cfg["type"]
        self._needs_images = model_type in self._VISION_MODELS

        model = _build_model_from_cfg(model_cfg)
        model = _load_weights_from_lightning_ckpt(
            model, cfg["ckpt_path"], self._device
        )
        self._model = model

        # ── Controller ────────────────────────────────────────────────────
        self._controller = TrajectoryController(
            max_speed       = cfg.get("max_speed",       8.0),
            steer_lookahead = cfg.get("steer_lookahead", 1.0),
            speed_lookahead = cfg.get("speed_lookahead", 2.0),
            k_steer         = cfg.get("k_steer",         1.0),
            k_throttle      = cfg.get("k_throttle",      0.5),
            k_brake         = cfg.get("k_brake",         1.5),
        )

        # ── History buffers ───────────────────────────────────────────────
        # Each entry: (world_x, world_y, speed_m/s, accel_3d)
        self._pos_buf:   Deque[Tuple[float, float]] = deque(maxlen=PAST_STEPS_TOTAL)
        self._speed_buf: Deque[float]               = deque(maxlen=PAST_STEPS_TOTAL)
        self._accel_buf: Deque[List[float]]         = deque(maxlen=PAST_STEPS_TOTAL)

        # ── Route / command ───────────────────────────────────────────────
        # Populated by set_global_plan(); list of (GPS, RoadOption) pairs
        self._global_plan_gps   = []
        self._global_plan_world = []
        self._current_command   = 5  # LANEFOLLOW default

        # ── Metric tracking ───────────────────────────────────────────────
        self.metric_info: Dict = {}
        self._step = 0

        # ── Hero vehicle reference (set by leaderboard framework) ─────────
        # self._hero is injected by the base class after setup() completes.

        print(f"[E2EBench2DriveAgent] model={model_type}  "
              f"ckpt={cfg['ckpt_path']}  device={device_str}  "
              f"vision={self._needs_images}")

    # ── Sensor list ───────────────────────────────────────────────────────

    def sensors(self) -> List[dict]:
        """Return the sensor suite required by our models."""
        sensors = [
            # Front-facing RGB camera (used by all vision models; ignored by MLP/transformer)
            {
                "type":   "sensor.camera.rgb",
                "id":     CAM_FRONT_ID,
                "x":      0.80, "y": 0.0, "z": 1.60,
                "roll":   0.0,  "pitch": 0.0, "yaw": 0.0,
                "width":  1600, "height": 900, "fov": 70,
            },
            # IMU — provides accelerometer (3-axis) and compass heading
            {
                "type": "sensor.other.imu",
                "id":   IMU_ID,
                "x": 0.0, "y": 0.0, "z": 0.0,
                "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
            },
            # Speedometer
            {
                "type": "sensor.speedometer",
                "id":   SPEED_ID,
            },
        ]
        return sensors

    # ── Route callback ────────────────────────────────────────────────────

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        """Store the planned route (called once before the first run_step)."""
        self._global_plan_gps   = global_plan_gps
        self._global_plan_world = global_plan_world_coord

    # ── Navigation command ────────────────────────────────────────────────

    def _update_command(self, ego_x: float, ego_y: float) -> int:
        """
        Walk the global plan to find the closest future waypoint and return
        its RoadOption mapped to our 0-indexed command.
        """
        if not self._global_plan_world:
            return 3   # LANEFOLLOW

        # Find the closest plan waypoint ahead of the current position
        best_dist = float("inf")
        best_cmd  = 5   # LANEFOLLOW
        for transform, road_option in self._global_plan_world:
            wx = transform.location.x
            wy = transform.location.y
            dist = math.sqrt((wx - ego_x) ** 2 + (wy - ego_y) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_cmd  = int(road_option)   # RoadOption enum value

        return COMMAND_MAP.get(best_cmd, 3)

    # ── Per-step helpers ──────────────────────────────────────────────────

    def _extract_state(self, input_data: dict):
        """
        Extract (world_x, world_y, theta, speed, accel_3d) from sensor data
        and the CARLA hero vehicle transform.

        theta convention: 0 = north, π/2 = east  (matches training data).
        accel_3d includes gravitational component, exactly as stored in
        bench2drive annotations (raw IMU accelerometer readings).
        """
        # World position & heading from hero vehicle transform (most accurate)
        transform = self._hero.get_transform()
        wx = transform.location.x
        wy = transform.location.y
        theta = carla_yaw_to_theta(transform.rotation.yaw)

        # Speed from speedometer sensor
        speed = float(input_data[SPEED_ID][1]["speed"])

        # Raw IMU accelerometer: [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, compass]
        imu = input_data[IMU_ID][1]
        accel_3d = [float(imu[0]), float(imu[1]), float(imu[2])]

        return wx, wy, theta, speed, accel_3d

    def _build_batch(self, wx: float, wy: float, theta: float) -> dict:
        """
        Assemble the batch dict expected by our models from history buffers.

        Returns a single-sample batch (B=1) on self._device.
        """
        # World positions → numpy (PAST_STEPS_TOTAL, 2)
        pos_np   = np.array(list(self._pos_buf),   dtype=np.float32)
        speed_np = np.array(list(self._speed_buf), dtype=np.float32)
        accel_np = np.array(list(self._accel_buf), dtype=np.float32)

        # Transform past positions to current ego frame
        past_traj = world_to_ego(pos_np, wx, wy, theta).astype(np.float32)

        cmd = self._current_command

        batch = {
            "past_traj":    torch.from_numpy(past_traj).unsqueeze(0).to(self._device),
            "speed":        torch.from_numpy(speed_np).unsqueeze(0).to(self._device),
            "acceleration": torch.from_numpy(accel_np).unsqueeze(0).to(self._device),
            "command":      torch.tensor([cmd], dtype=torch.long, device=self._device),
        }
        return batch

    # ── Main step ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def run_step(self, input_data: dict, timestamp: float):
        """
        Called at every simulation tick (~10 Hz).

        1. Extract sensor state.
        2. Append to history buffers.
        3. Return zero control until buffers are full.
        4. Run model inference → (50, 2) ego-frame trajectory.
        5. Convert trajectory to vehicle control via TrajectoryController.
        6. Record metric_info for smoothness/efficiency scoring.
        """
        import carla   # only available inside the CARLA environment

        wx, wy, theta, speed, accel_3d = self._extract_state(input_data)

        # Append to rolling buffers
        self._pos_buf.append((wx, wy))
        self._speed_buf.append(speed)
        self._accel_buf.append(accel_3d)

        # Update navigation command
        self._current_command = self._update_command(wx, wy)

        # ── Wait until history is full ─────────────────────────────────────
        if len(self._pos_buf) < PAST_STEPS_TOTAL:
            control = carla.VehicleControl()
            control.throttle = 0.0
            control.steer    = 0.0
            control.brake    = 1.0
            self._step += 1
            return control

        # ── Build batch ────────────────────────────────────────────────────
        batch = self._build_batch(wx, wy, theta)

        # Attach image for vision models
        if self._needs_images:
            rgba = input_data[CAM_FRONT_ID][1]   # (H, W, 4) uint8
            batch["images"] = preprocess_image(rgba, self._device)

        # ── Inference ──────────────────────────────────────────────────────
        raw_output = self._model(batch)

        # Handle dict output (MultiTask models return {"future_traj": ..., ...})
        if isinstance(raw_output, dict):
            traj_tensor = raw_output["future_traj"]
        else:
            traj_tensor = raw_output   # plain E2EDrivingModule output

        # (1, 50, 2) → (50, 2) numpy
        traj = traj_tensor[0].cpu().numpy()   # ego frame: x=fwd, y=left

        # ── Convert trajectory to vehicle control ──────────────────────────
        throttle, steer, brake = self._controller.control(traj, speed)

        control = carla.VehicleControl()
        control.throttle = float(throttle)
        control.steer    = float(steer)
        control.brake    = float(brake)

        # ── Record metric_info (needed by Bench2Drive for smoothness/efficiency) ──
        self.metric_info[self._step] = self._get_metric_info()
        self._step += 1

        return control

    # ── Metric info ───────────────────────────────────────────────────────

    def _get_metric_info(self) -> dict:
        """
        Collect ego vehicle state for Bench2Drive metric computation.
        Called at every run_step (≈10 Hz; leaderboard upsamples to 20 Hz
        internally if needed).
        """
        transform = self._hero.get_transform()
        velocity  = self._hero.get_velocity()
        accel     = self._hero.get_acceleration()
        ang_vel   = self._hero.get_angular_velocity()

        return {
            "location":         [transform.location.x,
                                  transform.location.y,
                                  transform.location.z],
            "rotation":         [transform.rotation.roll,
                                  transform.rotation.pitch,
                                  transform.rotation.yaw],
            "forward_vector":   [transform.get_forward_vector().x,
                                  transform.get_forward_vector().y,
                                  transform.get_forward_vector().z],
            "velocity":         [velocity.x, velocity.y, velocity.z],
            "acceleration":     [accel.x,   accel.y,   accel.z],
            "angular_velocity": [ang_vel.x, ang_vel.y, ang_vel.z],
        }

    # ── Cleanup ───────────────────────────────────────────────────────────

    def destroy(self) -> None:
        """Called after the route finishes; release resources if needed."""
        pass
