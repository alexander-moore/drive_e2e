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

import glob
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

# AutonomousAgent is only available when the leaderboard is on PYTHONPATH.
# Fall back to plain object so dry_run.py can import this file without CARLA.
try:
    from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
    _AGENT_BASE = AutonomousAgent
except ImportError:
    _AGENT_BASE = object
    Track = None


def get_entry_point():
    """Required by the Bench2Drive leaderboard evaluator."""
    return "E2EBench2DriveAgent"

# ---------------------------------------------------------------------------
# Constants — must match dataset.py
# ---------------------------------------------------------------------------

PAST_STEPS_TOTAL = 41    # 40 history frames + 1 anchor frame (current)
FUTURE_STEPS     = 50    # 5 s @ 10 Hz
IMAGE_H, IMAGE_W = 224, 224

# CARLA 0.9.15 RoadOption → our 0-indexed command (must match dataset.py)
#   CARLA: LEFT=1, RIGHT=2, STRAIGHT=3, LANEFOLLOW=4, CHANGELANELEFT=5, CHANGELANERIGHT=6
#   Ours:  LEFT=0, RIGHT=1, STRAIGHT=2, LANEFOLLOW=3
COMMAND_MAP = {1: 0, 2: 1, 3: 2, 4: 3, 5: 3, 6: 3}

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


def _make_example_batch(needs_images: bool, device: torch.device) -> dict:
    """Build a minimal dummy batch for TorchScript tracing (batch size 1)."""
    batch = {
        "past_traj":    torch.zeros(1, PAST_STEPS_TOTAL, 2, device=device),
        "speed":        torch.zeros(1, PAST_STEPS_TOTAL, device=device),
        "acceleration": torch.zeros(1, PAST_STEPS_TOTAL, 3, device=device),
        "command":      torch.zeros(1, dtype=torch.long, device=device),
    }
    if needs_images:
        batch["images"] = torch.zeros(1, 1, 3, IMAGE_H, IMAGE_W, device=device)
    return batch


def _try_forward(model: nn.Module, needs_images: bool,
                 device: torch.device, dtype: torch.dtype) -> Optional[Exception]:
    """
    Run a single dummy forward pass to validate a model configuration.
    Returns None on success, or the Exception on failure.
    This catches errors that only surface at runtime (e.g. ops unsupported
    in bf16 on older GPUs, or torch.compile graph breaks).
    """
    example = _make_example_batch(needs_images, device)
    if dtype != torch.float32:
        example = {
            k: v.to(dtype) if v.is_floating_point() else v
            for k, v in example.items()
        }
    try:
        with torch.no_grad():
            model(example)
        return None
    except Exception as exc:
        return exc


def _optimize_model(
    model: nn.Module,
    needs_images: bool,
    device: torch.device,
    inference_mode: str,
    quantization: str,
) -> Tuple[nn.Module, torch.dtype]:
    """
    Apply optional precision casting and graph compilation.

    Called once during setup() after weights are loaded and model is in eval().
    Runs a warmup forward pass after each step to catch runtime failures early
    (e.g. ops unsupported in bf16, torch.compile graph breaks) and falls back
    gracefully, printing exactly what is active so the user always knows.

    Returns (model, inference_dtype). The dtype is used in run_step to cast
    input batches before each forward pass.

    Default: inference_mode="compile", quantization="bf16"

    inference_mode : "compile"     — torch.compile() — default, best for CUDA
                     "torchscript" — torch.jit.trace + optimize_for_inference
                     "pytorch"     — eager mode, no graph optimization

    quantization   : "bf16"    — bfloat16 (default; Ampere+ GPUs recommended)
                     "fp16"    — float16 (older CUDA GPUs)
                     "none"    — float32, no precision change
                     "dynamic" — INT8 dynamic quant on Linear layers (CPU only)
    """
    _TAG = "[E2EBench2DriveAgent]"
    dtype = torch.float32

    # ── Step 1: Precision ────────────────────────────────────────────────
    if quantization in ("bf16", "fp16"):
        target_dtype = torch.bfloat16 if quantization == "bf16" else torch.float16
        if device.type != "cuda":
            print(f"{_TAG} NOTE: {quantization} works best on CUDA; on CPU it may be slower than float32.")

        print(f"{_TAG} Trying {quantization} precision...")
        candidate = model.to(target_dtype)
        err = _try_forward(candidate, needs_images, device, target_dtype)
        if err is None:
            model, dtype = candidate, target_dtype
            print(f"{_TAG} {quantization} active — model weights and inputs cast to {target_dtype}.")
        else:
            # bf16 failed (common on pre-Ampere GPUs) → try fp16 before giving up
            model = model.to(torch.float32)   # restore
            if quantization == "bf16":
                print(f"{_TAG} WARNING: bf16 failed ({err}). Trying fp16 fallback...")
                candidate = model.to(torch.float16)
                err2 = _try_forward(candidate, needs_images, device, torch.float16)
                if err2 is None:
                    model, dtype = candidate, torch.float16
                    print(f"{_TAG} fp16 active (bf16 was unsupported on this GPU).")
                else:
                    model = model.to(torch.float32)
                    print(f"{_TAG} WARNING: fp16 also failed ({err2}). Running float32.")
            else:
                print(f"{_TAG} WARNING: fp16 failed ({err}). Running float32.")

    elif quantization == "dynamic":
        if device.type != "cpu":
            print(
                f"{_TAG} WARNING: dynamic INT8 quantization requires CPU but device is CUDA. "
                "Use quantization: bf16 or fp16 for CUDA. Skipping."
            )
        else:
            print(f"{_TAG} Applying dynamic INT8 quantization (Linear layers)...")
            model = torch.quantization.quantize_dynamic(
                model, {nn.Linear}, dtype=torch.qint8
            )
            print(f"{_TAG} Dynamic INT8 quantization active.")

    elif quantization not in ("none", ""):
        print(f"{_TAG} WARNING: unknown quantization={quantization!r}; running float32.")

    # ── Step 2: Graph compilation ─────────────────────────────────────────
    if inference_mode == "compile":
        print(f"{_TAG} Trying torch.compile()...")
        try:
            compiled = torch.compile(model)
            # Warmup triggers the actual compilation and catches graph-break errors
            err = _try_forward(compiled, needs_images, device, dtype)
            if err is None:
                model = compiled
                print(f"{_TAG} torch.compile() active (warmup pass completed).")
            else:
                print(
                    f"{_TAG} WARNING: torch.compile() warmup failed ({err}). "
                    "Falling back to eager PyTorch."
                )
        except Exception as exc:
            print(f"{_TAG} WARNING: torch.compile() failed ({exc}). Falling back to eager PyTorch.")

    elif inference_mode == "torchscript":
        print(f"{_TAG} Trying TorchScript trace...")
        example_batch = _make_example_batch(needs_images, device)
        if dtype != torch.float32:
            example_batch = {
                k: v.to(dtype) if v.is_floating_point() else v
                for k, v in example_batch.items()
            }
        try:
            scripted = torch.jit.trace(model, example_inputs=(example_batch,))
            scripted = torch.jit.optimize_for_inference(scripted)
            print(f"{_TAG} TorchScript trace + optimize_for_inference active.")
            return scripted, dtype
        except Exception as exc:
            print(
                f"{_TAG} WARNING: TorchScript tracing failed ({exc}). "
                "Falling back to eager PyTorch."
            )

    elif inference_mode not in ("pytorch", ""):
        print(f"{_TAG} WARNING: unknown inference_mode={inference_mode!r}; using eager PyTorch.")

    # ── Summary ───────────────────────────────────────────────────────────
    mode_label = "torch.compile" if inference_mode == "compile" else inference_mode
    print(
        f"{_TAG} Inference ready — "
        f"mode={mode_label}  precision={dtype}  device={device}"
    )

    return model, dtype


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
# ---------------------------------------------------------------------------
# Frame-saving helper (for MP4 recording)
# ---------------------------------------------------------------------------

def _save_frame(
    rgba: "np.ndarray",
    traj: "np.ndarray",
    speed: float,
    throttle: float,
    steer: float,
    brake: float,
    step: int,
    frame_dir: str,
) -> None:
    """
    Save a single annotated camera frame as PNG.

    Draws the predicted trajectory as a bird's-eye inset (bottom-right corner)
    and overlays a HUD with speed / control values in the top-left.

    Args:
        rgba:       (H, W, 4) uint8 CARLA front-camera image.
        traj:       (50, 2) ego-frame waypoints (x=fwd, y=left), metres.
        speed:      current ego speed in m/s.
        throttle/steer/brake: controller outputs [0, 1].
        step:       frame index (used for filename ordering).
        frame_dir:  directory to write PNG files into.
    """
    import cv2  # available in b2d conda env

    # Convert RGBA → BGR for OpenCV
    bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    H, W = bgr.shape[:2]

    # ── HUD overlay (top-left) ────────────────────────────────────────────
    hud_lines = [
        f"Step   : {step:04d}",
        f"Speed  : {speed:.1f} m/s",
        f"Throttle: {throttle:.2f}",
        f"Steer  : {steer:.2f}",
        f"Brake  : {brake:.2f}",
    ]
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness  = 1
    line_h     = 18
    pad        = 6
    for i, line in enumerate(hud_lines):
        y = pad + (i + 1) * line_h
        cv2.putText(bgr, line, (pad, y), font, font_scale,
                    (0, 0, 0),   thickness + 1, cv2.LINE_AA)  # shadow
        cv2.putText(bgr, line, (pad, y), font, font_scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)

    # ── Bird's-eye trajectory inset (bottom-right) ────────────────────────
    bev_size  = min(H, W) // 3   # ~75 px for 224×224
    bev_range = 20.0             # metres shown ahead/side

    bev = np.zeros((bev_size, bev_size, 3), dtype=np.uint8)
    bev[:] = (30, 30, 30)        # dark background

    def world_to_bev(x_fwd, y_left):
        """Ego-frame metres → pixel coords inside bev canvas."""
        px = int(bev_size // 2 - y_left / bev_range * bev_size // 2)
        py = int(bev_size - 2   - x_fwd  / bev_range * (bev_size - 4))
        return np.clip(px, 0, bev_size - 1), np.clip(py, 0, bev_size - 1)

    # Draw ego vehicle marker
    cx, cy = world_to_bev(0, 0)
    cv2.circle(bev, (cx, cy), 4, (0, 200, 255), -1)

    # Draw trajectory dots (green → yellow gradient)
    n = len(traj)
    for idx, (x_fwd, y_left) in enumerate(traj):
        px, py = world_to_bev(x_fwd, y_left)
        t      = idx / max(n - 1, 1)
        color  = (0, int(255 * (1 - t)), int(255 * t))  # green→red
        cv2.circle(bev, (px, py), 1, color, -1)

    # Paste inset into bottom-right of frame
    bev_border = 4
    y1 = H - bev_size - bev_border
    x1 = W - bev_size - bev_border
    bgr[y1:y1 + bev_size, x1:x1 + bev_size] = bev

    # ── Write PNG ─────────────────────────────────────────────────────────
    out_path = os.path.join(frame_dir, f"{step:06d}.png")
    cv2.imwrite(out_path, bgr)


# The Agent
# ---------------------------------------------------------------------------

class E2EBench2DriveAgent(_AGENT_BASE):
    """
    Bench2Drive / CARLA leaderboard agent wrapping our trajectory-prediction
    models.

    Inherits from AutonomousAgent when the leaderboard is available on
    PYTHONPATH; falls back to plain object for offline testing (dry_run.py).

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

    def __init__(self, carla_host="localhost", carla_port=2000, debug=False):
        if _AGENT_BASE is not object:
            super().__init__(carla_host, carla_port, debug)
            # hero_actor is set by AutonomousAgent.__init__ → get_hero()
            self._hero = self.hero_actor

    def setup(self, path_to_conf_file: str) -> None:
        """Called once before evaluation starts."""
        import yaml

        # Required by the leaderboard to determine the sensor track.
        if Track is not None:
            self.track = Track.SENSORS

        # The leaderboard appends '+<save_name>' to the config path for logging;
        # strip it to get the actual YAML file path.
        yaml_path = path_to_conf_file.split('+')[0]
        with open(yaml_path) as f:
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

        inference_mode = cfg.get("inference_mode", "compile")
        quantization   = cfg.get("quantization",   "bf16")
        self._model, self._inference_dtype = _optimize_model(
            model, self._needs_images, self._device, inference_mode, quantization
        )

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

        # ── Video recording ───────────────────────────────────────────────
        # Frames are saved to <save_path>/frames/ and stitched to MP4 in destroy().
        save_path = os.environ.get("SAVE_PATH", "")
        if save_path:
            self._frame_dir = os.path.join(save_path, "frames")
            os.makedirs(self._frame_dir, exist_ok=True)
        else:
            self._frame_dir = None

        # ── Hero vehicle reference (set by leaderboard framework) ─────────
        # self._hero is injected by the base class after setup() completes.

        print(f"[E2EBench2DriveAgent] model={model_type}  "
              f"ckpt={cfg['ckpt_path']}  device={device_str}  "
              f"vision={self._needs_images}  "
              f"inference={inference_mode}  quantization={quantization}  "
              f"frame_dir={self._frame_dir}")

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
        Return the RoadOption command for the next waypoint AHEAD of the
        vehicle, mapped to our 0-indexed command encoding.

        Strategy: find the closest waypoint, then advance a few steps further
        along the plan so the command reflects an upcoming manoeuvre rather
        than the segment the vehicle is already on.
        """
        if not self._global_plan_world:
            return 3   # LANEFOLLOW

        # Find index of the closest plan waypoint
        best_idx  = 0
        best_dist = float("inf")
        for i, (transform, _) in enumerate(self._global_plan_world):
            wx = transform.location.x
            wy = transform.location.y
            dist = math.sqrt((wx - ego_x) ** 2 + (wy - ego_y) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_idx  = i

        # Advance 3 waypoints ahead to get the upcoming manoeuvre command
        lookahead_idx = min(best_idx + 3, len(self._global_plan_world) - 1)
        _, road_option = self._global_plan_world[lookahead_idx]
        return COMMAND_MAP.get(int(road_option), 3)

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
        rgba = None
        if self._needs_images:
            rgba = input_data[CAM_FRONT_ID][1]   # (H, W, 4) uint8
            batch["images"] = preprocess_image(rgba, self._device)

        # ── Inference ──────────────────────────────────────────────────────
        if self._inference_dtype != torch.float32:
            batch = {
                k: v.to(self._inference_dtype) if v.is_floating_point() else v
                for k, v in batch.items()
            }
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

        # ── Save annotated frame for MP4 ──────────────────────────────────
        if self._frame_dir is not None and rgba is not None:
            _save_frame(rgba, traj, speed, throttle, steer, brake,
                        self._step, self._frame_dir)

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
        """Called after the route finishes; stitch saved frames to MP4."""
        if self._frame_dir and os.path.isdir(self._frame_dir):
            frames = sorted(glob.glob(os.path.join(self._frame_dir, "*.png")))
            if frames:
                # Allow the wrapper script to name the MP4 after the scene.
                # E2E_VIDEO_NAME=HardBreakRoute_1  →  HardBreakRoute_1.mp4
                video_name = os.environ.get("E2E_VIDEO_NAME", "agent_video")
                out_dir    = os.environ.get("E2E_VIDEO_DIR",
                                            os.path.dirname(self._frame_dir))
                os.makedirs(out_dir, exist_ok=True)
                mp4_path = os.path.join(out_dir, f"{video_name}.mp4")
                cmd = (
                    f"ffmpeg -y -framerate 10 -pattern_type glob "
                    f"-i '{self._frame_dir}/*.png' "
                    f"-c:v libx264 -pix_fmt yuv420p '{mp4_path}'"
                )
                ret = os.system(cmd)
                if ret == 0:
                    print(f"[E2EBench2DriveAgent] MP4 saved → {mp4_path}")
                else:
                    print(f"[E2EBench2DriveAgent] ffmpeg failed (exit {ret}), frames kept in {self._frame_dir}")
