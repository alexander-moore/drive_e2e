"""
End-to-end autonomous driving dataset built on bench2drive_mini.

Mirrors the Waymo Vision E2E challenge format:
  Inputs:
    - images      : (T_past, N_cam, C, H, W)  -- past 4s @ 10Hz = 40 frames, 6 cameras
    - past_traj   : (T_past, 2)               -- ego (x, y) in ego frame at each past step
    - speed       : (T_past,)                 -- m/s
    - acceleration: (T_past, 3)               -- m/s^2  [x, y, z]
    - command     : int  -- 2=LEFT 3=RIGHT 4=STRAIGHT 5=LANEFOLLOW

  Output:
    - future_traj : (T_future, 2)             -- ego (x, y) in ego frame at each future step
                                                 (5s @ 10Hz = 50 waypoints)

Coordinate convention (ego frame):
    x = forward (along heading), y = left
    Matches Waymo challenge convention.
"""

import gzip
import json
import math
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

# ImageNet normalisation constants (used by all pretrained torchvision backbones)
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# CARLA 0.9.15 RoadOption enum → our 0-indexed command
#   CARLA: LEFT=1, RIGHT=2, STRAIGHT=3, LANEFOLLOW=4, CHANGELANELEFT=5, CHANGELANERIGHT=6
#   Ours:  LEFT=0, RIGHT=1, STRAIGHT=2, LANEFOLLOW=3
#   Lane-change commands (5,6) are treated as LANEFOLLOW (same lane-following behaviour).
COMMAND_MAP = {1: 0, 2: 1, 3: 2, 4: 3, 5: 3, 6: 3}
CAMERAS = [
    "rgb_front", "rgb_front_left", "rgb_front_right",
    "rgb_back", "rgb_back_left", "rgb_back_right",
]

PAST_STEPS = 40        # history length (4s @ 10Hz)
PAST_STEPS_TOTAL = 41  # PAST_STEPS + anchor frame; actual tensor length
FUTURE_STEPS = 50      # 5s @ 10Hz


def world_to_ego(points_world: np.ndarray, x0: float, y0: float, theta: float) -> np.ndarray:
    """
    Transform Nx2 world-frame (x,y) points into the ego coordinate frame.

    Args:
        points_world: (N, 2) array of world-frame positions
        x0, y0: ego position in world frame
        theta: ego heading in radians (CARLA yaw)

    Returns:
        (N, 2) array in ego frame where col-0 = forward, col-1 = left
    """
    dx = points_world[:, 0] - x0
    dy = points_world[:, 1] - y0
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    # CARLA theta=0 means the vehicle faces world -Y (not +X), so the forward
    # unit vector is (sin θ, -cos θ) and the left unit vector is (cos θ, sin θ).
    x_ego =  dx * sin_t - dy * cos_t   # forward
    y_ego =  dx * cos_t + dy * sin_t   # left
    return np.stack([x_ego, y_ego], axis=1)


class Bench2DriveDataset(Dataset):
    """
    PyTorch Dataset for end-to-end driving on bench2drive_mini.

    Each sample is a (anchor_scenario, anchor_frame_idx) pair where anchor
    is the current timestep. The dataset slides a window of size
    (PAST_STEPS + 1 + FUTURE_STEPS) over each scenario.

    Args:
        root: path to bench2drive_mini directory
        scenarios: list of scenario folder names to include (None = all)
        image_size: (H, W) to resize camera images to, or None to keep original
        load_images: if False, skips image loading (useful for fast trajectory-only experiments)
        transform: optional torchvision transform applied to each camera image tensor
        load_depth: if True, loads depth_{cam}/{frame_idx:05d}.png as float32 (1, H, W)
        load_semantic: if True, loads instance_{cam}/{frame_idx:05d}.png R-channel as int64 (H, W)
        front_cam_only: if True, restrict cameras to rgb_front only (C=1 instead of 6)
    """

    def __init__(
        self,
        root: str,
        scenarios: Optional[List[str]] = None,
        image_size: Optional[Tuple[int, int]] = (224, 400),
        load_images: bool = True,
        transform=None,
        normalize: bool = False,
        load_depth: bool = False,
        load_semantic: bool = False,
        front_cam_only: bool = False,
    ):
        self.root = Path(root)
        self.image_size = image_size
        self.load_images = load_images
        self.transform = transform
        self.normalize = normalize
        self.load_depth = load_depth
        self.load_semantic = load_semantic
        self.front_cam_only = front_cam_only
        self._cameras = ["rgb_front"] if front_cam_only else CAMERAS

        if scenarios is None:
            scenarios = sorted(
                d.name for d in self.root.iterdir()
                if d.is_dir() and (d / "anno").exists()
            )

        # Build index: list of (scenario_path, anchor_frame_idx)
        self.samples: List[Tuple[Path, int]] = []
        self.anno_cache: dict = {}  # scenario_path -> sorted list of anno file paths

        for scenario_name in scenarios:
            scenario_path = self.root / scenario_name
            anno_files = sorted((scenario_path / "anno").glob("*.json.gz"))
            n = len(anno_files)
            # Need PAST_STEPS frames before anchor + FUTURE_STEPS frames after
            for anchor_idx in range(PAST_STEPS, n - FUTURE_STEPS):
                self.samples.append((scenario_path, anchor_idx))
            # Pre-load all anno dicts into memory to avoid repeated gzip+JSON I/O
            # in __getitem__ (92 file opens per sample otherwise)
            parsed = []
            for f in anno_files:
                with gzip.open(f, "rt") as fp:
                    parsed.append(json.load(fp))
            self.anno_cache[scenario_path] = parsed

    def __len__(self) -> int:
        return len(self.samples)

    _VIZ_CHUNKS = 4  # number of temporal chunks per scenario for visualization

    def _viz_scenario_key(self, scenario_path: Path, anchor_idx: int) -> str:
        """Return a viz key that encodes which temporal chunk of a scenario this
        sample belongs to (e.g. 'ScenarioName_part2'). Used by the viz buffer
        to collect samples spread across the full timeline, not just the start."""
        n = len(self.anno_cache[scenario_path])
        valid_start = PAST_STEPS
        valid_end   = n - FUTURE_STEPS  # exclusive
        chunk = (anchor_idx - valid_start) * self._VIZ_CHUNKS // (valid_end - valid_start)
        chunk = min(chunk, self._VIZ_CHUNKS - 1)
        return f"{scenario_path.name}_part{chunk}"

    def _load_anno(self, scenario_path: Path, frame_idx: int) -> dict:
        return self.anno_cache[scenario_path][frame_idx]

    def _load_depth_label(self, scenario_path: Path, cam_suffix: str, frame_idx: int) -> torch.Tensor:
        """Load depth label as float32 tensor (1, H, W).

        Returns an all-zero tensor (treated as invalid by SILogLoss) if the
        file is missing or corrupt.
        """
        img_path = scenario_path / "camera" / f"depth_{cam_suffix}" / f"{frame_idx:05d}.png"
        H, W = self.image_size if self.image_size is not None else (None, None)
        try:
            img = Image.open(img_path)
            if self.image_size is not None:
                img = img.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
            arr = np.array(img, dtype=np.float32)
            if arr.ndim == 3:
                arr = arr[..., 0]
            return torch.from_numpy(arr).unsqueeze(0)  # (1, H, W) float32
        except Exception:
            out_h = self.image_size[0] if self.image_size is not None else 1
            out_w = self.image_size[1] if self.image_size is not None else 1
            return torch.zeros(1, out_h, out_w, dtype=torch.float32)

    def _load_semantic_label(self, scenario_path: Path, cam_suffix: str, frame_idx: int) -> torch.Tensor:
        """Load semantic label from instance PNG R-channel as int64 tensor (H, W)."""
        img_path = scenario_path / "camera" / f"instance_{cam_suffix}" / f"{frame_idx:05d}.png"
        img = Image.open(img_path).convert("RGBA")
        if self.image_size is not None:
            img = img.resize((self.image_size[1], self.image_size[0]), Image.NEAREST)
        arr = np.array(img, dtype=np.int64)[..., 0]  # R channel = semantic class ID
        return torch.from_numpy(arr)  # (H, W) int64

    def _load_image(self, scenario_path: Path, camera: str, frame_idx: int) -> torch.Tensor:
        img_path = scenario_path / "camera" / camera / f"{frame_idx:05d}.jpg"
        img = Image.open(img_path).convert("RGB")
        if self.image_size is not None:
            img = img.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
        tensor = TF.to_tensor(img)  # (3, H, W), float32 in [0, 1]
        if self.normalize:
            tensor = (tensor - _IMAGENET_MEAN) / _IMAGENET_STD
        if self.transform is not None:
            tensor = self.transform(tensor)
        return tensor

    def __getitem__(self, idx: int) -> dict:
        scenario_path, anchor_idx = self.samples[idx]

        # ── anchor frame (current timestep) ──────────────────────────────────
        anchor = self._load_anno(scenario_path, anchor_idx)
        x0, y0, theta0 = anchor["x"], anchor["y"], anchor["theta"]

        # Guard: a small number of frames have theta=nan (CARLA recording artefact).
        # theta is the anchor heading used to rotate all waypoints into ego frame —
        # there is no meaningful fallback, so skip to the next sample instead.
        if not math.isfinite(theta0):
            return self[(idx + 1) % len(self)]

        # ── past trajectory (indices anchor-PAST_STEPS .. anchor, inclusive) ─
        past_world = np.array(
            [[self._load_anno(scenario_path, i)["x"],
              self._load_anno(scenario_path, i)["y"]]
             for i in range(anchor_idx - PAST_STEPS, anchor_idx + 1)],
            dtype=np.float32,
        )  # (PAST_STEPS+1, 2)
        past_traj_ego = world_to_ego(past_world, x0, y0, theta0).astype(np.float32)

        speeds = np.array(
            [self._load_anno(scenario_path, i)["speed"]
             for i in range(anchor_idx - PAST_STEPS, anchor_idx + 1)],
            dtype=np.float32,
        )  # (PAST_STEPS+1,)

        accels = np.array(
            [self._load_anno(scenario_path, i)["acceleration"]
             for i in range(anchor_idx - PAST_STEPS, anchor_idx + 1)],
            dtype=np.float32,
        )  # (PAST_STEPS+1, 3)

        # ── navigation command (from anchor frame) ────────────────────────────
        raw_cmd = anchor["command_near"]
        command = COMMAND_MAP.get(raw_cmd, 3)  # default to LANEFOLLOW

        # ── future trajectory (indices anchor+1 .. anchor+FUTURE_STEPS) ───────
        future_world = np.array(
            [[self._load_anno(scenario_path, i)["x"],
              self._load_anno(scenario_path, i)["y"]]
             for i in range(anchor_idx + 1, anchor_idx + FUTURE_STEPS + 1)],
            dtype=np.float32,
        )  # (FUTURE_STEPS, 2)
        future_traj_ego = world_to_ego(future_world, x0, y0, theta0).astype(np.float32)

        sample = {
            # Trajectories (ego frame: x=forward, y=left)
            "past_traj": torch.from_numpy(past_traj_ego),    # (41, 2)
            "future_traj": torch.from_numpy(future_traj_ego),  # (50, 2)
            # Kinematics
            "speed": torch.from_numpy(speeds),               # (41,)
            "acceleration": torch.from_numpy(accels),        # (41, 3)
            # Command
            "command": torch.tensor(command, dtype=torch.long),
            # Metadata (not used for training, helpful for debugging)
            # scenario_viz subdivides each scenario's timeline into 4 equal chunks
            # so the viz buffer collects samples from different parts of each video.
            "scenario": str(scenario_path.name),
            "scenario_viz": self._viz_scenario_key(scenario_path, anchor_idx),
            "scenario_path": str(scenario_path),
            "anchor_idx": anchor_idx,
        }

        # ── camera images ─────────────────────────────────────────────────────
        if self.load_images:
            # Load only anchor frame images (cheapest baseline)
            # Shape: (C, 3, H, W) where C=1 if front_cam_only else 6
            imgs = torch.stack([
                self._load_image(scenario_path, cam, anchor_idx)
                for cam in self._cameras
            ])
            sample["images"] = imgs

        if self.load_depth:
            depth_imgs = []
            for cam in self._cameras:
                cam_suffix = cam[len("rgb_"):]  # e.g. "rgb_front" -> "front"
                depth_imgs.append(self._load_depth_label(scenario_path, cam_suffix, anchor_idx))
            sample["depth"] = torch.stack(depth_imgs)  # (C, 1, H, W) float32

        if self.load_semantic:
            sem_imgs = []
            for cam in self._cameras:
                cam_suffix = cam[len("rgb_"):]
                sem_imgs.append(self._load_semantic_label(scenario_path, cam_suffix, anchor_idx))
            sample["semantic"] = torch.stack(sem_imgs)  # (C, H, W) int64

        return sample


# ── Convenience factory functions ─────────────────────────────────────────────

# Bench2Drive-mini scenarios (from tools/download_mini.sh) used as the
# canonical validation split. Excluded from training regardless of whether
# they appear in the full dataset.
BENCH2DRIVE_MINI_SCENARIOS = [
    "HardBreakRoute_Town01_Route30_Weather3",
    "DynamicObjectCrossing_Town02_Route13_Weather6",
    "Accident_Town03_Route156_Weather0",
    "YieldToEmergencyVehicle_Town04_Route165_Weather7",
    "ConstructionObstacle_Town05_Route68_Weather8",
    "ParkedObstacle_Town10HD_Route371_Weather7",
    "ControlLoss_Town11_Route401_Weather11",
    "AccidentTwoWays_Town12_Route1444_Weather0",
    "OppositeVehicleTakingPriority_Town13_Route600_Weather2",
    "VehicleTurningRoute_Town15_Route443_Weather1",
]


def make_datasets(root: str, val_scenarios: Optional[List[str]] = None, **kwargs):
    """
    Split bench2drive into train/val datasets.

    By default uses the Bench2Drive-mini scenarios as validation. Any mini
    scenario present in the dataset root is used for val and excluded from
    training; mini scenarios not present are simply skipped.
    """
    all_scenarios = sorted(
        d.name for d in Path(root).iterdir()
        if d.is_dir() and (d / "anno").exists()
    )
    if val_scenarios is None:
        val_scenarios = [s for s in BENCH2DRIVE_MINI_SCENARIOS if s in all_scenarios]
    train_scenarios = [s for s in all_scenarios if s not in val_scenarios]

    train_ds = Bench2DriveDataset(root, scenarios=train_scenarios, **kwargs)
    val_ds = Bench2DriveDataset(root, scenarios=val_scenarios, **kwargs)
    return train_ds, val_ds


if __name__ == "__main__":
    import time

    root = "/workspace/bench2drive_mini"
    print("Building dataset index...")
    t0 = time.time()
    train_ds, val_ds = make_datasets(root, image_size=(224, 400), load_images=True)
    print(f"  Train samples: {len(train_ds)},  Val samples: {len(val_ds)}  ({time.time()-t0:.1f}s)")

    print("\nLoading sample 0...")
    t0 = time.time()
    sample = train_ds[0]
    print(f"  Load time: {time.time()-t0:.2f}s")
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:16s}: {tuple(v.shape)}  dtype={v.dtype}  "
                  f"min={v.min():.3f}  max={v.max():.3f}")
        else:
            print(f"  {k:16s}: {v}")

    print("\nLoading 10 samples to check speed...")
    t0 = time.time()
    for i in range(10):
        _ = train_ds[i * 50]
    print(f"  10 samples in {time.time()-t0:.2f}s  ({(time.time()-t0)/10*1000:.0f}ms/sample)")
