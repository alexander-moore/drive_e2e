"""
Video sequence dataset for self-supervised video prediction pretraining.

Each sample loads:
  input_frames:  (n_cams, n_frames, 3, H, W) — context frames ending at anchor
  target_frames: (n_cams, m_frames, 3, H, W) — future frames after anchor

With frame_stride > 1, every stride-th frame is sampled instead of consecutive
frames, covering a longer temporal window at lower temporal resolution.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import torch

try:
    from dataset import Bench2DriveDataset, CAMERAS
except ImportError:
    from e2e.dataset import Bench2DriveDataset, CAMERAS


class VideoSequenceDataset(Bench2DriveDataset):
    """
    Dataset for video prediction pretraining on bench2drive_mini.

    Args:
        root:          path to bench2drive_mini
        n_frames:      number of input context frames
        m_frames:      number of future frames to predict
        frame_stride:  step between sampled frames (1=consecutive, 2=every-other, etc.)
        scenarios:     list of scenario folder names (None = all)
        image_size:    (H, W) to resize camera images to
        front_cam_only: use only rgb_front camera
    """

    def __init__(
        self,
        root: str,
        n_frames: int = 4,
        m_frames: int = 4,
        frame_stride: int = 1,
        scenarios: Optional[List[str]] = None,
        image_size: Optional[Tuple[int, int]] = (224, 224),
        front_cam_only: bool = False,
    ):
        self.n_frames = n_frames
        self.m_frames = m_frames
        self.frame_stride = frame_stride

        # Call parent to build anno_cache and camera list; sample index rebuilt below.
        super().__init__(
            root=root,
            scenarios=scenarios,
            image_size=image_size,
            load_images=False,   # we handle image loading in our own __getitem__
            front_cam_only=front_cam_only,
        )

        # Rebuild samples with valid range for this stride/frame count.
        #
        # Input  window: anchor - (n_frames-1)*stride  ..  anchor  (inclusive)
        # Target window: anchor + stride  ..  anchor + m_frames*stride  (inclusive)
        #
        # Constraints:
        #   anchor - (n_frames-1)*stride >= 0   →  anchor >= (n_frames-1)*stride
        #   anchor + m_frames*stride  <= n-1    →  anchor <= n-1 - m_frames*stride
        anchor_min_offset = (n_frames - 1) * frame_stride
        anchor_max_offset = m_frames * frame_stride          # frames needed *after* anchor

        self.samples = []
        for scenario_path, anno_files in self.anno_cache.items():
            n = len(anno_files)
            anchor_min = anchor_min_offset
            anchor_max = n - 1 - anchor_max_offset           # inclusive upper bound
            for anchor_idx in range(anchor_min, anchor_max + 1):
                self.samples.append((scenario_path, anchor_idx))

    def __getitem__(self, idx: int) -> dict:
        try:
            return self._load_sample(idx)
        except Exception:
            # Corrupt or missing file — skip to the next sample
            return self[(idx + 1) % len(self)]

    def _load_sample(self, idx: int) -> dict:
        scenario_path, anchor_idx = self.samples[idx]

        # Frame indices for input (ending at anchor) and target (after anchor)
        input_indices = [
            anchor_idx - (self.n_frames - 1 - i) * self.frame_stride
            for i in range(self.n_frames)
        ]
        target_indices = [
            anchor_idx + (i + 1) * self.frame_stride
            for i in range(self.m_frames)
        ]
        all_indices = input_indices + target_indices   # n_frames + m_frames total

        # Load every frame once: (n_cams, n_frames+m_frames, 3, H, W)
        all_frames = torch.stack([
            torch.stack([self._load_image(scenario_path, cam, fi) for fi in all_indices])
            for cam in self._cameras
        ])

        return {
            # Combined sequence — primary input for the autoencoder encoder
            "all_frames":    all_frames,
            # Convenience splits used for visualization and per-phase loss logging
            "input_frames":  all_frames[:, :self.n_frames],   # (n_cams, n_frames, 3, H, W)
            "target_frames": all_frames[:, self.n_frames:],   # (n_cams, m_frames, 3, H, W)
            "scenario":      str(scenario_path.name),
            "anchor_idx":    anchor_idx,
        }


# ── Factory ───────────────────────────────────────────────────────────────────

def make_video_datasets(
    root: str,
    n_frames: int = 4,
    m_frames: int = 4,
    frame_stride: int = 1,
    val_scenarios: Optional[List[str]] = None,
    **kwargs,
) -> Tuple[VideoSequenceDataset, VideoSequenceDataset]:
    """
    Split bench2drive_mini into train/val datasets for video pretraining.
    Uses the last 2 scenarios (alphabetically) for validation by default.
    """
    all_scenarios = sorted(
        d.name for d in Path(root).iterdir()
        if d.is_dir() and (d / "anno").exists()
    )
    if val_scenarios is None:
        val_scenarios = all_scenarios[-2:]
    train_scenarios = [s for s in all_scenarios if s not in val_scenarios]

    train_ds = VideoSequenceDataset(
        root, n_frames=n_frames, m_frames=m_frames, frame_stride=frame_stride,
        scenarios=train_scenarios, **kwargs,
    )
    val_ds = VideoSequenceDataset(
        root, n_frames=n_frames, m_frames=m_frames, frame_stride=frame_stride,
        scenarios=val_scenarios, **kwargs,
    )
    return train_ds, val_ds


if __name__ == "__main__":
    import time

    root = "/workspace/bench2drive_mini"

    for stride in (1, 2, 3):
        print(f"\n── frame_stride={stride} ─────────────────────")
        t0 = time.time()
        train_ds, val_ds = make_video_datasets(
            root, n_frames=4, m_frames=4, frame_stride=stride,
            image_size=(224, 224), front_cam_only=True,
        )
        print(f"  Build time : {time.time()-t0:.2f}s")
        print(f"  Train / Val: {len(train_ds)} / {len(val_ds)} samples")

        if len(train_ds) == 0:
            print("  (empty — not enough frames for this stride)")
            continue

        t0 = time.time()
        sample = train_ds[0]
        print(f"  Load time  : {time.time()-t0:.2f}s")
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k:16s}: {tuple(v.shape)}  dtype={v.dtype}")
            else:
                print(f"  {k:16s}: {v}")
