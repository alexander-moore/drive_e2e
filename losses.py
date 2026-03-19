"""
losses.py — loss functions for E2E driving and auxiliary perception tasks.

Trajectory losses
-----------------
imitation_l1  (SOTA default)
    L1 over every future timestep, matching VAD (ICCV 2023), UniAD (CVPR 2023),
    and SparseDrive (ECCV 2024).  Supervises all T future waypoints uniformly.
    L1 is preferred over L2 because it is less sensitive to large outlier
    deviations and gives cleaner gradients for the long tail of the trajectory.

avg_l2  (legacy metric — kept for checkpoint compatibility and SOTA comparison)
    Mean L2 at the three standard horizon indices (1 s / 2 s / 3 s).
    Used as the primary *evaluation* metric across SOTA papers so we keep it
    for val checkpointing, but it is a poor *training* signal because it
    supervises only 3 of 50 future timesteps.

SOTA trajectory loss comparison
---------------------------------
| Paper                  | Training loss              | Extra terms                          | Key metrics                        |
|------------------------|----------------------------|--------------------------------------|------------------------------------|
| VAD (ICCV 2023)        | L1 imitation, all steps    | Collision hinge, boundary hinge,     | L2@1s/2s/3s + collision rate       |
|                        |                            | lane-direction cosine                |                                    |
| UniAD (CVPR 2023 Best) | Traj + occupancy IoU       | Tracking, mapping, motion, occupancy | L2, 0.31% collision, minADE 0.71 m |
| SparseDrive (ECCV 2024)| Focal + L1, winner-takes-all| 2-stage training                    | 19% better L2 than UniAD           |
| VADv2 (2024)           | KL divergence (probabilistic)| Conflict loss on unsafe actions    | 85.1 driving score (CARLA)         |

Gaps vs SOTA (as of 2024)
--------------------------
1. Supervision density  — we cover all T steps; old avg_l2 covered only 3 of 50.
2. Loss type            — L1 here; SOTA uses L1 / smooth-L1 (Huber).
3. Safety constraints   — no collision or boundary terms yet; VAD adds these.
4. Metrics              — we log ADE/FDE + L2@horizons; SOTA also tracks collision rate.

Future improvements
-------------------
* Smooth-L1 (Huber) loss for robustness to outlier waypoints.
* Collision hinge loss: max(0, safety_margin - dist_to_nearest_agent).
  Requires surrounding-agent boxes in the batch (not present yet).
* Boundary hinge loss: penalise predicted points outside the drivable area.
  Requires a lane/road mask per sample.
* Probabilistic output + KL divergence loss (VADv2 style).
* Winner-takes-all over K trajectory hypotheses (SparseDrive style).

Perception losses (auxiliary)
------------------------------
abs_rel   — Mean Absolute Relative Error for monocular depth.
DiceLoss  — Multi-class Dice for semantic segmentation.
SILogLoss — Scale-Invariant Log Loss (Eigen et al. 2014) for depth.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def imitation_l1(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """L1 imitation loss over all future timesteps (VAD / UniAD / SparseDrive style).

    Supervises every waypoint in the predicted trajectory, unlike the legacy
    avg_l2 metric which only evaluates at 3 horizon indices.  L1 is less
    sensitive to large outlier deviations than L2 and is standard in SOTA
    E2E planners.

    Args:
        pred: (B, T, 2) — predicted future waypoints in ego frame [metres]
        gt:   (B, T, 2) — ground-truth future waypoints in ego frame [metres]
    Returns:
        scalar tensor [metres]
    """
    return F.l1_loss(pred, gt)


def abs_rel(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Mean Absolute Relative Error: mean(|pred - gt| / gt) over valid pixels."""
    mask = gt > 0
    if mask.sum() == 0:
        return pred.sum() * 0.0
    return ((pred[mask] - gt[mask]).abs() / gt[mask]).mean()


class DiceLoss(nn.Module):
    """Multi-class Dice loss for semantic segmentation.

    Computes softmax over logits, one-hot encodes targets, then averages
    the per-class Dice coefficient across all classes and batch elements.
    Inherently class-balanced: rare classes contribute equally to the loss.

    Combine with cross-entropy for best results:
        loss = ce_loss + 0.5 * DiceLoss()(logits, targets)
    """
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (N, C, H, W)  — raw unnormalised scores
            targets: (N, H, W)     — integer class indices in [0, C)
        """
        C = logits.shape[1]
        probs       = torch.softmax(logits, dim=1).flatten(2)          # (N, C, H*W)
        targets_oh  = F.one_hot(targets, C).permute(0, 3, 1, 2).float().flatten(2)  # (N, C, H*W)

        intersection = (probs * targets_oh).sum(dim=2)                 # (N, C)
        denominator  = probs.sum(dim=2) + targets_oh.sum(dim=2)        # (N, C)
        dice         = (2 * intersection + self.smooth) / (denominator + self.smooth)
        return 1 - dice.mean()


class SILogLoss(nn.Module):
    """Scale-Invariant Log Loss (Eigen et al. 2014).

    Works in log-space, naturally upweights near objects vs sky/far regions.
    Used by ZoeDepth, AdaBins, DPT.

    beta=0.15 is equivalent to lambda=0.85 in the original formulation.
    """
    def __init__(self, beta: float = 0.15):
        super().__init__()
        self.beta = beta

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        mask = gt > 0
        if mask.sum() == 0:
            return pred.sum() * 0.0
        p = pred[mask].clamp(min=1e-3)
        g_val = gt[mask].clamp(min=1e-3)
        g = torch.log(p) - torch.log(g_val)
        return torch.sqrt(g.var() + self.beta * g.mean().pow(2))
