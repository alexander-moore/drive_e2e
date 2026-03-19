import torch
import torch.nn as nn
import torch.nn.functional as F


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
