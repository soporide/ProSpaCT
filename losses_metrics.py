
# losses_metrics.py
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


def compute_predictive_loss(
    probs: Tensor,
    y: Tensor,
    mask: Optional[Tensor] = None,
    pos_weight: float = 1.0,
    neg_weight: float = 1.0,
) -> Tensor:
    eps = 1e-8
    probs = probs.clamp(min=eps, max=1.0 - eps)
    y = y.float()
    if mask is not None:
        mask_float = mask.float()
    else:
        mask_float = torch.ones_like(y)

    loss_pos = -pos_weight * y * torch.log(probs)
    loss_neg = -neg_weight * (1.0 - y) * torch.log(1.0 - probs)
    loss = (loss_pos + loss_neg) * mask_float
    denom = mask_float.sum().clamp(min=1.0)
    return loss.sum() / denom


def compute_alignment_loss(
    probs: Tensor,
    patch_ids: Tensor,
    patch_baselines: Tensor,
    mask: Optional[Tensor] = None,
) -> Tensor:
    S, N = probs.shape
    patch_ids_long = patch_ids.long()
    baseline_nodes = patch_baselines[patch_ids_long]
    if mask is not None:
        mask_float = mask.float()
    else:
        mask_float = torch.ones_like(probs)
    mse = (probs - baseline_nodes) ** 2
    mse = mse * mask_float
    denom = mask_float.sum().clamp(min=1.0)
    return mse.sum() / denom


def compute_l2_reg(model) -> Tensor:
    l2 = torch.tensor(0.0, device=next(model.parameters()).device)
    for p in model.parameters():
        l2 = l2 + (p**2).sum()
    return l2


def compute_nll(
    probs: Tensor,
    y: Tensor,
    mask: Optional[Tensor] = None,
) -> Tensor:
    return compute_predictive_loss(
        probs=probs,
        y=y,
        mask=mask,
        pos_weight=1.0,
        neg_weight=1.0,
    )


def compute_brier_score(
    probs: Tensor,
    y: Tensor,
    mask: Optional[Tensor] = None,
) -> Tensor:
    y = y.float()
    if mask is not None:
        mask_float = mask.float()
    else:
        mask_float = torch.ones_like(y)
    mse = (probs - y) ** 2 * mask_float
    denom = mask_float.sum().clamp(min=1.0)
    return mse.sum() / denom


def compute_ece(
    probs: Tensor,
    y: Tensor,
    mask: Optional[Tensor] = None,
    n_bins: int = 10,
) -> Tensor:
    device = probs.device
    y = y.float()
    if mask is not None:
        mask_flat = mask.view(-1)
    else:
        mask_flat = torch.ones_like(probs).view(-1).bool()

    probs_flat = probs.view(-1)
    y_flat = y.view(-1)

    probs_flat = probs_flat[mask_flat]
    y_flat = y_flat[mask_flat]

    if probs_flat.numel() == 0:
        return torch.tensor(0.0, device=device)

    bin_boundaries = torch.linspace(0.0, 1.0, n_bins + 1, device=device)
    ece = torch.tensor(0.0, device=device)

    for i in range(n_bins):
        low = bin_boundaries[i]
        high = bin_boundaries[i + 1]
        in_bin = (probs_flat >= low) & (probs_flat < high)
        if i == n_bins - 1:
            in_bin = (probs_flat >= low) & (probs_flat <= high)
        if in_bin.any():
            prop = in_bin.float().mean()
            bin_probs = probs_flat[in_bin]
            bin_labels = y_flat[in_bin]
            accuracy = bin_labels.mean()
            confidence = bin_probs.mean()
            ece = ece + (confidence - accuracy).abs() * prop

    return ece
