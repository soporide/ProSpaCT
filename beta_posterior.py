
# beta_posterior.py
from typing import List, Optional, Sequence

import torch
from torch import Tensor


class PatchBetaPosterior:
    """Patch-level Beta posteriors with exponential forgetting."""

    def __init__(
        self,
        num_patches: int,
        alpha0: float = 1.0,
        beta0: float = 1.0,
        forgetting: float = 1.0,
        device: Optional[torch.device] = None,
    ) -> None:
        self.num_patches = num_patches
        self.alpha0 = float(alpha0)
        self.beta0 = float(beta0)
        self.forgetting = float(forgetting)
        self.device = device if device is not None else torch.device("cpu")
        self.alpha = torch.full(
            (num_patches,),
            self.alpha0,
            dtype=torch.float32,
            device=self.device,
        )
        self.beta = torch.full(
            (num_patches,),
            self.beta0,
            dtype=torch.float32,
            device=self.device,
        )
        self.n_eff = torch.zeros(num_patches, dtype=torch.float32, device=self.device)

    def reset(self) -> None:
        self.alpha.fill_(self.alpha0)
        self.beta.fill_(self.beta0)
        self.n_eff.zero_()

    def to(self, device: torch.device) -> "PatchBetaPosterior":
        self.device = device
        self.alpha = self.alpha.to(device)
        self.beta = self.beta.to(device)
        self.n_eff = self.n_eff.to(device)
        return self

    def _apply_forgetting(self) -> None:
        if self.forgetting < 1.0:
            self.alpha = self.alpha * self.forgetting
            self.beta = self.beta * self.forgetting
            self.n_eff = self.n_eff * self.forgetting

    def update_from_labels(
        self,
        y_list: Sequence[Tensor],
        patch_ids_list: Sequence[Tensor],
        slot_indices: Optional[Sequence[int]] = None,
    ) -> None:
        if slot_indices is None:
            slot_indices = range(len(y_list))
        for s in slot_indices:
            y_s = y_list[s].detach().to(self.device).float()
            pids_s = patch_ids_list[s].detach().to(self.device).long()
            self._apply_forgetting()
            for k in range(self.num_patches):
                mask_k = pids_s == k
                if mask_k.any():
                    count_k = mask_k.sum().float()
                    sum_y = y_s[mask_k].sum()
                    self.alpha[k] += sum_y
                    self.beta[k] += count_k - sum_y
                    self.n_eff[k] += count_k

    def get_baseline_rates(self) -> Tensor:
        theta = self.alpha / (self.alpha + self.beta + 1e-8)
        theta_clipped = theta.clamp(min=1e-4, max=1.0 - 1e-4)
        return theta_clipped

    def get_effective_counts(self) -> Tensor:
        return self.n_eff.clone()
