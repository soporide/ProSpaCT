
# probabilistic_head.py
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from beta_posterior import PatchBetaPosterior


class ProbabilisticHead(nn.Module):
    """Probabilistic head with logit shrinkage and global temperature scaling."""

    def __init__(
        self,
        input_dim: int,
        num_patches: int,
        init_log_temperature: float = 0.0,
        kappa_max: float = 0.7,
        n0: float = 10.0,
    ) -> None:
        super().__init__()
        self.logit_layer = nn.Linear(input_dim, 1)
        self.log_temperature = nn.Parameter(
            torch.tensor(float(init_log_temperature), dtype=torch.float32)
        )
        self.num_patches = num_patches
        self.kappa_max = float(kappa_max)
        self.n0 = float(n0)

    def forward(
        self,
        H_t: Tensor,
        patch_ids: Tensor,
        beta_posterior: PatchBetaPosterior,
        return_logits: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        device = H_t.device
        S, N, _ = H_t.shape

        raw_logits = self.logit_layer(H_t).squeeze(-1)

        theta = beta_posterior.get_baseline_rates().to(device)
        n_eff = beta_posterior.get_effective_counts().to(device)
        baseline_logits = torch.log(theta) - torch.log(1.0 - theta)

        kappa = self.kappa_max * (self.n0 / (n_eff + self.n0))
        kappa = kappa.clamp(min=0.0, max=self.kappa_max)

        patch_ids_long = patch_ids.long()
        baseline_logits_nodes = baseline_logits[patch_ids_long]
        kappa_nodes = kappa[patch_ids_long]

        logits_shrunk = (1.0 - kappa_nodes) * raw_logits + kappa_nodes * baseline_logits_nodes

        temperature = F.softplus(self.log_temperature) + 1e-4
        probs = torch.sigmoid(logits_shrunk / temperature)

        if return_logits:
            return probs, logits_shrunk
        return probs, logits_shrunk
