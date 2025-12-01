
# model.py
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from encoders import PatchEncoder, ClusterEncoder, TemporalEncoder
from probabilistic_head import ProbabilisticHead
from beta_posterior import PatchBetaPosterior


class ProSpaCTModel(nn.Module):
    """Full ProSpaCT model: pFormer + cFormer + tFormer + ProbabilisticHead."""

    def __init__(
        self,
        input_dim: int,
        num_patches: int,
        num_slots: int,
        d_model_spatial: int = 32,
        d_model_temporal: int = 64,
        num_heads_spatial: int = 4,
        num_heads_cluster: int = 4,
        num_heads_temporal: int = 4,
        num_pformer_layers: int = 2,
        num_cformer_layers: int = 1,
        num_tformer_layers: int = 2,
        time_dim: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_patches = num_patches
        self.num_slots = num_slots

        self.patch_encoder = PatchEncoder(
            input_dim=input_dim,
            hidden_dim=d_model_spatial,
            num_patches=num_patches,
            num_layers=num_pformer_layers,
            num_heads=num_heads_spatial,
            dropout=dropout,
        )

        self.cluster_encoder = ClusterEncoder(
            hidden_dim=d_model_spatial,
            num_patches=num_patches,
            num_layers=num_cformer_layers,
            num_heads=num_heads_cluster,
            dropout=dropout,
        )

        self.temporal_encoder = TemporalEncoder(
            spatial_dim=d_model_spatial,
            patch_dim=d_model_spatial,
            model_dim=d_model_temporal,
            num_slots=num_slots,
            num_layers=num_tformer_layers,
            num_heads=num_heads_temporal,
            time_dim=time_dim,
            dropout=dropout,
        )

        self.prob_head = ProbabilisticHead(
            input_dim=d_model_temporal,
            num_patches=num_patches,
        )

    def forward(
        self,
        X: Tensor,
        patch_ids: Tensor,
        beta_posterior: PatchBetaPosterior,
        temporal_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        H_p, patch_summaries = self.patch_encoder(X, patch_ids)
        H_spatial, patch_summaries_updated = self.cluster_encoder(H_p, patch_ids)
        H_t = self.temporal_encoder(
            H_spatial=H_spatial,
            patch_summaries=patch_summaries_updated,
            patch_ids=patch_ids,
            temporal_mask=temporal_mask,
        )
        probs, logits = self.prob_head(H_t, patch_ids, beta_posterior, return_logits=True)
        return probs, logits

    def predict_with_intervals(
        self,
        X: Tensor,
        patch_ids: Tensor,
        beta_posterior: PatchBetaPosterior,
        temporal_mask: Optional[Tensor] = None,
        num_samples: int = 20,
        interval_alpha: float = 0.1,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        S, N, _ = X.shape
        samples = []

        original_mode = self.training
        try:
            for _ in range(num_samples):
                self.train()
                with torch.no_grad():
                    probs, _ = self.forward(
                        X=X,
                        patch_ids=patch_ids,
                        beta_posterior=beta_posterior,
                        temporal_mask=temporal_mask,
                    )
                samples.append(probs.unsqueeze(0))
        finally:
            if original_mode:
                self.train()
            else:
                self.eval()

        all_samples = torch.cat(samples, dim=0)
        mean = all_samples.mean(dim=0)
        lower = all_samples.quantile(interval_alpha / 2.0, dim=0)
        upper = all_samples.quantile(1.0 - interval_alpha / 2.0, dim=0)
        return mean, lower, upper
