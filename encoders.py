
# encoders.py
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
from torch import Tensor

from linear_attention import LinearTransformerBlock


class PatchEncoder(nn.Module):
    """pFormer: within-patch encoder using linearized attention."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_patches: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.num_patches = num_patches
        self.hidden_dim = hidden_dim

        self.layers = nn.ModuleList(
            [
                LinearTransformerBlock(
                    embed_dim=hidden_dim,
                    num_heads=num_heads,
                    ffn_dim=hidden_dim * 4,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self, X: Tensor, patch_ids: Tensor
    ) -> Tuple[Tensor, Tensor]:
        S, N, _ = X.shape
        device = X.device
        hidden_dim = self.hidden_dim
        K = self.num_patches

        H_nodes = torch.zeros(S, N, hidden_dim, device=device)
        patch_summaries = torch.zeros(S, K, hidden_dim, device=device)

        X_proj = self.input_proj(X)

        for s in range(S):
            X_s = X_proj[s]
            pids_s = patch_ids[s].long()
            node_indices_per_patch: List[Tensor] = []
            lengths = []
            for k in range(K):
                idx = (pids_s == k).nonzero(as_tuple=False).view(-1)
                node_indices_per_patch.append(idx)
                lengths.append(int(idx.numel()))
            max_len = max(lengths) if lengths else 0
            if max_len == 0:
                continue

            patch_inputs = torch.zeros(K, max_len, hidden_dim, device=device)
            patch_masks = torch.zeros(K, max_len, dtype=torch.bool, device=device)
            for k in range(K):
                idx = node_indices_per_patch[k]
                if idx.numel() == 0:
                    continue
                Lk = idx.numel()
                patch_inputs[k, :Lk] = X_s[idx]
                patch_masks[k, :Lk] = True

            h = patch_inputs
            for layer in self.layers:
                h = layer(h, mask=patch_masks)

            for k in range(K):
                idx = node_indices_per_patch[k]
                if idx.numel() == 0:
                    continue
                Lk = idx.numel()
                H_nodes[s, idx] = h[k, :Lk]
                patch_summaries[s, k] = h[k, :Lk].mean(dim=0)

        return H_nodes, patch_summaries


class ClusterEncoder(nn.Module):
    """cFormer: inter-patch encoder over patch summaries."""

    def __init__(
        self,
        hidden_dim: int,
        num_patches: int,
        num_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_patches = num_patches
        self.layers = nn.ModuleList(
            [
                LinearTransformerBlock(
                    embed_dim=hidden_dim,
                    num_heads=num_heads,
                    ffn_dim=hidden_dim * 4,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, H_nodes: Tensor, patch_ids: Tensor
    ) -> Tuple[Tensor, Tensor]:
        S, N, hidden_dim = H_nodes.shape
        device = H_nodes.device
        K = self.num_patches

        patch_summaries = torch.zeros(S, K, hidden_dim, device=device)
        patch_masks = torch.zeros(S, K, dtype=torch.bool, device=device)

        for s in range(S):
            H_s = H_nodes[s]
            pids_s = patch_ids[s].long()
            for k in range(K):
                mask_k = pids_s == k
                if mask_k.any():
                    patch_summaries[s, k] = H_s[mask_k].mean(dim=0)
                    patch_masks[s, k] = True

        h = patch_summaries
        for layer in self.layers:
            h = layer(h, mask=patch_masks)

        patch_summaries_updated = h

        H_out = torch.zeros_like(H_nodes)
        for s in range(S):
            pids_s = patch_ids[s].long()
            patch_out_s = patch_summaries_updated[s]
            patch_broadcast = patch_out_s[pids_s]
            combined = torch.cat([H_nodes[s], patch_broadcast], dim=-1)
            node_update = self.node_mlp(combined)
            H_out[s] = H_nodes[s] + self.dropout(node_update)

        return H_out, patch_summaries_updated


class TemporalEncoder(nn.Module):
    """tFormer: temporal encoder over per-node sequences across slots."""

    def __init__(
        self,
        spatial_dim: int,
        patch_dim: int,
        model_dim: int,
        num_slots: int,
        num_layers: int = 2,
        num_heads: int = 4,
        time_dim: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.spatial_dim = spatial_dim
        self.patch_dim = patch_dim
        self.model_dim = model_dim
        self.num_slots = num_slots
        self.time_dim = time_dim

        self.time_embedding = nn.Embedding(num_slots, time_dim)
        input_dim = spatial_dim + patch_dim + time_dim + 1
        self.input_proj = nn.Linear(input_dim, model_dim)

        self.layers = nn.ModuleList(
            [
                LinearTransformerBlock(
                    embed_dim=model_dim,
                    num_heads=num_heads,
                    ffn_dim=model_dim * 4,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        H_spatial: Tensor,
        patch_summaries: Tensor,
        patch_ids: Tensor,
        temporal_mask: Optional[Tensor] = None,
    ) -> Tensor:
        S, N, _ = H_spatial.shape
        device = H_spatial.device

        if temporal_mask is None:
            temporal_mask = torch.ones(S, N, dtype=torch.bool, device=device)

        slot_indices = torch.arange(S, device=device)
        time_emb = self.time_embedding(slot_indices)

        patch_broadcast_all = torch.zeros(S, N, self.patch_dim, device=device)
        for s in range(S):
            pids_s = patch_ids[s].long()
            patch_broadcast_all[s] = patch_summaries[s][pids_s]

        time_embed_all = time_emb.unsqueeze(1).expand(S, N, self.time_dim)
        temporal_mask_float = temporal_mask.float().unsqueeze(-1)

        concat = torch.cat(
            [H_spatial, patch_broadcast_all, time_embed_all, temporal_mask_float],
            dim=-1,
        )
        temporal_input = self.input_proj(concat)
        temporal_input = self.dropout(temporal_input)

        x = temporal_input.permute(1, 0, 2).contiguous()
        mask_seq = temporal_mask.permute(1, 0).contiguous()

        for layer in self.layers:
            x = layer(x, mask=mask_seq)

        H_t = x.permute(1, 0, 2).contiguous()
        return H_t
