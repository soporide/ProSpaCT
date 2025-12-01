
# linear_attention.py
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LinearAttentionLayer(nn.Module):
    """Multi-head linearized self-attention with a positive feature map."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

    def _feature_map(self, x: Tensor) -> Tensor:
        return F.elu(x) + 1.0

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Apply linearized multi-head self-attention.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (B, L, D).
        mask : Optional[Tensor]
            Boolean mask of shape (B, L), where True indicates a valid token.
        """
        B, L, D = x.shape
        H = self.num_heads
        Dh = self.head_dim

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = Q.view(B, L, H, Dh).transpose(1, 2).contiguous().view(B * H, L, Dh)
        K = K.view(B, L, H, Dh).transpose(1, 2).contiguous().view(B * H, L, Dh)
        V = V.view(B, L, H, Dh).transpose(1, 2).contiguous().view(B * H, L, Dh)

        if mask is not None:
            mask_expanded = mask.unsqueeze(1).expand(B, H, L).reshape(B * H, L, 1)
            mask_float = mask_expanded.float()
            Q = Q * mask_float
            K = K * mask_float
            V = V * mask_float

        Q_phi = self._feature_map(Q)
        K_phi = self._feature_map(K)

        KV = torch.einsum("bld,blh->bdh", K_phi, V)
        K_sum = K_phi.sum(dim=1)

        numerator = torch.einsum("bld,bdh->blh", Q_phi, KV)
        denom = torch.einsum("bld,bd->bl", Q_phi, K_sum).unsqueeze(-1)
        denom = denom + 1e-6

        attn_output = numerator / denom
        attn_output = self.attn_dropout(attn_output)

        attn_output = (
            attn_output.view(B, H, L, Dh).transpose(1, 2).contiguous().view(B, L, D)
        )

        out = self.out_proj(attn_output)
        out = self.dropout(out)
        return out


class LinearTransformerBlock(nn.Module):
    """Transformer block built from linearized self-attention and FFN."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = LinearAttentionLayer(embed_dim, num_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        x_norm = self.norm1(x)
        attn_out = self.attn(x_norm, mask=mask)
        x = x + self.dropout1(attn_out)

        x_norm2 = self.norm2(x)
        ffn_out = self.ffn(x_norm2)
        x = x + self.dropout2(ffn_out)
        return x
