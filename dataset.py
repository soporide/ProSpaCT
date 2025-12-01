
# dataset.py
from typing import List, Tuple, Dict
import math
import random

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DynamicCPSDataset(Dataset):
    """Slot-aggregated dynamic CPS dataset.

    Stores X_list, A_list, M_list, y_list across S slots and supports
    temporal train/val/test splits.
    """

    def __init__(
        self,
        X_list: List[Tensor],
        A_list: List[Tensor],
        M_list: List[Tensor],
        y_list: List[Tensor],
        train_frac: float = 0.6,
        val_frac: float = 0.2,
        split: str = "train",
    ) -> None:
        assert len(X_list) == len(A_list) == len(M_list) == len(y_list)
        self.X_list = X_list
        self.A_list = A_list
        self.M_list = M_list
        self.y_list = y_list
        self.num_slots = len(X_list)
        self.num_nodes = X_list[0].shape[0]
        assert 0 < train_frac < 1 and 0 < val_frac < 1 and train_frac + val_frac < 1
        self.train_frac = train_frac
        self.val_frac = val_frac

        # Temporal split indices
        train_end = int(self.num_slots * train_frac)
        val_end = int(self.num_slots * (train_frac + val_frac))
        self.train_indices = list(range(0, train_end))
        self.val_indices = list(range(train_end, val_end))
        self.test_indices = list(range(val_end, self.num_slots))

        assert split in {"train", "val", "test"}
        self.split = split

    def set_split(self, split: str) -> None:
        assert split in {"train", "val", "test"}
        self.split = split

    def _current_index_list(self) -> List[int]:
        if self.split == "train":
            return self.train_indices
        elif self.split == "val":
            return self.val_indices
        else:
            return self.test_indices

    def __len__(self) -> int:
        return len(self._current_index_list())

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        slot_indices = self._current_index_list()
        s = slot_indices[idx]
        return {
            "slot": torch.tensor(s, dtype=torch.long),
            "X": self.X_list[s],
            "A": self.A_list[s],
            "M": self.M_list[s],
            "y": self.y_list[s],
        }


def generate_toy_dynamic_cps(
    num_nodes: int = 80,
    num_slots: int = 30,
    num_patches: int = 4,
    feature_dim: int = 8,
    intra_prob: float = 0.25,
    inter_prob: float = 0.03,
    edge_turnover_prob: float = 0.02,
    mask_prob: float = 0.1,
    noise_std: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]:
    """Generate a toy dynamic CPS dataset.

    This generator creates a latent partition, within- and cross-patch
    connectivity, slot-wise edge turnover, irregular sensing masks, node
    features, and labels based on multi-hop exposure.
    """
    set_seed(seed)
    device = torch.device("cpu")

    # Latent true patches
    assert num_nodes >= num_patches
    base_patch_sizes = [num_nodes // num_patches] * num_patches
    remainder = num_nodes - sum(base_patch_sizes)
    for i in range(remainder):
        base_patch_sizes[i] += 1
    true_patches = torch.empty(num_nodes, dtype=torch.long)
    start = 0
    for k, size in enumerate(base_patch_sizes):
        true_patches[start : start + size] = k
        start += size

    # Precompute patch embeddings
    patch_emb = torch.randn(num_patches, feature_dim, device=device)

    # Initialize adjacency for first slot
    A_list: List[Tensor] = []
    M_list: List[Tensor] = []
    X_list: List[Tensor] = []
    y_list: List[Tensor] = []

    # Helper: generate adjacency from probabilities
    def gen_adj_from_partition() -> Tensor:
        A = torch.zeros(num_nodes, num_nodes, dtype=torch.float32, device=device)
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                same_patch = true_patches[i] == true_patches[j]
                p = intra_prob if same_patch else inter_prob
                if random.random() < p:
                    A[i, j] = 1.0
                    A[j, i] = 1.0
        # Self-loops not needed
        return A

    A_prev = gen_adj_from_partition()

    for s in range(num_slots):
        # Edge turnover: randomly flip a fraction of edges
        A = A_prev.clone()
        num_possible_edges = num_nodes * (num_nodes - 1) // 2
        num_flip = int(edge_turnover_prob * num_possible_edges)
        for _ in range(num_flip):
            i = random.randrange(num_nodes)
            j = random.randrange(num_nodes)
            if i == j:
                continue
            if A[i, j] > 0:
                A[i, j] = 0.0
                A[j, i] = 0.0
            else:
                # Respect patch structure somewhat
                same_patch = true_patches[i] == true_patches[j]
                p = intra_prob if same_patch else inter_prob
                if random.random() < p:
                    A[i, j] = 1.0
                    A[j, i] = 1.0
        A_prev = A

        # Presence mask
        M = torch.ones(num_nodes, num_nodes, dtype=torch.float32, device=device)
        mask_edges = torch.rand(num_nodes, num_nodes, device=device) < mask_prob
        M[mask_edges] = 0.0
        M = torch.triu(M, diagonal=1)
        M = M + M.t()
        M.fill_diagonal_(1.0)
        A_obs = A * M

        # Features: patch embedding + time drift + noise
        t = float(s) / max(1.0, float(num_slots - 1))
        time_vec = torch.full((num_nodes, feature_dim), t, dtype=torch.float32, device=device)
        X = patch_emb[true_patches] + 0.5 * time_vec + noise_std * torch.randn(
            num_nodes, feature_dim, device=device
        )

        # Severity based on 1-hop and 2-hop exposure
        deg = A_obs.sum(dim=1)
        A_sq = torch.matmul(A_obs, A_obs)
        two_hop = A_sq.sum(dim=1)
        exposure = deg + 0.5 * two_hop
        exposure_norm = (exposure - exposure.mean()) / (exposure.std() + 1e-6)

        logits = (X.mean(dim=1) + 0.7 * exposure_norm).clamp(-4.0, 4.0)
        probs = torch.sigmoid(logits)
        y = torch.bernoulli(probs).float()

        A_list.append(A_obs)
        M_list.append(M)
        X_list.append(X)
        y_list.append(y)

    return X_list, A_list, M_list, y_list
