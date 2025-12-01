
# partitioner.py
from typing import List, Tuple

import torch
from torch import Tensor


def patch_sizes(patch_ids: Tensor, num_patches: int) -> Tensor:
    """Compute sizes of each patch for a 1D patch_id tensor."""
    sizes = torch.zeros(num_patches, dtype=torch.float32, device=patch_ids.device)
    for k in range(num_patches):
        sizes[k] = (patch_ids == k).sum()
    return sizes


def cut_cost(A: Tensor, patch_ids: Tensor, num_patches: int) -> Tensor:
    """Boundary cut cost: total edge weight crossing patch boundaries."""
    # A: (N, N)
    N = A.shape[0]
    patch_mat_i = patch_ids.view(N, 1).expand(N, N)
    patch_mat_j = patch_ids.view(1, N).expand(N, N)
    boundary_mask = (patch_mat_i != patch_mat_j).float()
    return (A * boundary_mask).sum()


def size_dispersion_cost(patch_ids: Tensor, num_patches: int) -> Tensor:
    """Size dispersion penalty based on variance from balanced partition."""
    N = patch_ids.shape[0]
    sizes = patch_sizes(patch_ids, num_patches)
    target = float(N) / float(num_patches)
    return ((sizes - target) ** 2).mean()


def vi_cost(patch_ids1: Tensor, patch_ids2: Tensor, num_patches: int) -> Tensor:
    """Variation of Information between two partitions (approximate)."""
    device = patch_ids1.device
    N = patch_ids1.shape[0]
    # Compute contingency table
    counts = torch.zeros(num_patches, num_patches, dtype=torch.float32, device=device)
    for i in range(N):
        a = int(patch_ids1[i].item())
        b = int(patch_ids2[i].item())
        if 0 <= a < num_patches and 0 <= b < num_patches:
            counts[a, b] += 1.0
    if counts.sum() == 0:
        return torch.tensor(0.0, device=device)

    p = counts / counts.sum()
    p1 = p.sum(dim=1, keepdim=True)
    p2 = p.sum(dim=0, keepdim=True)

    # Entropies and mutual information
    eps = 1e-10
    H1 = -torch.where(p1 > 0, p1 * torch.log(p1 + eps), torch.zeros_like(p1)).sum()
    H2 = -torch.where(p2 > 0, p2 * torch.log(p2 + eps), torch.zeros_like(p2)).sum()
    MI = torch.where(
        p > 0,
        p * torch.log((p + eps) / (p1 @ p2 + eps)),
        torch.zeros_like(p),
    ).sum()
    VI = H1 + H2 - 2.0 * MI
    return VI


def kmeans_init(X: Tensor, num_patches: int, num_iters: int = 20) -> Tensor:
    """Simple k-means clustering on node features for initialization."""
    N, _ = X.shape
    device = X.device
    if N < num_patches:
        raise ValueError("Number of nodes must be >= num_patches for k-means init.")
    indices = torch.randperm(N, device=device)[:num_patches]
    centers = X[indices].clone()
    patch_ids = torch.zeros(N, dtype=torch.long, device=device)

    for _ in range(num_iters):
        dists = torch.cdist(X, centers, p=2.0)  # (N, num_patches)
        patch_ids = torch.argmin(dists, dim=1)
        for k in range(num_patches):
            mask = patch_ids == k
            if mask.any():
                centers[k] = X[mask].mean(dim=0)
    return patch_ids


class MesoscalePartitioner:
    """Adaptive mesoscale partitioner with cut, size, and VI penalties."""

    def __init__(
        self,
        adjacency_list: List[Tensor],
        feature_list: List[Tensor],
        num_patches: int,
        cut_weight: float = 1.0,
        size_weight: float = 0.1,
        vi_weight: float = 0.1,
        balance_tol: float = 0.5,
    ) -> None:
        assert len(adjacency_list) == len(feature_list)
        self.adjacency_list = [A.clone().detach() for A in adjacency_list]
        self.feature_list = [X.clone().detach() for X in feature_list]
        self.num_slots = len(self.adjacency_list)
        self.num_nodes = self.adjacency_list[0].shape[0]
        self.num_patches = num_patches
        self.cut_weight = cut_weight
        self.size_weight = size_weight
        self.vi_weight = vi_weight
        self.balance_tol = balance_tol
        self.partition_ids: List[Tensor] = []

    def initialize_partitions(self) -> None:
        """Initialize slot-wise partitions using k-means on the first slot."""
        X0 = self.feature_list[0]
        base_patch_ids = kmeans_init(X0, self.num_patches, num_iters=25)
        self.partition_ids = []
        for _ in range(self.num_slots):
            self.partition_ids.append(base_patch_ids.clone())

    def structural_loss_slot(self, slot: int) -> Tensor:
        A = self.adjacency_list[slot]
        patch_ids = self.partition_ids[slot]
        loss_cut = cut_cost(A, patch_ids, self.num_patches)
        loss_size = size_dispersion_cost(patch_ids, self.num_patches)
        if slot > 0:
            loss_vi = vi_cost(
                patch_ids,
                self.partition_ids[slot - 1],
                self.num_patches,
            )
        else:
            loss_vi = torch.tensor(0.0, device=A.device)
        return (
            self.cut_weight * loss_cut
            + self.size_weight * loss_size
            + self.vi_weight * loss_vi
        )

    def total_structural_loss(self) -> Tensor:
        losses = []
        for s in range(self.num_slots):
            losses.append(self.structural_loss_slot(s))
        return torch.stack(losses).sum()

    def _is_move_balanced(
        self,
        sizes: Tensor,
        from_patch: int,
        to_patch: int,
    ) -> bool:
        """Check if moving one node keeps patch sizes within tolerance band."""
        N = sizes.sum().item()
        target = N / float(self.num_patches)
        min_size = (1.0 - self.balance_tol) * target
        max_size = (1.0 + self.balance_tol) * target

        sizes_new = sizes.clone()
        sizes_new[from_patch] -= 1.0
        sizes_new[to_patch] += 1.0
        if sizes_new[from_patch] < 1.0:
            return False
        if torch.any(sizes_new < min_size) or torch.any(sizes_new > max_size):
            return False
        return True

    def _boundary_nodes(self, A: Tensor, patch_ids: Tensor) -> Tensor:
        """Return indices of nodes lying on patch boundaries."""
        N = A.shape[0]
        device = A.device
        boundary = torch.zeros(N, dtype=torch.bool, device=device)
        for i in range(N):
            neighbors = (A[i] > 0).nonzero(as_tuple=False).view(-1)
            if neighbors.numel() == 0:
                continue
            if torch.any(patch_ids[neighbors] != patch_ids[i]):
                boundary[i] = True
        boundary_idx = boundary.nonzero(as_tuple=False).view(-1)
        return boundary_idx

    def update_slot(
        self,
        slot: int,
        max_boundary_passes: int = 3,
        min_relative_improvement: float = 1e-3,
    ) -> None:
        """Localized-Adjust boundary update for a single slot."""
        A = self.adjacency_list[slot]
        device = A.device
        patch_ids = self.partition_ids[slot].clone()
        sizes = patch_sizes(patch_ids, self.num_patches)

        base_loss = self.structural_loss_slot(slot).detach()
        if base_loss <= 0:
            return

        for _ in range(max_boundary_passes):
            boundary_nodes = self._boundary_nodes(A, patch_ids)
            if boundary_nodes.numel() == 0:
                break
            improved = False
            last_loss_reduction = torch.tensor(0.0, device=device)
            for i in boundary_nodes.tolist():
                current_patch = int(patch_ids[i].item())
                neighbors = (A[i] > 0).nonzero(as_tuple=False).view(-1)
                if neighbors.numel() == 0:
                    continue
                neighbor_patches = torch.unique(patch_ids[neighbors])
                for p in neighbor_patches.tolist():
                    if p == current_patch:
                        continue
                    if not self._is_move_balanced(sizes, current_patch, int(p)):
                        continue
                    old_patch_ids = patch_ids.clone()
                    patch_ids[i] = int(p)
                    self.partition_ids[slot] = patch_ids
                    new_loss = self.structural_loss_slot(slot).detach()
                    loss_reduction = base_loss - new_loss
                    if loss_reduction > 0:
                        base_loss = new_loss
                        sizes[current_patch] -= 1.0
                        sizes[int(p)] += 1.0
                        improved = True
                        last_loss_reduction = loss_reduction
                    else:
                        patch_ids = old_patch_ids
                        self.partition_ids[slot] = patch_ids
            if not improved:
                break
            relative_improvement = (
                last_loss_reduction / (base_loss.abs() + 1e-8)
            ).abs()
            if relative_improvement < min_relative_improvement:
                break
        self.partition_ids[slot] = patch_ids

    def update_all_slots(
        self,
        max_boundary_passes: int = 3,
        min_relative_improvement: float = 1e-3,
    ) -> None:
        """Run Localized-Adjust on all slots in sequence."""
        if not self.partition_ids:
            self.initialize_partitions()
        for s in range(self.num_slots):
            self.update_slot(
                slot=s,
                max_boundary_passes=max_boundary_passes,
                min_relative_improvement=min_relative_improvement,
            )

    def get_partition_ids(self) -> List[Tensor]:
        return [p.clone() for p in self.partition_ids]
