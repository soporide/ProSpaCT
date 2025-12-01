
# train_prospact_toy.py
import os
from typing import Tuple

import torch
from torch import Tensor
from torch.optim import Adam

from dataset import (
    DynamicCPSDataset,
    generate_toy_dynamic_cps,
    set_seed,
)
from partitioner import MesoscalePartitioner
from beta_posterior import PatchBetaPosterior
from model import ProSpaCTModel
from losses_metrics import (
    compute_predictive_loss,
    compute_alignment_loss,
    compute_l2_reg,
    compute_nll,
    compute_brier_score,
    compute_ece,
)


def prepare_sequence_tensors(
    X_list,
    y_list,
    patch_ids_list,
    dataset: DynamicCPSDataset,
    device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    num_slots = len(X_list)
    num_nodes = X_list[0].shape[0]

    X = torch.stack(X_list, dim=0).to(device)
    y = torch.stack(y_list, dim=0).to(device)
    patch_ids = torch.stack(patch_ids_list, dim=0).to(device)

    train_mask = torch.zeros(num_slots, num_nodes, dtype=torch.bool, device=device)
    val_mask = torch.zeros_like(train_mask)
    test_mask = torch.zeros_like(train_mask)

    for s in dataset.train_indices:
        train_mask[s] = True
    for s in dataset.val_indices:
        val_mask[s] = True
    for s in dataset.test_indices:
        test_mask[s] = True

    temporal_mask = torch.ones_like(train_mask)

    return X, y, patch_ids, train_mask, val_mask, test_mask, temporal_mask


def train_prospact_toy() -> None:
    set_seed(123)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_nodes = 80
    num_slots = 24
    num_patches = 4
    feature_dim = 8

    print("Generating toy dynamic CPS data...")
    X_list, A_list, M_list, y_list = generate_toy_dynamic_cps(
        num_nodes=num_nodes,
        num_slots=num_slots,
        num_patches=num_patches,
        feature_dim=feature_dim,
        seed=123,
    )

    dataset = DynamicCPSDataset(
        X_list=X_list,
        A_list=A_list,
        M_list=M_list,
        y_list=y_list,
        train_frac=0.6,
        val_frac=0.2,
        split="train",
    )

    print("Initializing mesoscale partitioner...")
    partitioner = MesoscalePartitioner(
        adjacency_list=A_list,
        feature_list=X_list,
        num_patches=num_patches,
        cut_weight=1.0,
        size_weight=0.1,
        vi_weight=0.1,
        balance_tol=0.5,
    )
    partitioner.initialize_partitions()

    beta_posterior = PatchBetaPosterior(
        num_patches=num_patches,
        alpha0=1.0,
        beta0=1.0,
        forgetting=0.95,
        device=device,
    )

    model = ProSpaCTModel(
        input_dim=feature_dim,
        num_patches=num_patches,
        num_slots=num_slots,
        d_model_spatial=32,
        d_model_temporal=64,
        num_heads_spatial=4,
        num_heads_cluster=4,
        num_heads_temporal=4,
        num_pformer_layers=2,
        num_cformer_layers=1,
        num_tformer_layers=2,
        time_dim=8,
        dropout=0.1,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=0.0)

    lambda_P = 0.01
    lambda_align = 0.5
    lambda_reg = 1e-5
    pos_weight = 2.0
    neg_weight = 1.0
    num_outer_cycles = 3
    num_param_updates_per_cycle = 15
    grad_clip = 1.0

    print("Starting alternating optimization...")
    for cycle in range(num_outer_cycles):
        print(f"\n=== Outer cycle {cycle + 1}/{num_outer_cycles} ===")

        print("Partition step: updating mesoscale partitions...")
        partitioner.update_all_slots(max_boundary_passes=2, min_relative_improvement=1e-3)
        partition_loss_value = partitioner.total_structural_loss()
        print(f"Partition structural loss L_P: {partition_loss_value.item():.4f}")

        partition_ids_list = partitioner.get_partition_ids()

        print("Refreshing Beta posteriors with exponential forgetting...")
        beta_posterior.reset()
        beta_posterior.update_from_labels(
            y_list=y_list,
            patch_ids_list=partition_ids_list,
            slot_indices=dataset.train_indices,
        )

        (
            X_all,
            y_all,
            patch_ids_all,
            train_mask,
            val_mask,
            test_mask,
            temporal_mask,
        ) = prepare_sequence_tensors(
            X_list, y_list, partition_ids_list, dataset, device
        )

        model.train()
        for update_idx in range(num_param_updates_per_cycle):
            optimizer.zero_grad()
            probs, logits = model(
                X=X_all,
                patch_ids=patch_ids_all,
                beta_posterior=beta_posterior,
                temporal_mask=temporal_mask,
            )
            loss_pred = compute_predictive_loss(
                probs=probs,
                y=y_all,
                mask=train_mask,
                pos_weight=pos_weight,
                neg_weight=neg_weight,
            )
            patch_baselines = beta_posterior.get_baseline_rates()
            loss_align = compute_alignment_loss(
                probs=probs,
                patch_ids=patch_ids_all,
                patch_baselines=patch_baselines,
                mask=train_mask,
            )
            loss_reg = compute_l2_reg(model)

            loss_total = (
                loss_pred
                + lambda_align * loss_align
                + lambda_reg * loss_reg
                + lambda_P * partition_loss_value.to(device)
            )
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            if (update_idx + 1) % 5 == 0:
                print(
                    f"  Param update {update_idx + 1}/{num_param_updates_per_cycle} "
                    f"Loss_total={loss_total.item():.4f} "
                    f"L_pred={loss_pred.item():.4f} "
                    f"L_align={loss_align.item():.4f}"
                )

        print("Calibration step: fitting temperature on validation window...")
        model.eval()
        temp_param = [model.prob_head.log_temperature]
        optimizer_temp = Adam(temp_param, lr=5e-3)
        for _ in range(30):
            optimizer_temp.zero_grad()
            probs_val, _ = model(
                X=X_all,
                patch_ids=patch_ids_all,
                beta_posterior=beta_posterior,
                temporal_mask=temporal_mask,
            )
            loss_val_nll = compute_nll(
                probs=probs_val,
                y=y_all,
                mask=val_mask,
            )
            loss_val_nll.backward()
            optimizer_temp.step()

        temperature_value = torch.nn.functional.softplus(
            model.prob_head.log_temperature
        ).item()
        print(f"Calibrated global temperature tau: {temperature_value:.4f}")

        with torch.no_grad():
            model.eval()
            probs_all, _ = model(
                X=X_all,
                patch_ids=patch_ids_all,
                beta_posterior=beta_posterior,
                temporal_mask=temporal_mask,
            )

            train_nll = compute_nll(probs_all, y_all, mask=train_mask).item()
            val_nll = compute_nll(probs_all, y_all, mask=val_mask).item()
            train_brier = compute_brier_score(probs_all, y_all, mask=train_mask).item()
            val_brier = compute_brier_score(probs_all, y_all, mask=val_mask).item()
            train_ece = compute_ece(probs_all, y_all, mask=train_mask).item()
            val_ece = compute_ece(probs_all, y_all, mask=val_mask).item()

        print(
            f"After cycle {cycle + 1}: "
            f"Train NLL={train_nll:.4f}, Brier={train_brier:.4f}, ECE={train_ece:.4f}; "
            f"Val NLL={val_nll:.4f}, Brier={val_brier:.4f}, ECE={val_ece:.4f}"
        )

    print("\n=== Final evaluation on test window ===")
    with torch.no_grad():
        model.eval()
        probs_all, _ = model(
            X=X_all,
            patch_ids=patch_ids_all,
            beta_posterior=beta_posterior,
            temporal_mask=temporal_mask,
        )
        test_nll = compute_nll(probs_all, y_all, mask=test_mask).item()
        test_brier = compute_brier_score(probs_all, y_all, mask=test_mask).item()
        test_ece = compute_ece(probs_all, y_all, mask=test_mask).item()

    print(
        f"Test NLL={test_nll:.4f}, Brier={test_brier:.4f}, "
        f"ECE={test_ece:.4f}"
    )

    print("\nExample predictive intervals for last slot:")
    with torch.no_grad():
        mean, lower, upper = model.predict_with_intervals(
            X=X_all,
            patch_ids=patch_ids_all,
            beta_posterior=beta_posterior,
            temporal_mask=temporal_mask,
            num_samples=30,
            interval_alpha=0.1,
        )

    last_slot = num_slots - 1
    y_last = y_all[last_slot]
    mean_last = mean[last_slot]
    lower_last = lower[last_slot]
    upper_last = upper[last_slot]

    for node_id in range(5):
        print(
            f"Node {node_id}: y={int(y_last[node_id].item())} "
            f"mean={mean_last[node_id].item():.3f}, "
            f"90% PI=({lower_last[node_id].item():.3f}, "
            f"{upper_last[node_id].item():.3f})"
        )


if __name__ == "__main__":
    train_prospact_toy()
