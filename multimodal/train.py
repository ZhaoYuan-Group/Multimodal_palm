from contextlib import nullcontext
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, matthews_corrcoef
from sklearn.metrics import precision_score, recall_score
from torch.utils.data import DataLoader
from torch_geometric.data import Data

from .config import CFG
from .utils import compute_auc, find_optimal_threshold


class WarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer: torch.optim.Optimizer, total_steps: int, warmup_ratio: float = 0.1, last_epoch: int = -1):
        self.total_steps = max(int(total_steps), 1)
        self.warmup_steps = max(int(self.total_steps * warmup_ratio), 1)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            scale = float(self.last_epoch + 1) / float(self.warmup_steps)
            return [base_lr * scale for base_lr in self.base_lrs]
        cosine_step = self.last_epoch - self.warmup_steps
        cosine_steps = max(self.total_steps - self.warmup_steps, 1)
        scale = 0.5 * (1.0 + math.cos(math.pi * cosine_step / cosine_steps))
        return [base_lr * scale for base_lr in self.base_lrs]


class BinaryCenterInterLoss(nn.Module):
    def __init__(self, feat_dim: int, margin: float):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(2, feat_dim) * 0.02)
        self.margin = margin

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        labels_long = (labels > 0.5).long()
        center_loss = nn.functional.mse_loss(features, self.centers[labels_long], reduction="mean")
        center_dist = torch.norm(self.centers[0] - self.centers[1], p=2)
        inter_loss = torch.relu(torch.tensor(float(self.margin), dtype=center_dist.dtype, device=center_dist.device) - center_dist)
        return center_loss, inter_loss


def _get_autocast(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    if hasattr(torch, "autocast"):
        return torch.autocast(device_type=device.type, enabled=True)
    if device.type == "cuda" and hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"):
        return torch.cuda.amp.autocast(enabled=True)
    return nullcontext()


def _get_grad_scaler(enabled: bool):
    if not enabled:
        if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "GradScaler"):
            return torch.cuda.amp.GradScaler(enabled=False)
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            try:
                return torch.amp.GradScaler("cuda", enabled=False)
            except TypeError:
                return torch.amp.GradScaler(enabled=False)
        return None

    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=True)
        except TypeError:
            return torch.amp.GradScaler(enabled=True)

    if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "GradScaler"):
        return torch.cuda.amp.GradScaler(enabled=True)

    return None


def _tensor_is_finite(tensor: Optional[torch.Tensor]) -> bool:
    if tensor is None:
        return True
    return bool(torch.isfinite(tensor).all().item())


def evaluate_model(model: nn.Module, loader: DataLoader, criterion: Optional[nn.Module], device: torch.device, threshold: float = 0.5):
    model.eval()
    amp_enabled = bool(CFG.use_amp and device.type == "cuda")
    y_true = []
    y_prob = []
    total_loss = 0.0
    total_alpha = []
    total_samples = 0
    skipped_batches = 0

    with torch.no_grad():
        for seq_x, graph_batch, labels in loader:
            seq_x = seq_x.to(device, non_blocking=True)
            graph_batch = graph_batch.to(device)
            labels = labels.to(device, non_blocking=True)
            if not _tensor_is_finite(seq_x) or not _tensor_is_finite(graph_batch.x) or not _tensor_is_finite(graph_batch.edge_attr) or not _tensor_is_finite(labels):
                skipped_batches += 1
                continue
            with _get_autocast(device, amp_enabled):
                outputs = model(seq_x, graph_batch)
                logits, modality_weights = outputs[0], outputs[1]
                if not _tensor_is_finite(logits):
                    skipped_batches += 1
                    continue
                if criterion is not None:
                    loss = criterion(logits, labels)
                    if not _tensor_is_finite(loss):
                        skipped_batches += 1
                        continue
                    total_loss += loss.item() * seq_x.size(0)
            y_prob.extend(torch.sigmoid(logits).float().cpu().numpy().tolist())
            y_true.extend(labels.cpu().numpy().tolist())
            total_alpha.append(modality_weights.float().cpu().numpy())
            total_samples += seq_x.size(0)

    if skipped_batches > 0:
        print(f"[WARN] Skipped {skipped_batches} non-finite eval batches.")

    y_true_np = np.array(y_true)
    y_prob_np = np.array(y_prob)
    y_pred_np = (y_prob_np > threshold).astype(int)
    cm = confusion_matrix(y_true_np, y_pred_np) if len(y_true_np) > 0 else np.array([[0]])
    if cm.size == 1:
        tn = cm[0, 0] if len(y_pred_np) > 0 and y_pred_np[0] == 0 else 0
        fp = fn = tp = 0
    else:
        tn, fp, fn, tp = cm.ravel()

    avg_weights = np.concatenate(total_alpha, axis=0).mean(axis=0).tolist() if total_alpha else [0.0, 0.0]
    all_modality_weights = np.concatenate(total_alpha, axis=0) if total_alpha else np.zeros((0, 2), dtype=np.float32)
    return {
        "auc": compute_auc(y_true_np, y_prob_np) if len(np.unique(y_true_np)) > 1 else 0.0,
        "acc": accuracy_score(y_true_np, y_pred_np),
        "f1": f1_score(y_true_np, y_pred_np, zero_division=0),
        "precision": precision_score(y_true_np, y_pred_np, zero_division=0),
        "recall": recall_score(y_true_np, y_pred_np, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "mcc": matthews_corrcoef(y_true_np, y_pred_np) if len(np.unique(y_pred_np)) > 1 and len(np.unique(y_true_np)) > 1 else 0.0,
        "loss": total_loss / total_samples if total_samples > 0 else 0.0,
        "y_true": y_true_np,
        "y_prob": y_prob_np,
        "avg_modality_weights": avg_weights,
        "modality_weights": all_modality_weights,
    }


def _build_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    sequence_params = list(model.sequence_encoder.parameters()) if hasattr(model, "sequence_encoder") else []
    graph_params = []
    if hasattr(model, "graph_encoder"):
        graph_params.extend(model.graph_encoder.parameters())
    if hasattr(model, "graph_classifier"):
        graph_params.extend(model.graph_classifier.parameters())

    grouped_param_ids = {id(param) for param in sequence_params + graph_params}
    head_params = [param for param in model.parameters() if id(param) not in grouped_param_ids]
    param_groups = []
    if sequence_params:
        param_groups.append({"params": sequence_params, "lr": CFG.lr * CFG.sequence_lr_scale, "name": "sequence"})
    if graph_params:
        param_groups.append({"params": graph_params, "lr": CFG.lr * CFG.graph_lr_scale, "name": "graph"})
    if head_params:
        param_groups.append({"params": head_params, "lr": CFG.lr * CFG.head_lr_scale, "name": "head"})
    return torch.optim.AdamW(param_groups, lr=CFG.lr, weight_decay=CFG.weight_decay)


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader, criterion: nn.Module, save_path: str):
    optimizer = _build_optimizer(model)
    center_regularizer = None
    if CFG.center_loss_weight > 0 or CFG.inter_loss_weight > 0:
        center_regularizer = BinaryCenterInterLoss(feat_dim=CFG.fusion_hidden_dim * 2, margin=CFG.inter_loss_margin).to(CFG.device)
        optimizer.add_param_group({"params": center_regularizer.parameters(), "lr": CFG.lr * CFG.head_lr_scale, "name": "center"})
    if CFG.scheduler_type == "warmup_cosine":
        scheduler = WarmupCosineAnnealingLR(
            optimizer,
            total_steps=max(CFG.epochs * len(train_loader), 1),
            warmup_ratio=CFG.warmup_ratio,
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=CFG.scheduler_patience)
    amp_enabled_global = bool(CFG.use_amp and CFG.device.type == "cuda")
    scaler = _get_grad_scaler(amp_enabled_global)
    best_state = None
    best_score = -np.inf
    epochs_no_improve = 0
    history = {"train_loss": [], "train_auc": [], "val_loss": [], "val_auc": [], "lr": []}

    for epoch in range(1, CFG.epochs + 1):
        model.train()
        train_probs = []
        train_targets = []
        losses = []
        amp_enabled = amp_enabled_global
        skipped_batches = 0

        for seq_x, graph_batch, labels in train_loader:
            seq_x = seq_x.to(CFG.device, non_blocking=True)
            graph_batch = graph_batch.to(CFG.device)
            labels = labels.to(CFG.device, non_blocking=True)
            if not _tensor_is_finite(seq_x) or not _tensor_is_finite(graph_batch.x) or not _tensor_is_finite(graph_batch.edge_attr) or not _tensor_is_finite(labels):
                skipped_batches += 1
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.zero_grad(set_to_none=True)
            with _get_autocast(CFG.device, amp_enabled):
                outputs = model(seq_x, graph_batch)
                logits, modality_weights = outputs[0], outputs[1]
                graph_logits = outputs[2] if len(outputs) > 2 else None
                final_emb = outputs[3] if len(outputs) > 3 else None
                structure_mask = outputs[4] if len(outputs) > 4 else None
                labels_smooth = labels * (1.0 - CFG.label_smooth_eps) + 0.5 * CFG.label_smooth_eps
                loss = criterion(logits, labels_smooth)
                if graph_logits is not None and CFG.graph_aux_loss_weight > 0:
                    if structure_mask is not None:
                        valid_structure = structure_mask.view(-1).to(graph_logits.device) > 0.5
                        if valid_structure.any():
                            loss = loss + CFG.graph_aux_loss_weight * criterion(graph_logits[valid_structure], labels_smooth[valid_structure])
                    else:
                        loss = loss + CFG.graph_aux_loss_weight * criterion(graph_logits, labels_smooth)
                if center_regularizer is not None and final_emb is not None:
                    center_loss, inter_loss = center_regularizer(final_emb, labels)
                    loss = loss + CFG.center_loss_weight * center_loss + CFG.inter_loss_weight * inter_loss
                if CFG.modality_balance_lambda > 0:
                    graph_weight = modality_weights[:, 1].mean()
                    target = torch.tensor(float(CFG.modality_graph_target_weight), dtype=graph_weight.dtype, device=graph_weight.device)
                    loss = loss + CFG.modality_balance_lambda * (graph_weight - target).pow(2)
            if not _tensor_is_finite(logits) or not _tensor_is_finite(loss):
                skipped_batches += 1
                optimizer.zero_grad(set_to_none=True)
                continue

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.clip_norm)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            if CFG.scheduler_type == "warmup_cosine":
                scheduler.step()

            losses.append(loss.item())
            train_probs.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
            train_targets.extend(labels.detach().cpu().numpy().tolist())

        if skipped_batches > 0:
            print(f"[WARN] Epoch {epoch}: skipped {skipped_batches} non-finite train batches.")

        train_auc = compute_auc(np.array(train_targets), np.array(train_probs)) if train_probs else 0.5
        val_metrics = evaluate_model(model, val_loader, criterion=criterion, device=CFG.device)

        history["train_loss"].append(float(np.mean(losses)) if losses else 0.0)
        history["train_auc"].append(train_auc)
        history["val_loss"].append(val_metrics["loss"])
        history["val_auc"].append(val_metrics["auc"])
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))
        if CFG.scheduler_type == "plateau":
            scheduler.step(val_metrics["auc"])

        print(
            f"Epoch {epoch}/{CFG.epochs} | "
            f"train_loss={history['train_loss'][-1]:.4f} | "
            f"train_auc={train_auc:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_auc={val_metrics['auc']:.4f} | "
            f"lr={history['lr'][-1]:.2e}"
        )

        current_score = float(val_metrics.get(CFG.best_metric, val_metrics["auc"]))
        if current_score > best_score + 1e-4:
            best_score = current_score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save(best_state, save_path)
            epochs_no_improve = 0
            print(f"[INFO] Saved new best model to {save_path} | best_{CFG.best_metric}={best_score:.4f}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= CFG.patience:
            print(f"[INFO] Early stopping triggered at epoch {epoch}: val_{CFG.best_metric} did not improve for {CFG.patience} epochs.")
            break

    if best_state is None:
        raise RuntimeError("Training finished without producing a valid best checkpoint.")

    model.load_state_dict(best_state)
    val_best = evaluate_model(model, val_loader, criterion=criterion, device=CFG.device)
    threshold = find_optimal_threshold(val_best["y_true"], val_best["y_prob"])
    test_metrics = evaluate_model(model, test_loader, criterion=criterion, device=CFG.device, threshold=threshold)
    test_metrics["optimal_threshold_from_val"] = threshold
    return history, val_best, test_metrics


def _site_count_stats(groups: Sequence[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for group in groups:
        counts[str(group)] = counts.get(str(group), 0) + 1
    return {
        "one_site_proteins": int(sum(1 for count in counts.values() if count == 1)),
        "two_site_proteins": int(sum(1 for count in counts.values() if count == 2)),
        "three_plus_site_proteins": int(sum(1 for count in counts.values() if count >= 3)),
    }


def split_indices(
    labels: np.ndarray,
    groups: Sequence[str],
    seed: int = CFG.seed,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices_by_group: Dict[str, List[int]] = {}
    for idx, group in enumerate(groups):
        indices_by_group.setdefault(str(group), []).append(idx)

    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []
    split_probs = np.asarray([train_ratio, val_ratio, test_ratio], dtype=np.float64)
    split_probs = split_probs / split_probs.sum()

    for group_indices in indices_by_group.values():
        shuffled = np.asarray(group_indices, dtype=int)
        rng.shuffle(shuffled)
        num_sites = len(shuffled)

        if num_sites == 1:
            target = rng.choice(3, p=split_probs)
            if target == 0:
                train_idx.append(int(shuffled[0]))
            elif target == 1:
                val_idx.append(int(shuffled[0]))
            else:
                test_idx.append(int(shuffled[0]))
            continue

        if num_sites == 2:
            train_idx.append(int(shuffled[0]))
            test_idx.append(int(shuffled[1]))
            continue

        train_idx.append(int(shuffled[0]))
        val_idx.append(int(shuffled[1]))
        test_idx.append(int(shuffled[2]))
        for idx in shuffled[3:]:
            target = rng.choice(3, p=split_probs)
            if target == 0:
                train_idx.append(int(idx))
            elif target == 1:
                val_idx.append(int(idx))
            else:
                test_idx.append(int(idx))

    return np.sort(np.asarray(train_idx, dtype=int)), np.sort(np.asarray(val_idx, dtype=int)), np.sort(np.asarray(test_idx, dtype=int))


def build_cv_splits(
    labels: np.ndarray,
    groups: Sequence[str],
    num_folds: int,
    seed: int = CFG.seed,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if num_folds < 1:
        raise ValueError("num_folds must be at least 1.")
    return [
        split_indices(labels, groups, seed=seed + fold_idx, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)
        for fold_idx in range(num_folds)
    ]


def summarize_split(indices: np.ndarray, labels: np.ndarray, groups: Sequence[str], sample_keys: Sequence[str]) -> dict:
    if len(indices) == 0:
        return {"samples": 0, "positives": 0, "negatives": 0, "unique_proteins": 0, "duplicate_sample_keys": 0}
    idx = np.asarray(indices, dtype=int)
    split_labels = labels[idx]
    split_groups = np.asarray(groups, dtype=object)[idx]
    split_keys = np.asarray(sample_keys, dtype=object)[idx]
    return {
        "samples": int(len(idx)),
        "positives": int((split_labels == 1).sum()),
        "negatives": int((split_labels == 0).sum()),
        "positive_ratio": float((split_labels == 1).mean()),
        "unique_proteins": int(len(set(split_groups.tolist()))),
        "duplicate_sample_keys": int(len(split_keys) - len(set(split_keys.tolist()))),
    }


def split_overlap_report(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    groups: Sequence[str],
    sample_keys: Sequence[str],
) -> dict:
    groups_arr = np.asarray(groups, dtype=object)
    keys_arr = np.asarray(sample_keys, dtype=object)

    def _set(values: np.ndarray) -> set:
        return set(values.tolist())

    train_groups, val_groups, test_groups = _set(groups_arr[train_idx]), _set(groups_arr[val_idx]), _set(groups_arr[test_idx])
    train_keys, val_keys, test_keys = _set(keys_arr[train_idx]), _set(keys_arr[val_idx]), _set(keys_arr[test_idx])
    return {
        "protein_overlap": {
            "train_val": int(len(train_groups & val_groups)),
            "train_test": int(len(train_groups & test_groups)),
            "val_test": int(len(val_groups & test_groups)),
        },
        "sample_key_overlap": {
            "train_val": int(len(train_keys & val_keys)),
            "train_test": int(len(train_keys & test_keys)),
            "val_test": int(len(val_keys & test_keys)),
        },
    }


def subset_graphs(graphs: List[Data], indices: np.ndarray) -> List[Data]:
    return [graphs[int(idx)] for idx in indices]
