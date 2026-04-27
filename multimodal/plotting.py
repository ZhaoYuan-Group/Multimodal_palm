from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.manifold import TSNE

from .config import CFG


def _load_plot_libs():
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _resolve_perplexity(num_samples: int) -> int:
    if num_samples <= 5:
        return 2
    return max(5, min(30, num_samples // 8))


def _subsample(features: np.ndarray, labels: np.ndarray, splits: Optional[np.ndarray], max_samples: int):
    if len(features) <= max_samples:
        return features, labels, splits
    rng = np.random.default_rng(CFG.seed)
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    half = max_samples // 2
    keep_pos = rng.choice(pos_idx, size=min(len(pos_idx), half), replace=False) if len(pos_idx) > 0 else np.array([], dtype=int)
    keep_neg = rng.choice(neg_idx, size=min(len(neg_idx), max_samples - len(keep_pos)), replace=False) if len(neg_idx) > 0 else np.array([], dtype=int)
    keep = np.concatenate([keep_pos, keep_neg])
    if len(keep) < max_samples:
        remaining = np.setdiff1d(np.arange(len(features)), keep)
        extra = rng.choice(remaining, size=min(len(remaining), max_samples - len(keep)), replace=False)
        keep = np.concatenate([keep, extra])
    keep = np.sort(keep)
    kept_splits = splits[keep] if splits is not None else None
    return features[keep], labels[keep], kept_splits


def pool_esm_features(seq_features: np.ndarray) -> np.ndarray:
    pooled = []
    for sample in seq_features:
        mask = ~np.all(np.isclose(sample, 0.0), axis=1)
        if mask.any():
            pooled.append(sample[mask].mean(axis=0))
        else:
            pooled.append(sample.mean(axis=0))
    return np.asarray(pooled, dtype=np.float32)


def extract_graph_embeddings(model, loader, device) -> np.ndarray:
    model.eval()
    embeddings = []
    with torch.no_grad():
        for _, graph_batch, _ in loader:
            graph_batch = graph_batch.to(device)
            graph_emb = model.graph_encoder(graph_batch)
            embeddings.append(graph_emb.float().cpu().numpy())
    if not embeddings:
        return np.zeros((0, CFG.graph_emb_dim), dtype=np.float32)
    return np.concatenate(embeddings, axis=0).astype(np.float32)


def extract_fusion_embeddings(model, loader, device) -> np.ndarray:
    model.eval()
    embeddings = []
    with torch.no_grad():
        for seq_x, graph_batch, _ in loader:
            seq_x = seq_x.to(device, non_blocking=True)
            graph_batch = graph_batch.to(device)
            outputs = model(seq_x, graph_batch)
            if len(outputs) > 3:
                fusion_emb = outputs[3]
            else:
                seq_emb, graph_emb = model.encode_modalities(seq_x, graph_batch)
                fusion_emb, _ = model.build_multimodal_embedding(seq_emb, graph_emb)
            embeddings.append(fusion_emb.float().cpu().numpy())
    if not embeddings:
        return np.zeros((0, CFG.fusion_hidden_dim * 2), dtype=np.float32)
    return np.concatenate(embeddings, axis=0).astype(np.float32)


def plot_tsne(features: np.ndarray, labels: np.ndarray, output_path: str, title: str, max_samples: int, split_names: Optional[np.ndarray] = None) -> Optional[str]:
    if len(features) < 3:
        return None
    plt = _load_plot_libs()
    features, labels, split_names = _subsample(features, labels, split_names, max_samples=max_samples)
    perplexity = _resolve_perplexity(len(features))
    coords = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=CFG.seed,
    ).fit_transform(features)

    plt.figure(figsize=(8, 6))
    colors = {0: "#2563eb", 1: "#dc2626"}
    label_names = {0: "negative", 1: "positive"}
    for label in [0, 1]:
        idx = labels == label
        if idx.any():
            plt.scatter(coords[idx, 0], coords[idx, 1], s=24, alpha=0.7, c=colors[label], label=label_names[label])
    plt.title(title)
    plt.xlabel("t-SNE-1")
    plt.ylabel("t-SNE-2")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def plot_roc_curves(curves: List[Dict], output_path: str, title: str) -> Optional[str]:
    valid_curves = [curve for curve in curves if len(curve["y_true"]) > 0 and len(np.unique(curve["y_true"])) > 1]
    if not valid_curves:
        return None
    plt = _load_plot_libs()
    from sklearn.metrics import roc_curve

    plt.figure(figsize=(7, 6))
    palette = {"train": "#2563eb", "val": "#f59e0b", "test": "#dc2626"}
    for curve in valid_curves:
        fpr, tpr, _ = roc_curve(curve["y_true"], curve["y_prob"])
        color = palette.get(curve["name"], None)
        plt.plot(fpr, tpr, linewidth=2, color=color, label=f"{curve['name']} AUC={curve['auc']:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, color="#6b7280")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def plot_modality_weight_kde(weights_by_split: Dict[str, np.ndarray], output_path: str, title: str) -> Optional[str]:
    if not any(arr.size > 0 for arr in weights_by_split.values()):
        return None
    plt = _load_plot_libs()
    from scipy.stats import gaussian_kde

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    palette = {"train": "#2563eb", "val": "#f59e0b", "test": "#dc2626"}
    x_grid = np.linspace(0.0, 1.0, 300)
    for idx, modality_name in enumerate(["sequence", "graph"]):
        ax = axes[idx]
        for split_name in ["train", "val", "test"]:
            weights = weights_by_split.get(split_name)
            if weights is None or weights.size == 0:
                continue
            values = weights[:, idx]
            if len(np.unique(values)) >= 2:
                kde = gaussian_kde(values)
                y = kde(x_grid)
                ax.plot(x_grid, y, linewidth=2, color=palette[split_name], label=split_name)
            else:
                ax.axvline(float(values[0]), linewidth=2, color=palette[split_name], label=split_name)
        ax.set_title(f"{modality_name.capitalize()} Weight KDE")
        ax.set_xlabel("Weight")
        ax.set_ylabel("Density")
        ax.set_xlim(0.0, 1.0)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_loss_curves(history: Dict[str, List[float]], output_path: str, title: str) -> Optional[str]:
    if not history.get("train_loss"):
        return None
    plt = _load_plot_libs()
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], linewidth=2, color="#2563eb", label="train_loss")
    if history.get("val_loss"):
        plt.plot(epochs, history["val_loss"], linewidth=2, color="#dc2626", label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def generate_training_plots(
    model,
    full_loader,
    seq_features: np.ndarray,
    labels: np.ndarray,
    train_metrics: Dict,
    val_metrics: Dict,
    test_metrics: Dict,
    history: Dict[str, List[float]],
    output_dir: str,
    roc_mode: str,
    tsne_max_samples: int,
    split_names: Optional[np.ndarray],
    split_aware_embeddings: bool,
) -> Dict[str, Optional[str]]:
    os_paths = {}
    split_names = None

    esm_pooled = pool_esm_features(seq_features)
    plot_errors = {}

    def _safe_plot(name: str, fn):
        try:
            os_paths[name] = fn()
        except Exception as exc:
            os_paths[name] = None
            plot_errors[name] = str(exc)
            print(f"[WARN] Failed to generate {name}: {exc}")

    _safe_plot(
        "esm_tsne",
        lambda: plot_tsne(
            esm_pooled,
            labels,
            output_path=f"{output_dir}/esm2_tsne.png",
            title="ESM2 Feature t-SNE",
            max_samples=tsne_max_samples,
            split_names=split_names,
        ),
    )

    graph_embeddings = extract_graph_embeddings(model, full_loader, CFG.device)
    _safe_plot(
        "gnn_tsne",
        lambda: plot_tsne(
            graph_embeddings,
            labels,
            output_path=f"{output_dir}/gnn_tsne.png",
            title="GNN Feature t-SNE",
            max_samples=tsne_max_samples,
            split_names=split_names,
        ),
    )

    fusion_embeddings = extract_fusion_embeddings(model, full_loader, CFG.device)
    _safe_plot(
        "fusion_tsne",
        lambda: plot_tsne(
            fusion_embeddings,
            labels,
            output_path=f"{output_dir}/fusion_tsne.png",
            title="Fusion Feature t-SNE",
            max_samples=tsne_max_samples,
            split_names=split_names,
        ),
    )

    roc_curves = [{"name": "test", "y_true": test_metrics["y_true"], "y_prob": test_metrics["y_prob"], "auc": test_metrics["auc"]}]
    roc_output = f"{output_dir}/roc_test.png"
    if roc_mode == "all":
        roc_curves = [
            {"name": "train", "y_true": train_metrics["y_true"], "y_prob": train_metrics["y_prob"], "auc": train_metrics["auc"]},
            {"name": "val", "y_true": val_metrics["y_true"], "y_prob": val_metrics["y_prob"], "auc": val_metrics["auc"]},
            {"name": "test", "y_true": test_metrics["y_true"], "y_prob": test_metrics["y_prob"], "auc": test_metrics["auc"]},
        ]
        roc_output = f"{output_dir}/roc_train_val_test.png"
    _safe_plot("roc", lambda: plot_roc_curves(roc_curves, output_path=roc_output, title="ROC Curves"))

    _safe_plot(
        "modality_weight_kde",
        lambda: plot_modality_weight_kde(
            {
                "train": train_metrics.get("modality_weights", np.zeros((0, 2), dtype=np.float32)),
                "val": val_metrics.get("modality_weights", np.zeros((0, 2), dtype=np.float32)),
                "test": test_metrics.get("modality_weights", np.zeros((0, 2), dtype=np.float32)),
            },
            output_path=f"{output_dir}/modality_weight_kde.png",
            title="Modality Weight Distribution",
        ),
    )

    _safe_plot("loss_curve", lambda: plot_loss_curves(history, output_path=f"{output_dir}/loss_curve.png", title="Training Loss"))
    if plot_errors:
        os_paths["plot_errors"] = plot_errors
    return os_paths
