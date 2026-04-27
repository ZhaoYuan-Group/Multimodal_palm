import argparse
import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from .config import CFG
from .data import (
    MultiModalDataset,
    ProteinGraphBuilder,
    align_samples_with_graphs,
    collect_graphs,
    collate_multimodal,
    extract_esm2_features,
    load_esm2_model,
    read_sequence_dataframe,
    summarize_graph_collection,
)
from .model import FocalLoss, MultiModalPredictor
from .plotting import generate_training_plots
from .train import (
    build_cv_splits,
    evaluate_model,
    split_indices,
    split_overlap_report,
    subset_graphs,
    summarize_split,
    train_model,
)
from .utils import collect_dir_fingerprints, file_fingerprint, save_metrics, set_seed, stable_hash


def create_loader(dataset, batch_size: int, shuffle: bool = False, sampler=None) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=CFG.num_workers,
        pin_memory=bool(CFG.device.type == "cuda"),
        collate_fn=collate_multimodal,
    )


def build_graph_cache_path(cache_dir: str, pos_graph_dir: str, neg_graph_dir: str, pdb_dir: str) -> str:
    signature = {
        "graph_cache_version": CFG.graph_cache_version,
        "graph_dist_threshold": CFG.graph_dist_threshold,
        "graph_node_dim": CFG.graph_node_dim,
        "graph_edge_dim": CFG.graph_edge_dim,
        "graph_use_geometry": CFG.graph_use_geometry,
        "use_plddt_structure_mask": CFG.use_plddt_structure_mask,
        "structure_plddt_threshold": CFG.structure_plddt_threshold,
        "normalize_graph_node_features": CFG.normalize_graph_node_features,
        "normalize_graph_edge_features": CFG.normalize_graph_edge_features,
        "edge_seq_dist_log1p": CFG.edge_seq_dist_log1p,
        "pos_graph_files": collect_dir_fingerprints(pos_graph_dir, ".csv"),
        "neg_graph_files": collect_dir_fingerprints(neg_graph_dir, ".csv"),
        "pdb_files": collect_dir_fingerprints(pdb_dir, ".pdb") + collect_dir_fingerprints(pdb_dir, ".PDB"),
    }
    return os.path.join(cache_dir, f"graphs_{stable_hash(signature)}.pt")


def build_esm_cache_path(cache_dir: str, aligned_df, pos_csv: str, neg_csv: str) -> str:
    sample_records = aligned_df[["protein_id", "site_norm", "chain_norm", "sequence", "label"]].fillna("").to_dict("records")
    signature = {
        "esm_cache_version": CFG.esm_cache_version,
        "esm_model": CFG.esm_model,
        "max_len": CFG.max_len,
        "pos_csv": file_fingerprint(pos_csv),
        "neg_csv": file_fingerprint(neg_csv),
        "samples": sample_records,
    }
    return os.path.join(cache_dir, f"esm_{stable_hash(signature)}.npz")


def _graph_report_stats(graph_report: dict):
    edge_stats_undirected = graph_report.get("edge_count_stats_undirected", graph_report.get("edge_count_stats", {}))
    density_stats_undirected = graph_report.get("density_stats_undirected", {})
    return edge_stats_undirected, density_stats_undirected


def prepare_aligned_inputs(args, metrics_dir: str, cache_dir: str, threshold_tag: str):
    seq_df = read_sequence_dataframe(args.pos_csv, args.neg_csv, args.seq_col, args.id_col, args.site_col, args.chain_col)
    print(f"[INFO] Loaded {len(seq_df)} sequence samples for threshold={threshold_tag}.")

    graph_cache_path = build_graph_cache_path(cache_dir, args.pos_graph_dir, args.neg_graph_dir, args.pdb_dir)
    use_graph_cache = os.path.exists(graph_cache_path) and not args.refresh_graph_cache and not args.refresh_cache
    if use_graph_cache:
        print(f"[INFO] Loading cached graphs from {graph_cache_path}")
        try:
            graphs_by_exact = torch.load(graph_cache_path, map_location="cpu", weights_only=False)
        except TypeError:
            graphs_by_exact = torch.load(graph_cache_path, map_location="cpu")
    else:
        if os.path.exists(graph_cache_path):
            print(f"[INFO] Refreshing graph cache at {graph_cache_path}")
        graph_builder = ProteinGraphBuilder(args.pdb_dir, dist_threshold=CFG.graph_dist_threshold)
        graphs_by_exact = collect_graphs(graph_builder, args.pos_graph_dir, args.neg_graph_dir)
        torch.save(graphs_by_exact, graph_cache_path)
        print(f"[INFO] Saved graph cache to {graph_cache_path}")

    graph_report = None
    if not args.skip_data_scan:
        graph_report = summarize_graph_collection(graphs_by_exact)
        graph_report_path = os.path.join(metrics_dir, f"graph_data_report_{threshold_tag}.json")
        save_metrics(graph_report, graph_report_path)
        edge_stats_undirected, density_stats_undirected = _graph_report_stats(graph_report)
        print(
            "[INFO] Graph scan | "
            f"threshold={threshold_tag} | "
            f"graphs={graph_report['num_graphs']} | "
            f"nodes(mean)={graph_report['node_count_stats'].get('mean', 0):.2f} | "
            f"edges_undir(mean)={edge_stats_undirected.get('mean', 0):.2f} | "
            f"density_undir(mean)={density_stats_undirected.get('mean', 0):.4f} | "
            f"anomalies={graph_report['anomaly_count']}"
        )
        print(f"[INFO] Graph report saved to: {graph_report_path}")

    aligned_df = align_samples_with_graphs(seq_df, graphs_by_exact, strict=args.strict_align)
    aligned_df = aligned_df.copy()
    aligned_df["sample_key"] = (
        aligned_df["protein_id"].astype(str)
        + "|"
        + aligned_df["chain_norm"].fillna("").astype(str)
        + "|"
        + aligned_df["site_norm"].astype(str)
    )
    aligned_df["group_id"] = aligned_df["protein_id"].astype(str)

    conflicting = aligned_df.groupby("sample_key")["label"].nunique()
    conflicting = conflicting[conflicting > 1]
    if not conflicting.empty:
        preview = conflicting.index[:5].tolist()
        raise ValueError(f"Found conflicting labels for identical sample keys. Examples: {preview}")

    before_dedup = len(aligned_df)
    aligned_df = aligned_df.drop_duplicates(subset=["sample_key", "label"]).reset_index(drop=True)
    removed_duplicates = before_dedup - len(aligned_df)
    if removed_duplicates > 0:
        print(f"[INFO] Removed {removed_duplicates} exact duplicate aligned samples before training.")

    esm_cache_path = build_esm_cache_path(cache_dir, aligned_df, args.pos_csv, args.neg_csv)
    use_esm_cache = os.path.exists(esm_cache_path) and not args.refresh_esm_cache and not args.refresh_cache
    if use_esm_cache:
        print(f"[INFO] Loading cached ESM2 features from {esm_cache_path}")
        seq_features = np.load(esm_cache_path)["features"].astype("float32")
    else:
        if os.path.exists(esm_cache_path):
            print(f"[INFO] Refreshing ESM2 cache at {esm_cache_path}")
        tokenizer, esm_model = load_esm2_model(CFG.esm_model, CFG.device)
        seq_features = extract_esm2_features(
            aligned_df["sequence"].tolist(),
            tokenizer=tokenizer,
            model=esm_model,
            max_len=CFG.max_len,
            batch_size=CFG.esm_batch_size,
            device=CFG.device,
        )
        np.savez_compressed(esm_cache_path, features=seq_features)
        print(f"[INFO] Saved ESM2 cache to {esm_cache_path}")

    labels = aligned_df["label"].astype("float32").values
    graphs = aligned_df["graph"].tolist()
    groups = aligned_df["group_id"].tolist()
    sample_keys = aligned_df["sample_key"].tolist()
    diagnostics = {
        "dedup_removed": int(removed_duplicates),
        "unique_sample_keys": int(aligned_df["sample_key"].nunique()),
        "unique_proteins": int(aligned_df["group_id"].nunique()),
    }
    return aligned_df, labels, graphs, groups, sample_keys, seq_features, graph_report, diagnostics


def _print_split_diagnostics(split_name: str, report: dict):
    print(
        f"[INFO] {split_name} split | "
        f"samples={report['samples']} | "
        f"pos={report['positives']} | "
        f"neg={report['negatives']} | "
        f"pos_ratio={report.get('positive_ratio', 0.0):.4f} | "
        f"proteins={report['unique_proteins']} | "
        f"dup_keys={report['duplicate_sample_keys']}"
    )


def _assert_clean_split(overlap_report: dict):
    sample_overlap_total = sum(overlap_report["sample_key_overlap"].values())
    if sample_overlap_total != 0:
        raise RuntimeError(f"Detected duplicate site samples across splits: {overlap_report}")


def run_single_split(
    args,
    metrics_dir: str,
    model_dir: str,
    labels: np.ndarray,
    graphs,
    groups,
    sample_keys,
    seq_features: np.ndarray,
    graph_report,
    aligned_sample_count: int,
    threshold_tag: str,
) -> dict:
    train_idx, val_idx, test_idx = split_indices(
        labels,
        groups,
        seed=CFG.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    split_names = np.array(["train"] * len(labels), dtype=object)
    split_names[val_idx] = "val"
    split_names[test_idx] = "test"

    train_report = summarize_split(train_idx, labels, groups, sample_keys)
    val_report = summarize_split(val_idx, labels, groups, sample_keys)
    test_report = summarize_split(test_idx, labels, groups, sample_keys)
    overlap_report = split_overlap_report(train_idx, val_idx, test_idx, groups, sample_keys)
    _print_split_diagnostics("train", train_report)
    _print_split_diagnostics("val", val_report)
    _print_split_diagnostics("test", test_report)
    _assert_clean_split(overlap_report)
    print(
        "[INFO] Split shared proteins | "
        f"protein(train-val)={overlap_report['protein_overlap']['train_val']} | "
        f"protein(train-test)={overlap_report['protein_overlap']['train_test']} | "
        f"protein(val-test)={overlap_report['protein_overlap']['val_test']}"
    )

    train_labels = labels[train_idx]
    n_pos = int((train_labels == 1).sum())
    n_neg = len(train_labels) - n_pos
    sampler = None
    if n_pos > 0 and n_neg > 0:
        weights = [(n_neg / n_pos) if labels[idx] == 1 else 1.0 for idx in train_idx]
        sampler = WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), len(weights), replacement=True)

    train_ds = MultiModalDataset(seq_features[train_idx], subset_graphs(graphs, train_idx), labels[train_idx], augment=True)
    train_eval_ds = MultiModalDataset(seq_features[train_idx], subset_graphs(graphs, train_idx), labels[train_idx], augment=False)
    val_ds = MultiModalDataset(seq_features[val_idx], subset_graphs(graphs, val_idx), labels[val_idx], augment=False)
    test_ds = MultiModalDataset(seq_features[test_idx], subset_graphs(graphs, test_idx), labels[test_idx], augment=False)
    full_ds = MultiModalDataset(seq_features, graphs, labels, augment=False)

    train_loader = create_loader(train_ds, batch_size=CFG.batch_size, shuffle=(sampler is None), sampler=sampler)
    train_eval_loader = create_loader(train_eval_ds, batch_size=CFG.batch_size)
    val_loader = create_loader(val_ds, batch_size=CFG.batch_size)
    test_loader = create_loader(test_ds, batch_size=CFG.batch_size)
    full_loader = create_loader(full_ds, batch_size=CFG.batch_size)

    model = MultiModalPredictor(seq_input_dim=seq_features.shape[2]).to(CFG.device)
    if CFG.use_focal:
        criterion = FocalLoss(alpha=CFG.focal_alpha, gamma=CFG.focal_gamma)
    elif CFG.use_class_weight and n_pos > 0 and n_neg > 0:
        pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=CFG.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    save_path = os.path.join(model_dir, f"best_multimodal_v1_{threshold_tag}.pth")
    history, val_metrics, test_metrics = train_model(model, train_loader, val_loader, test_loader, criterion, save_path)
    threshold = test_metrics["optimal_threshold_from_val"]
    train_metrics = evaluate_model(model, train_eval_loader, criterion=criterion, device=CFG.device, threshold=threshold)
    val_metrics_plot = evaluate_model(model, val_loader, criterion=criterion, device=CFG.device, threshold=threshold)
    metrics_path = os.path.join(metrics_dir, f"multimodal_v1_metrics_{threshold_tag}.json")
    figures_dir = os.path.join(os.path.dirname(metrics_dir), "figures", threshold_tag)
    os.makedirs(figures_dir, exist_ok=True)
    figure_paths = {}
    if not args.skip_plots:
        try:
            figure_paths = generate_training_plots(
                model=model,
                full_loader=full_loader,
                seq_features=seq_features,
                labels=labels,
                train_metrics=train_metrics,
                val_metrics=val_metrics_plot,
                test_metrics=test_metrics,
                history=history,
                output_dir=figures_dir,
                roc_mode=args.roc_mode,
                tsne_max_samples=args.tsne_max_samples,
                split_names=split_names,
                split_aware_embeddings=False,
            )
        except Exception as exc:
            print(f"[WARN] Failed to generate plots: {exc}")
    payload = {
        "history": history,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "val_metrics_thresholded": val_metrics_plot,
        "test_metrics": test_metrics,
        "aligned_samples": aligned_sample_count,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds),
        "graph_dist_threshold": CFG.graph_dist_threshold,
        "figure_paths": figure_paths,
        "split_report": {
            "train": train_report,
            "val": val_report,
            "test": test_report,
            "overlap": overlap_report,
        },
    }
    if graph_report is not None:
        edge_stats_undirected, density_stats_undirected = _graph_report_stats(graph_report)
        payload["graph_report_summary"] = {
            "num_graphs": graph_report["num_graphs"],
            "anomaly_count": graph_report["anomaly_count"],
            "edge_count_stats_undirected": edge_stats_undirected,
            "density_stats_undirected": density_stats_undirected,
        }
    save_metrics(payload, metrics_path)

    print("[INFO] Training completed.")
    print(f"[INFO] Validation AUC: {val_metrics['auc']:.4f} | Test AUC: {test_metrics['auc']:.4f}")
    print(
        f"[INFO] Avg modality weights (val): "
        f"seq={val_metrics['avg_modality_weights'][0]:.4f}, "
        f"graph={val_metrics['avg_modality_weights'][1]:.4f}"
    )
    print(f"[INFO] Model saved to: {save_path}")
    print(f"[INFO] Metrics saved to: {metrics_path}")
    saved_figures = {key: value for key, value in figure_paths.items() if isinstance(value, str) and value}
    if saved_figures:
        print(f"[INFO] Figures saved to: {figures_dir}")
    elif not args.skip_plots:
        print("[WARN] No figures were saved. Check plotting warnings above and installed packages.")
    return {
        "threshold_tag": threshold_tag,
        "graph_dist_threshold": CFG.graph_dist_threshold,
        "val_auc": float(val_metrics["auc"]),
        "test_auc": float(test_metrics["auc"]),
        "aligned_samples": int(aligned_sample_count),
        "graph_anomalies": int(graph_report["anomaly_count"]) if graph_report is not None else None,
        "graph_density_undirected_mean": float(_graph_report_stats(graph_report)[1].get("mean", 0.0)) if graph_report is not None else None,
        "avg_modality_weights_val": [float(x) for x in val_metrics["avg_modality_weights"]],
        "model_path": save_path,
        "metrics_path": metrics_path,
        "figure_paths": figure_paths,
    }


def run_cross_validation(
    args,
    metrics_dir: str,
    model_dir: str,
    labels: np.ndarray,
    graphs,
    groups,
    sample_keys,
    seq_features: np.ndarray,
    graph_report,
    aligned_sample_count: int,
    threshold_tag: str,
) -> dict:
    splits = build_cv_splits(
        labels,
        groups,
        num_folds=args.num_folds,
        seed=CFG.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    fold_results = []
    for fold_idx, (train_idx, val_idx, test_idx) in enumerate(splits, start=1):
        print(f"[INFO] Running repeated split {fold_idx}/{args.num_folds}")
        split_names = np.array(["train"] * len(labels), dtype=object)
        split_names[val_idx] = "val"
        split_names[test_idx] = "test"

        train_report = summarize_split(train_idx, labels, groups, sample_keys)
        val_report = summarize_split(val_idx, labels, groups, sample_keys)
        test_report = summarize_split(test_idx, labels, groups, sample_keys)
        overlap_report = split_overlap_report(train_idx, val_idx, test_idx, groups, sample_keys)
        _print_split_diagnostics(f"fold{fold_idx}-train", train_report)
        _print_split_diagnostics(f"fold{fold_idx}-val", val_report)
        _print_split_diagnostics(f"fold{fold_idx}-test", test_report)
        _assert_clean_split(overlap_report)
        print(
            "[INFO] Fold shared proteins | "
            f"protein(train-val)={overlap_report['protein_overlap']['train_val']} | "
            f"protein(train-test)={overlap_report['protein_overlap']['train_test']} | "
            f"protein(val-test)={overlap_report['protein_overlap']['val_test']}"
        )

        train_labels = labels[train_idx]
        n_pos = int((train_labels == 1).sum())
        n_neg = len(train_labels) - n_pos
        sampler = None
        if n_pos > 0 and n_neg > 0:
            weights = [(n_neg / n_pos) if labels[idx] == 1 else 1.0 for idx in train_idx]
            sampler = WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), len(weights), replacement=True)

        train_ds = MultiModalDataset(seq_features[train_idx], subset_graphs(graphs, train_idx), labels[train_idx], augment=True)
        train_eval_ds = MultiModalDataset(seq_features[train_idx], subset_graphs(graphs, train_idx), labels[train_idx], augment=False)
        val_ds = MultiModalDataset(seq_features[val_idx], subset_graphs(graphs, val_idx), labels[val_idx], augment=False)
        test_ds = MultiModalDataset(seq_features[test_idx], subset_graphs(graphs, test_idx), labels[test_idx], augment=False)
        full_ds = MultiModalDataset(seq_features, graphs, labels, augment=False)

        train_loader = create_loader(train_ds, batch_size=CFG.batch_size, shuffle=(sampler is None), sampler=sampler)
        train_eval_loader = create_loader(train_eval_ds, batch_size=CFG.batch_size)
        val_loader = create_loader(val_ds, batch_size=CFG.batch_size)
        test_loader = create_loader(test_ds, batch_size=CFG.batch_size)
        full_loader = create_loader(full_ds, batch_size=CFG.batch_size)

        model = MultiModalPredictor(seq_input_dim=seq_features.shape[2]).to(CFG.device)
        if CFG.use_focal:
            criterion = FocalLoss(alpha=CFG.focal_alpha, gamma=CFG.focal_gamma)
        elif CFG.use_class_weight and n_pos > 0 and n_neg > 0:
            pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=CFG.device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = nn.BCEWithLogitsLoss()

        save_path = os.path.join(model_dir, f"best_multimodal_v1_{threshold_tag}_fold{fold_idx}.pth")
        history, val_metrics, test_metrics = train_model(model, train_loader, val_loader, test_loader, criterion, save_path)
        threshold = test_metrics["optimal_threshold_from_val"]
        train_metrics = evaluate_model(model, train_eval_loader, criterion=criterion, device=CFG.device, threshold=threshold)
        val_metrics_plot = evaluate_model(model, val_loader, criterion=criterion, device=CFG.device, threshold=threshold)

        figures_dir = os.path.join(os.path.dirname(metrics_dir), "figures", threshold_tag, f"fold_{fold_idx}")
        os.makedirs(figures_dir, exist_ok=True)
        figure_paths = {}
        if not args.skip_plots:
            try:
                figure_paths = generate_training_plots(
                    model=model,
                    full_loader=full_loader,
                    seq_features=seq_features,
                    labels=labels,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics_plot,
                    test_metrics=test_metrics,
                    history=history,
                    output_dir=figures_dir,
                    roc_mode=args.roc_mode,
                    tsne_max_samples=args.tsne_max_samples,
                    split_names=split_names,
                    split_aware_embeddings=False,
                )
            except Exception as exc:
                print(f"[WARN] Failed to generate plots for fold {fold_idx}: {exc}")

        fold_results.append(
            {
                "fold": fold_idx,
                "history": history,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "val_metrics_thresholded": val_metrics_plot,
                "test_metrics": test_metrics,
                "train_samples": len(train_ds),
                "val_samples": len(val_ds),
                "test_samples": len(test_ds),
                "split_report": {
                    "train": train_report,
                    "val": val_report,
                    "test": test_report,
                    "overlap": overlap_report,
                },
                "model_path": save_path,
                "figure_paths": figure_paths,
            }
        )

    summary = {
        "val_auc_mean": float(np.mean([item["val_metrics"]["auc"] for item in fold_results])),
        "val_auc_std": float(np.std([item["val_metrics"]["auc"] for item in fold_results])),
        "test_auc_mean": float(np.mean([item["test_metrics"]["auc"] for item in fold_results])),
        "test_auc_std": float(np.std([item["test_metrics"]["auc"] for item in fold_results])),
    }
    metrics_path = os.path.join(metrics_dir, f"multimodal_v1_cv_metrics_{threshold_tag}.json")
    payload = {
        "num_repeated_splits": args.num_folds,
        "aligned_samples": int(aligned_sample_count),
        "graph_dist_threshold": CFG.graph_dist_threshold,
        "summary": summary,
        "fold_results": fold_results,
    }
    if graph_report is not None:
        edge_stats_undirected, density_stats_undirected = _graph_report_stats(graph_report)
        payload["graph_report_summary"] = {
            "num_graphs": graph_report["num_graphs"],
            "anomaly_count": graph_report["anomaly_count"],
            "edge_count_stats_undirected": edge_stats_undirected,
            "density_stats_undirected": density_stats_undirected,
        }
    save_metrics(payload, metrics_path)
    print(
        "[INFO] Repeated site-aware splits completed | "
        f"val_auc={summary['val_auc_mean']:.4f}±{summary['val_auc_std']:.4f} | "
        f"test_auc={summary['test_auc_mean']:.4f}±{summary['test_auc_std']:.4f}"
    )
    return {
        "threshold_tag": threshold_tag,
        "graph_dist_threshold": CFG.graph_dist_threshold,
        "num_repeated_splits": args.num_folds,
        "val_auc_mean": summary["val_auc_mean"],
        "val_auc_std": summary["val_auc_std"],
        "test_auc_mean": summary["test_auc_mean"],
        "test_auc_std": summary["test_auc_std"],
        "metrics_path": metrics_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Multimodal v1: ESM2 sequence modality + GNN graph modality")
    parser.add_argument("--pos_csv", type=str, required=True)
    parser.add_argument("--neg_csv", type=str, required=True)
    parser.add_argument("--pos_graph_dir", type=str, default="./pos-results/hbond_data")
    parser.add_argument("--neg_graph_dir", type=str, default="./neg-results/hbond_data")
    parser.add_argument("--pdb_dir", type=str, default="../pdbs")
    parser.add_argument("--seq_col", type=str, default="Fragment")
    parser.add_argument("--id_col", type=str, default="ID")
    parser.add_argument("--site_col", type=str, default="Site")
    parser.add_argument("--chain_col", type=str, default="Chain")
    parser.add_argument("--save_dir", type=str, default="./results_multimodal_v1")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--strict_align", action="store_true")
    parser.add_argument("--skip_data_scan", action="store_true")
    parser.add_argument("--max_len", type=int, default=CFG.max_len)
    parser.add_argument("--batch_size", type=int, default=CFG.batch_size)
    parser.add_argument("--esm_batch_size", type=int, default=CFG.esm_batch_size)
    parser.add_argument("--epochs", type=int, default=CFG.epochs)
    parser.add_argument("--patience", type=int, default=CFG.patience, help="Stop when validation AUC has not improved for this many epochs.")
    parser.add_argument("--num_workers", type=int, default=CFG.num_workers)
    parser.add_argument("--lr", type=float, default=CFG.lr)
    parser.add_argument("--weight_decay", type=float, default=CFG.weight_decay)
    parser.add_argument("--sequence_lr_scale", type=float, default=CFG.sequence_lr_scale, help="LR multiplier for the ESM2 sequence encoder head.")
    parser.add_argument("--graph_lr_scale", type=float, default=CFG.graph_lr_scale, help="LR multiplier for graph encoder and graph auxiliary head.")
    parser.add_argument("--head_lr_scale", type=float, default=CFG.head_lr_scale, help="LR multiplier for fusion and final classifier layers.")
    parser.add_argument("--scheduler_type", type=str, default=CFG.scheduler_type, choices=["warmup_cosine", "plateau"], help="M3Site-style warmup cosine scheduler or ReduceLROnPlateau.")
    parser.add_argument("--warmup_ratio", type=float, default=CFG.warmup_ratio, help="Warmup fraction for warmup_cosine scheduler.")
    parser.add_argument("--scheduler_patience", type=int, default=CFG.scheduler_patience, help="ReduceLROnPlateau patience measured in validation epochs.")
    parser.add_argument("--dropout_lstm", type=float, default=CFG.dropout_lstm)
    parser.add_argument("--dropout_head", type=float, default=CFG.dropout_head)
    parser.add_argument("--spatial_dropout", type=float, default=CFG.spatial_dropout)
    parser.add_argument("--modality_dropout", type=float, default=CFG.modality_dropout)
    parser.add_argument("--label_smooth_eps", type=float, default=CFG.label_smooth_eps)
    parser.add_argument("--modality_min_weight", type=float, default=CFG.modality_min_weight, help="Minimum weight assigned to each modality after softmax gating.")
    parser.add_argument("--modality_graph_target_weight", type=float, default=CFG.modality_graph_target_weight, help="Target average graph-modality weight used by the optional balance regularizer.")
    parser.add_argument("--modality_balance_lambda", type=float, default=CFG.modality_balance_lambda, help="Strength of graph-modality balance regularization.")
    parser.add_argument("--graph_aux_loss_weight", type=float, default=CFG.graph_aux_loss_weight, help="Auxiliary supervised loss weight for the graph encoder branch.")
    parser.add_argument("--center_loss_weight", type=float, default=CFG.center_loss_weight, help="M3Site-style center loss weight on fused embeddings.")
    parser.add_argument("--inter_loss_weight", type=float, default=CFG.inter_loss_weight, help="M3Site-style inter-class center separation loss weight.")
    parser.add_argument("--inter_loss_margin", type=float, default=CFG.inter_loss_margin, help="Minimum distance encouraged between binary class centers.")
    parser.add_argument("--best_metric", type=str, default=CFG.best_metric, choices=["auc", "f1", "mcc"], help="Validation metric used for checkpoint selection and early stopping.")
    parser.add_argument("--graph_mpnn_layers", type=int, default=CFG.graph_mpnn_layers, help="Number of edge-aware MPNN layers in the graph encoder.")
    parser.add_argument("--no_graph_geometry", action="store_true", help="Disable M3Site-lite coordinate features inside graph message passing.")
    parser.add_argument("--structure_plddt_threshold", type=float, default=CFG.structure_plddt_threshold, help="Use graph structure only when local mean pLDDT is at least this value.")
    parser.add_argument("--no_plddt_structure_mask", action="store_true", help="Always use graph structure regardless of local mean pLDDT.")
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--graph_dist_threshold", type=float, default=CFG.graph_dist_threshold)
    parser.add_argument("--scan_graph_thresholds", type=str, default=None, help="Comma-separated graph distance thresholds, e.g. 6,7,8")
    parser.add_argument("--refresh_cache", action="store_true", help="Rebuild both graph cache and ESM cache for this run.")
    parser.add_argument("--refresh_graph_cache", action="store_true", help="Rebuild only the graph cache for this run.")
    parser.add_argument("--refresh_esm_cache", action="store_true", help="Rebuild only the ESM cache for this run.")
    parser.add_argument("--skip_plots", action="store_true", help="Skip post-training plots.")
    parser.add_argument("--roc_mode", type=str, default="all", choices=["test", "all"], help="Plot only test ROC or train/val/test together.")
    parser.add_argument("--tsne_max_samples", type=int, default=2000, help="Maximum samples used to draw each t-SNE figure.")
    parser.add_argument("--embedding_plot_mode", type=str, default="pooled", choices=["pooled"], help="Draw t-SNE embeddings pooled across all splits and color only by label.")
    parser.add_argument("--num_folds", type=int, default=1, help="Run this many repeated site-aware train/val/test splits when >1.")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Random assignment weight for train split.")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Random assignment weight for validation split.")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Random assignment weight for test split.")
    args = parser.parse_args()

    if args.train_ratio <= 0 or args.val_ratio <= 0 or args.test_ratio <= 0:
        raise ValueError("--train_ratio, --val_ratio, and --test_ratio must all be positive.")

    CFG.max_len = args.max_len
    CFG.batch_size = args.batch_size
    CFG.esm_batch_size = args.esm_batch_size
    CFG.epochs = args.epochs
    CFG.patience = args.patience
    CFG.num_workers = args.num_workers
    CFG.lr = args.lr
    CFG.weight_decay = args.weight_decay
    CFG.sequence_lr_scale = args.sequence_lr_scale
    CFG.graph_lr_scale = args.graph_lr_scale
    CFG.head_lr_scale = args.head_lr_scale
    CFG.scheduler_type = args.scheduler_type
    CFG.warmup_ratio = args.warmup_ratio
    CFG.scheduler_patience = args.scheduler_patience
    CFG.dropout_lstm = args.dropout_lstm
    CFG.dropout_head = args.dropout_head
    CFG.spatial_dropout = args.spatial_dropout
    CFG.modality_dropout = args.modality_dropout
    CFG.label_smooth_eps = args.label_smooth_eps
    CFG.modality_min_weight = args.modality_min_weight
    CFG.modality_graph_target_weight = args.modality_graph_target_weight
    CFG.modality_balance_lambda = args.modality_balance_lambda
    CFG.graph_aux_loss_weight = args.graph_aux_loss_weight
    CFG.center_loss_weight = args.center_loss_weight
    CFG.inter_loss_weight = args.inter_loss_weight
    CFG.inter_loss_margin = args.inter_loss_margin
    CFG.best_metric = args.best_metric
    CFG.graph_mpnn_layers = args.graph_mpnn_layers
    CFG.graph_use_geometry = not args.no_graph_geometry
    CFG.structure_plddt_threshold = args.structure_plddt_threshold
    CFG.use_plddt_structure_mask = not args.no_plddt_structure_mask
    CFG.graph_dist_threshold = args.graph_dist_threshold
    if not 0.0 <= CFG.modality_min_weight < 0.5:
        raise ValueError("--modality_min_weight must be in [0.0, 0.5).")
    if not 0.0 <= CFG.modality_graph_target_weight <= 1.0:
        raise ValueError("--modality_graph_target_weight must be in [0.0, 1.0].")
    if CFG.modality_balance_lambda < 0.0:
        raise ValueError("--modality_balance_lambda must be non-negative.")
    if CFG.graph_aux_loss_weight < 0.0:
        raise ValueError("--graph_aux_loss_weight must be non-negative.")
    if CFG.graph_mpnn_layers < 1:
        raise ValueError("--graph_mpnn_layers must be at least 1.")
    if CFG.structure_plddt_threshold < 0.0:
        raise ValueError("--structure_plddt_threshold must be non-negative.")
    if CFG.scheduler_patience < 0:
        raise ValueError("--scheduler_patience must be non-negative.")
    if not 0.0 <= CFG.warmup_ratio < 1.0:
        raise ValueError("--warmup_ratio must be in [0.0, 1.0).")
    if CFG.center_loss_weight < 0.0 or CFG.inter_loss_weight < 0.0:
        raise ValueError("M3Site-style loss weights must be non-negative.")
    if CFG.inter_loss_margin <= 0.0:
        raise ValueError("--inter_loss_margin must be positive.")
    if min(CFG.sequence_lr_scale, CFG.graph_lr_scale, CFG.head_lr_scale) <= 0:
        raise ValueError("LR scales must be positive.")
    if min(CFG.dropout_lstm, CFG.dropout_head, CFG.spatial_dropout, CFG.modality_dropout) < 0 or max(
        CFG.dropout_lstm, CFG.dropout_head, CFG.spatial_dropout, CFG.modality_dropout
    ) >= 1:
        raise ValueError("Dropout values must be in [0.0, 1.0).")
    if not 0.0 <= CFG.label_smooth_eps < 1.0:
        raise ValueError("--label_smooth_eps must be in [0.0, 1.0).")
    if args.use_amp and args.no_amp:
        raise ValueError("Choose only one of --use_amp or --no_amp.")
    if args.use_amp:
        CFG.use_amp = True
    elif args.no_amp:
        CFG.use_amp = False

    set_seed(CFG.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    metrics_dir = os.path.join(args.save_dir, "metrics")
    model_dir = os.path.join(args.save_dir, "models")
    cache_dir = args.cache_dir or os.path.join(args.save_dir, "cache")
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    original_cfg = deepcopy(CFG)
    thresholds = [CFG.graph_dist_threshold]
    if args.scan_graph_thresholds:
        thresholds = [float(item.strip()) for item in args.scan_graph_thresholds.split(",") if item.strip()]
        if not thresholds:
            raise ValueError("--scan_graph_thresholds was provided but no valid thresholds were parsed.")

    scan_results = []
    for threshold in thresholds:
        CFG.graph_dist_threshold = float(threshold)
        threshold_tag = f"thr_{str(threshold).replace('.', 'p')}"
        print(f"[INFO] Running experiment for graph_dist_threshold={CFG.graph_dist_threshold}")
        prepared = prepare_aligned_inputs(args, metrics_dir, cache_dir, threshold_tag)
        aligned_df, labels, graphs, groups, sample_keys, seq_features, graph_report, diagnostics = prepared
        print(
            "[INFO] Data diagnostics | "
            f"aligned={len(aligned_df)} | "
            f"unique_proteins={diagnostics['unique_proteins']} | "
            f"unique_sample_keys={diagnostics['unique_sample_keys']} | "
            f"dedup_removed={diagnostics['dedup_removed']}"
        )
        if args.num_folds > 1:
            scan_results.append(
                run_cross_validation(
                    args,
                    metrics_dir,
                    model_dir,
                    labels,
                    graphs,
                    groups,
                    sample_keys,
                    seq_features,
                    graph_report,
                    len(aligned_df),
                    threshold_tag,
                )
            )
        else:
            scan_results.append(
                run_single_split(
                    args,
                    metrics_dir,
                    model_dir,
                    labels,
                    graphs,
                    groups,
                    sample_keys,
                    seq_features,
                    graph_report,
                    len(aligned_df),
                    threshold_tag,
                )
            )

    if len(scan_results) > 1:
        summary_path = os.path.join(metrics_dir, "graph_threshold_scan_summary.json")
        save_metrics({"scan_results": scan_results}, summary_path)
        best_metric = "val_auc_mean" if args.num_folds > 1 else "val_auc"
        best_result = max(scan_results, key=lambda item: item[best_metric])
        print(f"[INFO] Threshold scan summary saved to: {summary_path}")
        if args.num_folds > 1:
            print(
                "[INFO] Best threshold by validation AUC mean | "
                f"threshold={best_result['graph_dist_threshold']:.2f} | "
                f"val_auc={best_result['val_auc_mean']:.4f} | "
                f"test_auc={best_result['test_auc_mean']:.4f}"
            )
        else:
            print(
                "[INFO] Best threshold by validation AUC | "
                f"threshold={best_result['graph_dist_threshold']:.2f} | "
                f"val_auc={best_result['val_auc']:.4f} | "
                f"test_auc={best_result['test_auc']:.4f}"
            )

    CFG.graph_dist_threshold = original_cfg.graph_dist_threshold


if __name__ == "__main__":
    main()
