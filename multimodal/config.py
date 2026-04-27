from dataclasses import dataclass
from typing import Literal

import torch


@dataclass
class Config:
    graph_cache_version: str = "v4_m3site_lite"
    esm_cache_version: str = "v2"
    esm_model: str = "facebook/esm2_t36_3B_UR50D"
    seed: int = 42
    batch_size: int = 16
    esm_batch_size: int = 8
    epochs: int = 40
    lr: float = 1e-4
    weight_decay: float = 5e-5
    sequence_lr_scale: float = 0.5
    graph_lr_scale: float = 1.25
    head_lr_scale: float = 1.0
    scheduler_type: Literal["warmup_cosine", "plateau"] = "warmup_cosine"
    warmup_ratio: float = 0.1
    scheduler_patience: int = 1
    patience: int = 5
    clip_norm: float = 1.0
    num_workers: int = 0
    use_amp: bool = False
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_len: int = 31
    graph_node_dim: int = 48
    graph_edge_dim: int = 10
    graph_hidden_dim: int = 256
    graph_emb_dim: int = 256
    graph_mpnn_layers: int = 3
    graph_use_geometry: bool = True
    use_plddt_structure_mask: bool = True
    structure_plddt_threshold: float = 85.0
    lstm_units: int = 128
    conv_filters: int = 128
    dropout_lstm: float = 0.5
    dropout_head: float = 0.35
    spatial_dropout: float = 0.25
    modality_dropout: float = 0.1
    fusion_hidden_dim: int = 128
    modality_min_weight: float = 0.1
    modality_graph_target_weight: float = 0.2
    modality_balance_lambda: float = 0.02
    graph_aux_loss_weight: float = 0.2
    center_loss_weight: float = 0.01
    inter_loss_weight: float = 0.001
    inter_loss_margin: float = 1.0
    best_metric: Literal["auc", "f1", "mcc"] = "auc"
    graph_dist_threshold: float = 8.0
    normalize_graph_node_features: bool = True
    normalize_graph_edge_features: bool = True
    edge_seq_dist_log1p: bool = True
    gauss_std: float = 0.01
    drop_prob: float = 0.05
    max_crop_frac: float = 0.08
    use_focal: bool = True
    focal_alpha: float = 0.8
    focal_gamma: float = 2.0
    use_class_weight: bool = True
    label_smooth_eps: float = 0.01


CFG = Config()
