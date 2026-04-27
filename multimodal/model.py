import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import MessagePassing, global_add_pool, global_max_pool, global_mean_pool

from .config import CFG


class SequenceEncoder(nn.Module):
    def __init__(self, seq_input_dim: int):
        super().__init__()
        self.conv = nn.Conv1d(seq_input_dim, CFG.conv_filters, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.bn = nn.BatchNorm1d(CFG.conv_filters)
        self.spatial_dropout = nn.Dropout2d(CFG.spatial_dropout)
        self.lstm = nn.LSTM(CFG.conv_filters, CFG.lstm_units, num_layers=1, batch_first=True, bidirectional=True)
        self.lstm_dropout = nn.Dropout(CFG.dropout_lstm)
        self.att_linear = nn.Linear(2 * CFG.lstm_units, CFG.lstm_units)
        self.att_score = nn.Linear(CFG.lstm_units, 1)

    def forward(self, seq_x: torch.Tensor) -> torch.Tensor:
        x = seq_x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.relu(x)
        x = self.spatial_dropout(x.unsqueeze(-1)).squeeze(-1)
        x = self.pool(x)
        x = self.bn(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.lstm_dropout(lstm_out)
        att = torch.tanh(self.att_linear(lstm_out))
        att = self.att_score(att)
        weights = torch.softmax(att, dim=1)
        return torch.sum(lstm_out * weights, dim=1)


class EdgeAwareMPNNLayer(MessagePassing):
    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float, use_geometry: bool = True):
        super().__init__(aggr="mean")
        self.use_geometry = use_geometry
        extra_dim = 4 if use_geometry else 0
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim + extra_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, pos: torch.Tensor = None) -> torch.Tensor:
        if edge_index.numel() == 0:
            return x
        rel_geom = None
        if self.use_geometry and pos is not None:
            rel_vec = (pos[edge_index[0]] - pos[edge_index[1]]) / max(float(CFG.graph_dist_threshold), 1.0)
            rel_dist2 = rel_vec.pow(2).sum(dim=1, keepdim=True)
            rel_geom = torch.cat([rel_vec, rel_dist2], dim=1)
        aggregated = self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr, rel_geom=rel_geom)
        updated = self.gru(aggregated, x)
        return self.norm(x + self.dropout(updated))

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: torch.Tensor, rel_geom: torch.Tensor = None) -> torch.Tensor:
        pieces = [x_i, x_j, edge_attr]
        if self.use_geometry:
            if rel_geom is None:
                rel_geom = torch.zeros((edge_attr.size(0), 4), dtype=edge_attr.dtype, device=edge_attr.device)
            pieces.append(rel_geom)
        return self.message_mlp(torch.cat(pieces, dim=1))


class GraphEncoder(nn.Module):
    def __init__(self, node_input_dim: int = CFG.graph_node_dim, edge_dim: int = CFG.graph_edge_dim):
        super().__init__()
        hidden = CFG.graph_hidden_dim
        self.node_project = nn.Sequential(
            nn.Linear(node_input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(CFG.dropout_head),
        )
        self.mpnn_layers = nn.ModuleList(
            [
                EdgeAwareMPNNLayer(
                    hidden_dim=hidden,
                    edge_dim=edge_dim,
                    dropout=CFG.dropout_head,
                    use_geometry=CFG.graph_use_geometry,
                )
                for _ in range(CFG.graph_mpnn_layers)
            ]
        )
        self.project = nn.Sequential(
            nn.Linear(hidden * 3, CFG.graph_emb_dim),
            nn.ReLU(),
            nn.Dropout(CFG.dropout_head),
            nn.Linear(CFG.graph_emb_dim, CFG.graph_emb_dim),
        )

    def forward(self, graph_batch: Batch) -> torch.Tensor:
        x, edge_index, edge_attr, batch = graph_batch.x, graph_batch.edge_index, graph_batch.edge_attr, graph_batch.batch
        pos = graph_batch.pos if hasattr(graph_batch, "pos") else None
        x = self.node_project(x)
        for layer in self.mpnn_layers:
            x = layer(x, edge_index, edge_attr, pos=pos)
        pooled = torch.cat([global_mean_pool(x, batch), global_add_pool(x, batch), global_max_pool(x, batch)], dim=1)
        return self.project(pooled)


class CrossAttention(nn.Module):
    def __init__(self, query_dim: int, context_dim: int, dropout: float):
        super().__init__()
        self.query_dim = query_dim
        self.key = nn.Linear(context_dim, query_dim)
        self.value = nn.Linear(context_dim, query_dim)
        self.out = nn.Linear(query_dim, query_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, return_attn: bool = False):
        key_proj = self.key(key)
        value_proj = self.value(value)
        scores = torch.matmul(query, key_proj.transpose(-2, -1)) / (self.query_dim ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(self.dropout(weights), value_proj)
        output = self.out(context)
        if return_attn:
            return output, weights
        return output


class FeedForward(nn.Module):
    def __init__(self, dim: int, ff_dim: int, dropout: float, condition_dim: int = 0):
        super().__init__()
        self.condition_dim = condition_dim
        self.fc1 = nn.Linear(dim + condition_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, condition: torch.Tensor = None):
        if self.condition_dim > 0 and condition is not None:
            condition = condition.unsqueeze(1).expand(-1, x.size(1), -1)
            x = torch.cat([x, condition], dim=-1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class FunICross(nn.Module):
    def __init__(self, query_dim: int, context_dim: int, ff_dim: int, dropout: float, condition_dim: int = 0):
        super().__init__()
        self.attn = CrossAttention(query_dim=query_dim, context_dim=context_dim, dropout=dropout)
        self.attn_norm = nn.LayerNorm(query_dim)
        self.ff = FeedForward(query_dim, ff_dim=ff_dim, dropout=dropout, condition_dim=condition_dim)
        self.ff_norm = nn.LayerNorm(query_dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, condition: torch.Tensor = None, return_attn: bool = False):
        if return_attn:
            attn_out, attn_weights = self.attn(query, key, value, return_attn=True)
        else:
            attn_out = self.attn(query, key, value)
            attn_weights = None
        query = self.attn_norm(query + attn_out)
        query = self.ff_norm(query + self.ff(query, condition=condition))
        if return_attn:
            return query, attn_weights
        return query


class FunICrossModalFusion(nn.Module):
    def __init__(self, seq_dim: int, graph_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.seq_proj = nn.Linear(seq_dim, hidden_dim)
        self.graph_proj = nn.Linear(graph_dim, hidden_dim)
        self.graph_to_seq = FunICross(query_dim=hidden_dim, context_dim=hidden_dim, ff_dim=hidden_dim, dropout=dropout)
        self.seq_to_graph = FunICross(query_dim=hidden_dim, context_dim=hidden_dim, ff_dim=hidden_dim, dropout=dropout)

    def forward(self, seq_emb: torch.Tensor, graph_emb: torch.Tensor):
        seq_base = self.seq_proj(seq_emb).unsqueeze(1)
        graph_base = self.graph_proj(graph_emb).unsqueeze(1)
        seq_cross = self.graph_to_seq(seq_base, graph_base, graph_base).squeeze(1)
        graph_cross = self.seq_to_graph(graph_base, seq_base, seq_base).squeeze(1)
        return seq_base.squeeze(1), graph_base.squeeze(1), seq_cross, graph_cross


class MultiModalPredictor(nn.Module):
    def __init__(self, seq_input_dim: int):
        super().__init__()
        self.sequence_encoder = SequenceEncoder(seq_input_dim=seq_input_dim)
        self.graph_encoder = GraphEncoder()
        self.modality_dropout = nn.Dropout(CFG.modality_dropout)
        seq_dim = 2 * CFG.lstm_units
        graph_dim = CFG.graph_emb_dim
        hidden_dim = CFG.fusion_hidden_dim
        fused_dim = hidden_dim * 2

        self.cross_modal_fusion = FunICrossModalFusion(
            seq_dim=seq_dim,
            graph_dim=graph_dim,
            hidden_dim=hidden_dim,
            dropout=CFG.dropout_head,
        )
        self.modality_gate = nn.Sequential(nn.Linear(fused_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2))
        self.graph_classifier = nn.Sequential(
            nn.Dropout(CFG.dropout_head),
            nn.Linear(graph_dim, 64),
            nn.ReLU(),
            nn.Dropout(CFG.dropout_head),
            nn.Linear(64, 1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(CFG.dropout_head),
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.Dropout(CFG.dropout_head),
            nn.Linear(128, 1),
        )

    def encode_modalities(self, seq_x: torch.Tensor, graph_batch: Batch):
        seq_emb = self.sequence_encoder(seq_x)
        graph_emb = self.graph_encoder(graph_batch)
        structure_mask = getattr(graph_batch, "structure_mask", None)
        if structure_mask is not None:
            structure_mask = structure_mask.view(-1, 1).to(graph_emb.device, dtype=graph_emb.dtype)
            graph_emb = graph_emb * structure_mask
        return seq_emb, graph_emb

    def build_multimodal_embedding(self, seq_emb: torch.Tensor, graph_emb: torch.Tensor, structure_mask: torch.Tensor = None):
        seq_base, graph_base, seq_cross, graph_cross = self.cross_modal_fusion(seq_emb, graph_emb)
        raw_weights = torch.softmax(self.modality_gate(torch.cat([seq_cross, graph_cross], dim=1)), dim=1)
        min_weight = float(CFG.modality_min_weight)
        modality_weights = min_weight + (1.0 - 2.0 * min_weight) * raw_weights
        if structure_mask is not None:
            structure_mask = structure_mask.view(-1, 1).to(modality_weights.device, dtype=modality_weights.dtype)
            modality_weights = torch.cat(
                [
                    modality_weights[:, 0:1] * structure_mask + (1.0 - structure_mask),
                    modality_weights[:, 1:2] * structure_mask,
                ],
                dim=1,
            )
        final_emb = torch.cat(
            [seq_cross * modality_weights[:, 0:1], graph_cross * modality_weights[:, 1:2]],
            dim=1,
        )
        return final_emb, modality_weights

    def forward(self, seq_x: torch.Tensor, graph_batch: Batch):
        structure_mask = getattr(graph_batch, "structure_mask", None)
        seq_emb, graph_emb = self.encode_modalities(seq_x, graph_batch)
        seq_emb = self.modality_dropout(seq_emb)
        graph_emb = self.modality_dropout(graph_emb)
        final_emb, modality_weights = self.build_multimodal_embedding(seq_emb, graph_emb, structure_mask=structure_mask)
        logits = self.classifier(final_emb).squeeze(1)
        graph_logits = self.graph_classifier(graph_emb).squeeze(1)
        return logits, modality_weights, graph_logits, final_emb, structure_mask


class FocalLoss(nn.Module):
    def __init__(self, alpha: float, gamma: float, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        pt = torch.exp(-bce_loss)
        alpha_tensor = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal = alpha_tensor * (1 - pt) ** self.gamma * bce_loss
        if self.reduction == "sum":
            return focal.sum()
        if self.reduction == "mean":
            return focal.mean()
        return focal
