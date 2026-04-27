import os
import re
from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

import numpy as np
import pandas as pd
import torch
from Bio.PDB import PDBParser
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from .config import CFG


AA_ORDER = [
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
    "UNK",
]
AA_TO_IDX = {aa: idx for idx, aa in enumerate(AA_ORDER)}

# Kyte-Doolittle hydrophobicity, approximate residue volume, molecular weight,
# polarity flag, pH-7 charge, and aromatic flag. Continuous values are scaled
# into a small numeric range before concatenation with graph node features.
AA_PROPERTIES = {
    "ALA": [1.8, 88.6, 89.1, 0.0, 0.0, 0.0, 2.34],
    "ARG": [-4.5, 173.4, 174.2, 1.0, 1.0, 0.0, 9.04],
    "ASN": [-3.5, 114.1, 132.1, 1.0, 0.0, 0.0, 2.02],
    "ASP": [-3.5, 111.1, 133.1, 1.0, -1.0, 0.0, 1.88],
    "CYS": [2.5, 108.5, 121.2, 1.0, 0.0, 0.0, 1.96],
    "GLN": [-3.5, 143.8, 146.2, 1.0, 0.0, 0.0, 2.17],
    "GLU": [-3.5, 138.4, 147.1, 1.0, -1.0, 0.0, 2.19],
    "GLY": [-0.4, 60.1, 75.1, 0.0, 0.0, 0.0, 2.34],
    "HIS": [-3.2, 153.2, 155.2, 1.0, 0.1, 1.0, 1.82],
    "ILE": [4.5, 166.7, 131.2, 0.0, 0.0, 0.0, 2.36],
    "LEU": [3.8, 166.7, 131.2, 0.0, 0.0, 0.0, 2.36],
    "LYS": [-3.9, 168.6, 146.2, 1.0, 1.0, 0.0, 2.18],
    "MET": [1.9, 162.9, 149.2, 0.0, 0.0, 0.0, 2.28],
    "PHE": [2.8, 189.9, 165.2, 0.0, 0.0, 1.0, 1.83],
    "PRO": [-1.6, 112.7, 115.1, 0.0, 0.0, 0.0, 1.99],
    "SER": [-0.8, 89.0, 105.1, 1.0, 0.0, 0.0, 2.21],
    "THR": [-0.7, 116.1, 119.1, 1.0, 0.0, 0.0, 2.15],
    "TRP": [-0.9, 227.8, 204.2, 0.0, 0.0, 1.0, 2.83],
    "TYR": [-1.3, 193.6, 181.2, 1.0, 0.0, 1.0, 2.20],
    "VAL": [4.2, 140.0, 117.1, 0.0, 0.0, 0.0, 2.32],
    "UNK": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
}

ATOM_MASS = {"C": 12.011, "N": 14.007, "O": 15.999, "S": 32.06, "H": 1.008}
ATOM_NUMBER = {"C": 6.0, "N": 7.0, "O": 8.0, "S": 16.0, "H": 1.0}
VDW_RADIUS = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "H": 1.20}
RING_ATOMS = {
    "HIS": {"ND1", "CE1", "NE2", "CD2"},
    "PHE": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "TYR": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "TRP": {"CG", "CD1", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"},
}


def amino_acid_features(resnames: List[str]) -> np.ndarray:
    features = []
    for raw_name in resnames:
        aa = str(raw_name).strip().upper()
        if aa not in AA_TO_IDX:
            aa = "UNK"
        one_hot = np.eye(len(AA_ORDER), dtype=np.float32)[AA_TO_IDX[aa]]
        hydrophobicity, volume, mass, polar, charge, aromatic, pka = AA_PROPERTIES[aa]
        props = np.array(
            [
                hydrophobicity / 4.5,
                volume / 228.0,
                mass / 204.2,
                polar,
                charge,
                aromatic,
                pka / 14.0,
            ],
            dtype=np.float32,
        )
        features.append(np.concatenate([one_hot, props], axis=0))
    return np.asarray(features, dtype=np.float32)


def atom_summary_features(residue) -> np.ndarray:
    atom_rows = []
    ring_atoms = RING_ATOMS.get(str(residue.resname).upper(), set())
    for atom in residue:
        element = str(getattr(atom, "element", "")).strip().upper()
        if element == "H":
            continue
        atom_name = str(atom.get_id()).strip()
        atom_rows.append(
            [
                ATOM_MASS.get(element, 0.0) / 32.06,
                float(getattr(atom, "bfactor", 0.0)) / 100.0,
                1.0 if atom_name not in {"N", "CA", "C", "O"} else 0.0,
                ATOM_NUMBER.get(element, 0.0) / 16.0,
                VDW_RADIUS.get(element, 0.0) / 2.0,
                1.0 if atom_name in ring_atoms else 0.0,
                1.0,
            ]
        )
    if not atom_rows:
        return np.zeros(7, dtype=np.float32)
    arr = np.asarray(atom_rows, dtype=np.float32)
    out = arr.mean(axis=0)
    out[-1] = min(float(arr.shape[0]) / 20.0, 1.0)
    return out.astype(np.float32)


def _get_autocast(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    if hasattr(torch, "autocast"):
        return torch.autocast(device_type=device.type, enabled=True)
    if device.type == "cuda" and hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"):
        return torch.cuda.amp.autocast(enabled=True)
    return nullcontext()


def normalize_site(site_value) -> str:
    try:
        return str(int(float(site_value)))
    except (TypeError, ValueError):
        return str(site_value).strip()


def normalize_chain(chain_value) -> Optional[str]:
    if chain_value is None:
        return None
    chain_str = str(chain_value).strip()
    if not chain_str or chain_str.lower() == "nan":
        return None
    return chain_str


def make_exact_key(protein_id: str, chain: str, site: str) -> str:
    return f"{protein_id}|{chain}|{site}"


def make_loose_key(protein_id: str, site: str) -> str:
    return f"{protein_id}|{site}"


def parse_graph_filename(filename: str) -> Optional[Tuple[str, str, str]]:
    match = re.match(r"^hbond_sidechain_(?P<protein_id>.+?)_(?P<chain>[^_]+)_(?P<site>\d+)\.csv$", filename)
    if not match:
        return None
    return match.group("protein_id"), match.group("chain"), match.group("site")


def read_sequence_dataframe(pos_csv: str, neg_csv: str, seq_col: str, id_col: str, site_col: str, chain_col: str) -> pd.DataFrame:
    pos = pd.read_csv(pos_csv)
    neg = pd.read_csv(neg_csv)
    required_cols = {seq_col, id_col, site_col}
    for df_name, df in [("pos_csv", pos), ("neg_csv", neg)]:
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"{df_name} missing required columns: {missing}")

    pos = pos.copy()
    neg = neg.copy()
    pos["label"] = 1
    neg["label"] = 0
    df = pd.concat([pos, neg], ignore_index=True)
    df = df.dropna(subset=[seq_col, id_col, site_col]).reset_index(drop=True)
    df["protein_id"] = df[id_col].astype(str).str.strip()
    df["site_norm"] = df[site_col].map(normalize_site)
    df["chain_norm"] = df[chain_col].map(normalize_chain) if chain_col in df.columns else None
    df["sequence"] = df[seq_col].astype(str)
    return df


class ProteinGraphBuilder:
    def __init__(self, pdb_dir: str, dist_threshold: float):
        self.pdb_dir = pdb_dir
        self.dist_threshold = dist_threshold
        self.scaler = StandardScaler()
        self.edge_scaler = StandardScaler()
        self.parser = PDBParser(QUIET=True)
        self.skip_reasons: Dict[str, int] = {}

    def _skip(self, reason: str) -> None:
        self.skip_reasons[reason] = self.skip_reasons.get(reason, 0) + 1

    def _get_pdb_path(self, protein_id: str) -> Optional[str]:
        candidates = [
            os.path.join(self.pdb_dir, f"{protein_id}.pdb"),
            os.path.join(self.pdb_dir, f"{protein_id}.PDB"),
            os.path.join(self.pdb_dir, f"{protein_id.split('-')[0]}.pdb"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return None

    def _get_pdb_residue_info(self, protein_id: str) -> Optional[Dict[Tuple[str, int], dict]]:
        pdb_path = self._get_pdb_path(protein_id)
        if pdb_path is None:
            return None
        residue_info: Dict[Tuple[str, int], dict] = {}
        try:
            structure = self.parser.get_structure(protein_id, pdb_path)
            for model in structure:
                for chain in model:
                    chain_id = str(chain.get_id()).strip()
                    for residue in chain:
                        if "CA" not in residue:
                            continue
                        residue_info[(chain_id, int(residue.get_id()[1]))] = {
                            "coord": residue["CA"].get_coord(),
                            "atom_features": atom_summary_features(residue),
                            "plddt": float(residue["CA"].get_bfactor()),
                        }
            return residue_info
        except Exception as exc:
            print(f"[WARN] Failed to parse PDB {pdb_path}: {exc}")
            return None

    def fit_scaler(self, csv_paths: List[str]) -> None:
        values = []
        for path in csv_paths:
            try:
                df = pd.read_csv(path)
                df.columns = [col.lower() for col in df.columns]
                if {"density", "acc"}.issubset(df.columns) and not df.empty:
                    values.append(df[["density", "acc"]].values)
            except Exception:
                continue
        if not values:
            raise RuntimeError("No valid graph CSV files found for scaler fitting.")
        if CFG.normalize_graph_node_features:
            self.scaler.fit(np.vstack(values))

    def build_graph(self, csv_path: str, label: int) -> Optional[Data]:
        filename = os.path.basename(csv_path)
        parsed = parse_graph_filename(filename)
        if parsed is None:
            self._skip("bad_graph_filename")
            return None

        protein_id, chain, site = parsed
        residue_info = self._get_pdb_residue_info(protein_id)
        if not residue_info:
            self._skip("missing_or_unreadable_pdb")
            return None

        try:
            df = pd.read_csv(csv_path)
            df.columns = [col.lower() for col in df.columns]
            required = {"resname", "resnum", "density", "acc", "hbond_type"}
            if not required.issubset(df.columns):
                self._skip("missing_required_columns")
                return None
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.dropna(subset=["resnum", "density", "acc", "hbond_type"]).reset_index(drop=True)
            if df.empty:
                self._skip("empty_after_required_feature_filter")
                return None

            df["chain_key"] = df["chain"].astype(str).str.strip() if "chain" in df.columns else chain
            matched_info = [residue_info.get((chain_id, int(res_num))) for chain_id, res_num in zip(df["chain_key"], df["resnum"])]
            df["coord"] = [item["coord"] if item is not None else None for item in matched_info]
            df["atom_features"] = [item["atom_features"] if item is not None else None for item in matched_info]
            df["plddt"] = [item["plddt"] if item is not None else np.nan for item in matched_info]
            df = df.dropna(subset=["coord"]).reset_index(drop=True)
            if df.empty:
                self._skip("no_residue_coords_matched")
                return None

            xyz = np.vstack(df["coord"].values)
            atom_features = np.asarray(df["atom_features"].tolist(), dtype=np.float32)
            if atom_features.shape[0] != df.shape[0]:
                atom_features = np.zeros((df.shape[0], 7), dtype=np.float32)
            mean_plddt = float(np.nanmean(df["plddt"].astype(float).values))
            if not np.isfinite(mean_plddt):
                mean_plddt = 0.0
            structure_mask = 1.0 if (not CFG.use_plddt_structure_mask or mean_plddt >= CFG.structure_plddt_threshold) else 0.0
            node_cont_raw = df[["density", "acc"]].values.astype(np.float32)
            if CFG.normalize_graph_node_features:
                node_cont = self.scaler.transform(node_cont_raw)
            else:
                node_cont = node_cont_raw
            hbond_raw = np.clip(df["hbond_type"].astype(int).values, 0, 3)
            hbond_onehot = np.eye(4, dtype=np.float32)[hbond_raw]
            aa_features = amino_acid_features(df["resname"].tolist())
            res_idx = df["resnum"].astype(int).values
            site_int = int(site)
            center = xyz.mean(axis=0, keepdims=True)
            centered_xyz = (xyz - center) / max(float(CFG.graph_dist_threshold), 1.0)
            site_coord = xyz[np.argmin(np.abs(res_idx - site_int))]
            site_dist = np.linalg.norm(xyz - site_coord.reshape(1, 3), axis=1, keepdims=True) / max(float(CFG.graph_dist_threshold), 1.0)
            seq_offset = ((res_idx - site_int).astype(np.float32) / max(float(CFG.max_len), 1.0)).reshape(-1, 1)
            abs_seq_offset = np.abs(seq_offset)
            site_indicator = (res_idx == site_int).astype(np.float32).reshape(-1, 1)
            geom_features = np.hstack([centered_xyz, site_dist, seq_offset, abs_seq_offset, site_indicator]).astype(np.float32)
            node_features = np.hstack([node_cont, hbond_onehot, aa_features, atom_features, geom_features]).astype(np.float32)

            if len(df) == 1:
                edge_index = torch.zeros((2, 0), dtype=torch.long)
                edge_attr = torch.zeros((0, CFG.graph_edge_dim), dtype=torch.float32)
            else:
                dist_matrix = squareform(pdist(xyz))
                row, col = np.where((dist_matrix < self.dist_threshold) & (dist_matrix > 0))
                edge_index = torch.tensor(np.array([row, col]), dtype=torch.long)
                edge_attrs = []
                rbf_centers = np.linspace(2.0, float(self.dist_threshold), 6)
                rbf_width = max(float(self.dist_threshold) / 6.0, 1e-6)
                for i, j in zip(row, col):
                    raw_dist = float(dist_matrix[i, j])
                    inv_dist = 1.0 / (raw_dist + 1e-6)
                    seq_dist = abs(res_idx[i] - res_idx[j])
                    seq_dist_value = np.log1p(float(seq_dist)) if CFG.edge_seq_dist_log1p else float(seq_dist)
                    rbf = np.exp(-((raw_dist - rbf_centers) ** 2) / (2.0 * rbf_width ** 2))
                    edge_attrs.append([inv_dist, seq_dist_value, 1.0 if seq_dist == 1 else 0.0, raw_dist / max(float(self.dist_threshold), 1.0), *rbf.tolist()])
                edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)

            graph = Data(
                x=torch.tensor(node_features, dtype=torch.float32),
                edge_index=edge_index,
                edge_attr=edge_attr,
                pos=torch.tensor(xyz, dtype=torch.float32),
                structure_mask=torch.tensor([structure_mask], dtype=torch.float32),
                mean_plddt=torch.tensor([mean_plddt], dtype=torch.float32),
                y=torch.tensor([label], dtype=torch.float32),
            )
            graph.sample_key = make_exact_key(protein_id, chain, site)
            graph.sample_loose_key = make_loose_key(protein_id, site)
            graph.filename = filename
            graph.num_residues = int(df.shape[0])
            return graph
        except Exception:
            self._skip("exception_during_graph_build")
            return None


def collect_graphs(builder: ProteinGraphBuilder, pos_dir: str, neg_dir: str) -> Dict[str, Data]:
    pos_paths = [os.path.join(pos_dir, name) for name in os.listdir(pos_dir) if name.endswith(".csv")]
    neg_paths = [os.path.join(neg_dir, name) for name in os.listdir(neg_dir) if name.endswith(".csv")]
    builder.fit_scaler(pos_paths + neg_paths)

    graph_by_key: Dict[str, Data] = {}
    duplicate_keys = set()
    for label, paths in [(1, pos_paths), (0, neg_paths)]:
        for path in tqdm(paths, desc=f"Build graphs label={label}"):
            graph = builder.build_graph(path, label)
            if graph is None:
                continue
            if graph.sample_key in graph_by_key:
                duplicate_keys.add(graph.sample_key)
            graph_by_key[graph.sample_key] = graph

    if duplicate_keys:
        raise ValueError(f"Duplicate graph keys found: {sorted(list(duplicate_keys))[:5]}")
    if not graph_by_key:
        print(f"[ERROR] Graph build skip reasons: {builder.skip_reasons}")
        raise RuntimeError("No valid graphs were built from the graph directories.")

    if CFG.normalize_graph_edge_features:
        edge_values = []
        for graph in graph_by_key.values():
            if graph.edge_attr.numel() > 0:
                edge_values.append(graph.edge_attr.cpu().numpy())
        if edge_values:
            stacked = np.vstack(edge_values)
            builder.edge_scaler.fit(stacked[:, :2])
            for graph in graph_by_key.values():
                if graph.edge_attr.numel() == 0:
                    continue
                edge_attr_np = graph.edge_attr.cpu().numpy()
                edge_attr_np[:, :2] = builder.edge_scaler.transform(edge_attr_np[:, :2])
                graph.edge_attr = torch.tensor(edge_attr_np, dtype=torch.float32)
    return graph_by_key


def summarize_graph_collection(graphs_by_exact: Dict[str, Data]) -> dict:
    num_graphs = len(graphs_by_exact)
    node_counts = []
    edge_counts = []
    undirected_edge_counts = []
    directed_densities = []
    undirected_densities = []
    node_feature_values = []
    edge_feature_values = []
    mean_plddt_values = []
    structure_mask_values = []
    anomalies = []

    for key, graph in graphs_by_exact.items():
        num_nodes = int(graph.x.shape[0])
        num_edges = int(graph.edge_index.shape[1])
        num_undirected_edges = num_edges // 2
        node_counts.append(num_nodes)
        edge_counts.append(num_edges)
        undirected_edge_counts.append(num_undirected_edges)
        if graph.x.numel() > 0:
            node_feature_values.append(graph.x.cpu().numpy())
        if graph.edge_attr.numel() > 0:
            edge_feature_values.append(graph.edge_attr.cpu().numpy())
        if hasattr(graph, "mean_plddt"):
            mean_plddt_values.append(float(graph.mean_plddt.view(-1)[0].item()))
        if hasattr(graph, "structure_mask"):
            structure_mask_values.append(float(graph.structure_mask.view(-1)[0].item()))

        reasons = []
        if num_nodes < 3:
            reasons.append("too_few_nodes")
        if num_edges == 0:
            reasons.append("no_edges")
        if graph.x.numel() > 0 and not np.isfinite(graph.x.cpu().numpy()).all():
            reasons.append("non_finite_node_features")
        if graph.edge_attr.numel() > 0 and not np.isfinite(graph.edge_attr.cpu().numpy()).all():
            reasons.append("non_finite_edge_features")
        directed_density = (num_edges / max(num_nodes * (num_nodes - 1), 1)) if num_nodes > 1 else 0.0
        undirected_density = (2.0 * num_undirected_edges / max(num_nodes * (num_nodes - 1), 1)) if num_nodes > 1 else 0.0
        directed_densities.append(directed_density)
        undirected_densities.append(undirected_density)
        if undirected_density > 0.8:
            reasons.append("very_dense_graph")
        if num_nodes > 200:
            reasons.append("very_large_graph")
        if reasons:
            anomalies.append({
                "sample_key": key,
                "filename": getattr(graph, "filename", ""),
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "num_undirected_edges": num_undirected_edges,
                "directed_density": directed_density,
                "undirected_density": undirected_density,
                "reasons": reasons,
            })

    def _stats(values: List[int]) -> dict:
        if not values:
            return {"count": 0}
        arr = np.array(values)
        return {
            "count": int(arr.size),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)),
        }

    report = {
        "num_graphs": num_graphs,
        "node_count_stats": _stats(node_counts),
        "edge_count_stats_directed": _stats(edge_counts),
        "edge_count_stats_undirected": _stats(undirected_edge_counts),
        "density_stats_directed": _stats(directed_densities),
        "density_stats_undirected": _stats(undirected_densities),
        "anomaly_count": len(anomalies),
        "anomalies_preview": anomalies[:50],
    }

    if node_feature_values:
        node_arr = np.vstack(node_feature_values)
        report["node_feature_summary"] = {
            "mean": node_arr.mean(axis=0).tolist(),
            "std": node_arr.std(axis=0).tolist(),
            "min": node_arr.min(axis=0).tolist(),
            "max": node_arr.max(axis=0).tolist(),
        }
    if edge_feature_values:
        edge_arr = np.vstack(edge_feature_values)
        report["edge_feature_summary"] = {
            "mean": edge_arr.mean(axis=0).tolist(),
            "std": edge_arr.std(axis=0).tolist(),
            "min": edge_arr.min(axis=0).tolist(),
            "max": edge_arr.max(axis=0).tolist(),
        }
    if mean_plddt_values:
        plddt_arr = np.asarray(mean_plddt_values, dtype=np.float32)
        report["mean_plddt_stats"] = {
            "min": float(plddt_arr.min()),
            "max": float(plddt_arr.max()),
            "mean": float(plddt_arr.mean()),
            "median": float(np.median(plddt_arr)),
            "p10": float(np.percentile(plddt_arr, 10)),
        }
    if structure_mask_values:
        mask_arr = np.asarray(structure_mask_values, dtype=np.float32)
        report["structure_mask_summary"] = {
            "enabled": bool(CFG.use_plddt_structure_mask),
            "threshold": float(CFG.structure_plddt_threshold),
            "use_graph_count": int(mask_arr.sum()),
            "sequence_only_count": int((mask_arr == 0).sum()),
            "use_graph_ratio": float(mask_arr.mean()),
        }
    return report


def resolve_graph(graphs_by_exact: Dict[str, Data], graphs_by_loose: Dict[str, List[str]], protein_id: str, site: str, chain: Optional[str]) -> Optional[Data]:
    if chain is not None:
        graph = graphs_by_exact.get(make_exact_key(protein_id, chain, site))
        if graph is not None:
            return graph
    candidates = graphs_by_loose.get(make_loose_key(protein_id, site), [])
    if len(candidates) == 1:
        return graphs_by_exact[candidates[0]]
    return None


def align_samples_with_graphs(df: pd.DataFrame, graphs_by_exact: Dict[str, Data], strict: bool = True) -> pd.DataFrame:
    graphs_by_loose: Dict[str, List[str]] = {}
    for exact_key, graph in graphs_by_exact.items():
        graphs_by_loose.setdefault(graph.sample_loose_key, []).append(exact_key)

    matched_graphs = []
    unmatched_rows = []
    for idx, row in df.iterrows():
        graph = resolve_graph(graphs_by_exact, graphs_by_loose, row["protein_id"], row["site_norm"], row["chain_norm"])
        matched_graphs.append(graph)
        if graph is None:
            unmatched_rows.append(idx)

    if unmatched_rows and strict:
        preview = df.loc[unmatched_rows[:5], ["protein_id", "site_norm", "chain_norm"]].to_dict("records")
        raise ValueError(f"Failed to align {len(unmatched_rows)} sequence samples with graph view. Examples: {preview}")

    aligned = df.copy()
    aligned["graph"] = matched_graphs
    aligned = aligned.dropna(subset=["graph"]).reset_index(drop=True)
    if aligned.empty:
        raise RuntimeError("No aligned samples remain after graph matching.")
    print(f"[INFO] Aligned {len(aligned)}/{len(df)} sequence samples with graph view.")
    return aligned


def load_esm2_model(model_name: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    return tokenizer, model


def pad_features(features: np.ndarray, max_len: int) -> np.ndarray:
    num_samples, seq_len, dim = features.shape
    if seq_len == max_len:
        return features
    out = np.zeros((num_samples, max_len, dim), dtype=features.dtype)
    copy_len = min(seq_len, max_len)
    out[:, :copy_len, :] = features[:, :copy_len, :]
    return out


def extract_esm2_features(sequences: List[str], tokenizer, model, max_len: int, batch_size: int, device: torch.device) -> np.ndarray:
    features = []
    amp_enabled = bool(CFG.use_amp and device.type == "cuda")
    for start in tqdm(range(0, len(sequences), batch_size), desc="Extract ESM2"):
        batch = sequences[start:start + batch_size]
        inputs = tokenizer(
            batch,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=max_len + 2,
            return_tensors="pt",
        )
        inputs = {key: value.to(device, non_blocking=True) for key, value in inputs.items()}
        with torch.inference_mode():
            with _get_autocast(device, amp_enabled):
                outputs = model(**inputs)
                residue = outputs.last_hidden_state[:, 1:-1, :]
        features.append(residue.float().cpu().numpy())
    features_np = np.concatenate(features, axis=0)
    if features_np.shape[1] != max_len:
        features_np = pad_features(features_np, max_len=max_len)
    return np.nan_to_num(features_np, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


class MultiModalDataset(Dataset):
    def __init__(self, seq_features: np.ndarray, graphs: List[Data], labels: np.ndarray, augment: bool = False):
        self.seq_features = seq_features.astype(np.float32)
        self.graphs = graphs
        self.labels = labels.astype(np.float32)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        seq_x = self.seq_features[idx].copy()
        if self.augment:
            length, dim = seq_x.shape
            pad_mask = np.all(np.isclose(seq_x, 0.0), axis=1)
            if CFG.gauss_std > 0:
                noise = np.random.normal(0, CFG.gauss_std, size=seq_x.shape).astype(np.float32)
                noise[pad_mask] = 0.0
                seq_x = seq_x + noise
            if 0.0 < CFG.drop_prob < 1.0:
                keep_mask = np.random.rand(length) >= CFG.drop_prob
                keep_mask = np.logical_or(keep_mask, pad_mask)
                seq_x = seq_x * keep_mask[:, None].astype(np.float32)
            if CFG.max_crop_frac > 0:
                max_crop = max(1, int(length * CFG.max_crop_frac))
                crop = np.random.randint(-max_crop, max_crop + 1)
                if crop > 0:
                    seq_x = np.vstack([seq_x[crop:], np.zeros((crop, dim), dtype=np.float32)])
                elif crop < 0:
                    crop = -crop
                    seq_x = np.vstack([np.zeros((crop, dim), dtype=np.float32), seq_x[:-crop]])
            scale = 1.0 + np.random.uniform(-0.03, 0.03)
            seq_x = np.nan_to_num(seq_x * scale, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.tensor(seq_x, dtype=torch.float32), self.graphs[idx], torch.tensor(self.labels[idx], dtype=torch.float32)


def collate_multimodal(batch):
    seq_x, graphs, labels = zip(*batch)
    return torch.stack(seq_x, dim=0), Batch.from_data_list(list(graphs)), torch.stack(labels, dim=0)
