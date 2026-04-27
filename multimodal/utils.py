import json
import hashlib
import os
import random
from datetime import datetime

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    try:
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return 0.5


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    from sklearn.metrics import precision_recall_curve

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1s = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
    return float(thresholds[np.argmax(f1s)]) if len(f1s) > 0 else 0.5


def convert_ndarray_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {key: convert_ndarray_to_list(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [convert_ndarray_to_list(item) for item in obj]
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj


def save_metrics(metrics: dict, path: str) -> None:
    payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": convert_ndarray_to_list(metrics),
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def stable_hash(payload) -> str:
    normalized = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def file_fingerprint(path: str) -> dict:
    stat = os.stat(path)
    return {
        "path": os.path.abspath(path),
        "size": stat.st_size,
        "mtime": stat.st_mtime_ns,
    }


def collect_dir_fingerprints(root: str, suffix: str) -> list:
    fingerprints = []
    for current_root, _, filenames in os.walk(root):
        for filename in sorted(filenames):
            if filename.endswith(suffix):
                fingerprints.append(file_fingerprint(os.path.join(current_root, filename)))
    return fingerprints
