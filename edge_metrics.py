import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, f1_score, normalized_mutual_info_score


def align_predicted_labels(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.size == 0:
        return y_pred
    n_class = int(max(y_true.max(), y_pred.max()) + 1)
    mat = np.zeros((n_class, n_class), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        mat[p, t] += 1
    row, col = linear_sum_assignment(mat.max() - mat)
    mapping = {int(r): int(c) for r, c in zip(row, col)}
    return np.asarray([mapping.get(int(p), int(p)) for p in y_pred], dtype=np.int64)


def clustering_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    aligned = align_predicted_labels(y_true, y_pred)
    return float(np.mean(aligned == y_true)) if y_true.size else 0.0


def evaluate_node_clustering(labels: np.ndarray, pred_y: np.ndarray) -> dict:
    aligned_pred = align_predicted_labels(labels, pred_y)
    return {
        "ACC": float(np.mean(aligned_pred == labels)) if labels.size else 0.0,
        "NMI": float(normalized_mutual_info_score(labels, pred_y)),
        "ARI": float(adjusted_rand_score(labels, pred_y)),
        "F1": float(f1_score(labels, aligned_pred, average="weighted")),
    }
