import os

import numpy as np
from scipy.optimize import linear_sum_assignment


def _load_rows(path):
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return np.empty((0, 3), dtype=np.float64)
    rows = np.loadtxt(path, ndmin=2)
    if rows.shape[1] < 3:
        raise ValueError(f"evaluation file must have 3 columns: {path}")
    return rows[:, :3].astype(np.float64, copy=False)


def evaluateDetection_py(res_fpath, gt_fpath, dist_thres, dataset_name=None):
    """Compute CLEAR detection metrics over the union of GT/detection frames."""
    ground_truth = _load_rows(gt_fpath)
    detections = _load_rows(res_fpath)
    if ground_truth.shape[0] == 0:
        return 0.0, 0.0, 0.0, 0.0

    frames = np.union1d(ground_truth[:, 0], detections[:, 0])
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    matched_quality = 0.0

    for frame in frames:
        gt_points = ground_truth[ground_truth[:, 0] == frame, 1:3]
        det_points = detections[detections[:, 0] == frame, 1:3]
        if len(gt_points) == 0:
            false_positives += len(det_points)
            continue
        if len(det_points) == 0:
            false_negatives += len(gt_points)
            continue

        distances = np.linalg.norm(
            gt_points[:, None, :] - det_points[None, :, :], axis=2
        )
        costs = distances.copy()
        costs[costs >= dist_thres] = 1e6
        gt_indices, det_indices = linear_sum_assignment(costs)
        matched_distances = distances[gt_indices, det_indices]
        valid = matched_distances < dist_thres
        matches = int(valid.sum())
        true_positives += matches
        false_negatives += len(gt_points) - matches
        false_positives += len(det_points) - matches
        if matches:
            matched_quality += np.sum(
                1.0 - matched_distances[valid] / dist_thres
            )

    total_gt = true_positives + false_negatives
    recall = 100.0 * true_positives / total_gt if total_gt else 0.0
    predicted = true_positives + false_positives
    precision = 100.0 * true_positives / predicted if predicted else 0.0
    moda = 100.0 * (
        1.0 - (false_negatives + false_positives) / total_gt
    ) if total_gt else 0.0
    moda = max(0.0, moda)
    modp = 100.0 * matched_quality / true_positives \
        if true_positives else 0.0
    return recall, precision, moda, modp
