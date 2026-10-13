import numpy as np


def affiliation_precision_recall(pred, gt):
    """
    Affiliation-based Precision and Recall
    论文: "Local Evaluation of Time Series Anomaly Detection Algorithms" (KDD 2022)
    """
    pred = pred.astype(int)
    gt = gt.astype(int)
    n = len(gt)

    gt_indices = np.where(gt == 1)[0]
    pred_indices = np.where(pred == 1)[0]

    if len(gt_indices) == 0:
        return 1.0, 1.0

    # 每个位置到最近真实异常的距离
    if len(gt_indices) > 0:
        dist_to_gt = np.abs(np.arange(n)[:, None] - gt_indices[None, :])
        min_dist_to_gt = np.min(dist_to_gt, axis=1)
    else:
        min_dist_to_gt = np.ones(n) * n

    # 每个位置到最近预测异常的距离
    if len(pred_indices) > 0:
        dist_to_pred = np.abs(np.arange(n)[:, None] - pred_indices[None, :])
        min_dist_to_pred = np.min(dist_to_pred, axis=1)
    else:
        min_dist_to_pred = np.ones(n) * n

    # Affiliation Precision: 预测异常点与真实异常的距离
    if np.sum(pred) > 0:
        scores = [np.exp(-min_dist_to_gt[t] / 1.0) for t in range(n) if pred[t] == 1]
        aff_precision = np.mean(scores)
    else:
        aff_precision = 0.0

    # Affiliation Recall: 真实异常点与预测异常的距离
    if len(gt_indices) > 0:
        scores = [np.exp(-min_dist_to_pred[t] / 1.0) for t in range(n) if gt[t] == 1]
        aff_recall = np.mean(scores)
    else:
        aff_recall = 1.0

    return aff_precision, aff_recall