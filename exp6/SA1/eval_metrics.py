"""
Exp6 evaluation metrics.

实现 EXP6_PROPOSAL_v3.md §7.1 锁定的 5 个继承指标 + 公共工具:
  0. min_image_l2         — 公共工具,frac → Å,min-image 折叠
  1. hungarian_rmsd       — 主指标,与 Exp4 直接对比
  2. set_level_type_acc   — Exp3 教训核心,position-decoupled
  3. multiset_f1_macro    — 元素分布层面诊断 (majority class 检测)
  4. in_cutoff_counts     — Exp2 起监控
  5. close_pair_type_acc  — Exp3 末期建立的"可信配对"指标

公式逐字符服从 proposal §7.1,SA1 不许优化重写。

唯一对 proposal 的扩展: min_image_l2 支持任意 leading 维度的 broadcast
(matcher.py 的 batch-flatten layout 与 eval per-sample layout 都能复用同一函数)。
扩展严格保持公式语义不变 (diff = pred-gt; -round; * lengths; norm),只支持
矩阵化批处理。在使用现场行为与 proposal §7.1 原版完全一致。

NO_OBJECT_IDX 是模块级配置,使用前必须 set_no_object_idx() 注入,值 = N_NEIGHBOR_TYPES.
"""
import torch
from collections import Counter
from scipy.optimize import linear_sum_assignment

# ---------------------------------------------------------------------------
# Module-level NO_OBJECT_IDX. Set once after loading exp6_element_vocab.json.
# ---------------------------------------------------------------------------
NO_OBJECT_IDX = None  # int, must be set before any indicator is called


def set_no_object_idx(idx: int):
    """Initialize NO_OBJECT_IDX from vocab JSON. MUST be called before metrics."""
    global NO_OBJECT_IDX
    NO_OBJECT_IDX = int(idx)


# ---------------------------------------------------------------------------
# Public utility (proposal §7.1 公共工具函数)
# ---------------------------------------------------------------------------
def min_image_l2(pred, gt, lengths):
    """
    Min-image L2 distance, frac coord input, Cartesian Å output.

    Proposal §7.1 原版签名 (per-sample):
        pred: (M, 3), gt: (N, 3), lengths: (3,) → return (M, N) Å

    Ergonomic broadcast extension:
        pred: (..., M, 3), gt: (..., N, 3), lengths: (3,) → return (..., M, N) Å
    Useful for matcher.py batch-flatten layout. Semantics unchanged.

    Formula (verbatim from proposal §7.1):
        diff = pred[:, None] - gt[None, :]   # (M, N, 3) frac
        diff = diff - round(diff)            # min-image fold
        cart = diff * lengths                # (M, N, 3) Å
        return norm(cart, dim=-1)            # (M, N) Å
    """
    diff = pred.unsqueeze(-2) - gt.unsqueeze(-3)   # (..., M, N, 3) frac
    diff = diff - torch.round(diff)                # min-image fold
    cart = diff * lengths                          # (..., M, N, 3) Å
    return torch.norm(cart, dim=-1)                # (..., M, N) Å


# ---------------------------------------------------------------------------
# Indicator 1 — Hungarian RMSD (min-image)
# Proposal §7.1, lines 401-417 verbatim.
# ---------------------------------------------------------------------------
def hungarian_rmsd(pred_pos, pred_types_argmax, gt_pos, gt_types, lengths):
    """
    Returns: rmsd (scalar Å), matched_pairs (list of (pred_idx, gt_idx))

    pred 端先 filter no_object,gt 端不动 (proposal §7.1 注释).
    """
    assert NO_OBJECT_IDX is not None, "Call set_no_object_idx() before metrics"

    valid_pred_mask = (pred_types_argmax != NO_OBJECT_IDX)   # (Q,)
    if valid_pred_mask.sum() == 0:
        return float('inf'), []
    valid_pred_pos = pred_pos[valid_pred_mask]               # (k, 3)

    cost = min_image_l2(valid_pred_pos, gt_pos, lengths)     # (k, n) Å
    row, col = linear_sum_assignment(cost.detach().cpu().numpy())

    matched_dists = cost[row, col]                           # (min(k,n),)
    rmsd = torch.sqrt((matched_dists ** 2).mean()).item()
    return rmsd, list(zip(row.tolist(), col.tolist()))


# ---------------------------------------------------------------------------
# Indicator 2 — Set-Level TypeAcc (Exp3 教训核心)
# Proposal §7.1, lines 426-441 verbatim.
# ---------------------------------------------------------------------------
def set_level_type_acc(pred_types_argmax, gt_types):
    """
    Per-sample multiset 交集大小 / max(|pred|, |gt|),与坐标完全解耦.
    """
    assert NO_OBJECT_IDX is not None, "Call set_no_object_idx() before metrics"

    valid_pred = pred_types_argmax[pred_types_argmax != NO_OBJECT_IDX].tolist()
    gt_list = gt_types.tolist()

    pred_counter = Counter(valid_pred)
    gt_counter = Counter(gt_list)

    intersection = sum((pred_counter & gt_counter).values())
    denominator = max(len(valid_pred), len(gt_list))
    if denominator == 0:
        return 0.0
    return intersection / denominator


# ---------------------------------------------------------------------------
# Indicator 3 — Multiset F1 macro (majority class 诊断)
# Proposal §7.1, lines 448-484 verbatim.
# ---------------------------------------------------------------------------
def multiset_f1_macro(all_pred_types_list, all_gt_types_list, n_elements):
    """
    Dataset-level macro-F1 across element classes.

    Args:
        all_pred_types_list / all_gt_types_list: list of per-sample tensor.
            all_pred 已 filter 掉 no_object (caller 责任).
        n_elements: N_NEIGHBOR_TYPES (不含 no_object).

    Returns:
        scalar dataset-level macro F1.
    """
    f1_per_class = []
    for c in range(n_elements):
        tp = fp = fn = 0
        for pred, gt in zip(all_pred_types_list, all_gt_types_list):
            pc = Counter(pred.tolist())
            gc = Counter(gt.tolist())
            tp += min(pc.get(c, 0), gc.get(c, 0))
            fp += max(pc.get(c, 0) - gc.get(c, 0), 0)
            fn += max(gc.get(c, 0) - pc.get(c, 0), 0)
        if tp + fp == 0 or tp + fn == 0:
            continue                 # 类 c 在 dataset 不出现,跳过
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        if p + r == 0:
            f1_per_class.append(0.0)
        else:
            f1_per_class.append(2 * p * r / (p + r))

    if len(f1_per_class) == 0:
        return 0.0
    return sum(f1_per_class) / len(f1_per_class)


# ---------------------------------------------------------------------------
# Indicator 4 — pred_in_cutoff / true_in_cutoff
# Proposal §7.1, lines 492-507 verbatim.
# ---------------------------------------------------------------------------
def in_cutoff_counts(pred_pos, pred_types_argmax, gt_pos, eval_cutoff, lengths):
    """
    eval_cutoff: per-sample scalar (Å), from dataset's eval_cutoff field.
    Returns: (n_pred_in, n_true_in)
    """
    assert NO_OBJECT_IDX is not None, "Call set_no_object_idx() before metrics"

    valid_pred_mask = (pred_types_argmax != NO_OBJECT_IDX)
    valid_pred_pos = pred_pos[valid_pred_mask]
    pred_cart_dist = torch.norm(valid_pred_pos * lengths, dim=-1)
    n_pred_in = (pred_cart_dist <= eval_cutoff).sum().item()

    gt_cart_dist = torch.norm(gt_pos * lengths, dim=-1)
    n_true_in = (gt_cart_dist <= eval_cutoff).sum().item()

    return n_pred_in, n_true_in


# ---------------------------------------------------------------------------
# Indicator 5 — Close-pair TypeAcc (距离 < 0.5 Å)
# Proposal §7.1, lines 514-540 verbatim.
# ---------------------------------------------------------------------------
def close_pair_type_acc(pred_pos, pred_types_argmax, gt_pos, gt_types,
                        lengths, distance_threshold=0.5):
    """
    Hungarian 匹配后,只统计配对 cartesian 距离 < threshold (Å) 的对的 type
    命中率.
    """
    assert NO_OBJECT_IDX is not None, "Call set_no_object_idx() before metrics"

    rmsd, matched = hungarian_rmsd(pred_pos, pred_types_argmax, gt_pos,
                                    gt_types, lengths)
    if len(matched) == 0:
        return 0.0

    valid_pred_mask = (pred_types_argmax != NO_OBJECT_IDX)
    valid_pred_pos = pred_pos[valid_pred_mask]
    valid_pred_types = pred_types_argmax[valid_pred_mask]

    cost = min_image_l2(valid_pred_pos, gt_pos, lengths)

    correct = total = 0
    for pred_i, gt_j in matched:
        if cost[pred_i, gt_j] < distance_threshold:
            total += 1
            if valid_pred_types[pred_i].item() == gt_types[gt_j].item():
                correct += 1

    if total == 0:
        return 0.0
    return correct / total
