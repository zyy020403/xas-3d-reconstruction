# step5b_2_metrics_v2.py
# Step 5b v2 — Experiment 3 解耦诊断指标计算（val + test）
# ============================================================
# 基于 step5b_2_metrics.py，新增以下四类诊断指标：
#
#   指标 A — Set-Level TypeAcc（完全解耦坐标）
#     纯类型匈牙利匹配（cost = 0 if match else 1），计算 Top-1 / Top-3
#     反映 TypeClassifier 内禀准确率，与坐标误差无关
#
#   指标 B — Multiset F1
#     不看位置，只看元素计数分布的 Precision / Recall / F1
#
#   指标 C — 近配对 TypeAcc（坐标配对距离 < 0.5 Å）
#     在坐标匈牙利匹配的基础上，只统计距离 < 0.5Å 的配对对
#
#   指标 D — 分壳层准确率 vs 全猜O基线对比
#     第一壳层（≤2.5Å）  baseline=0.595
#     第二壳层（2.5-3.5Å） baseline=0.162
#     第三壳层（3.5-4.0Å） baseline=0.339
#
# ★ 输入：experiment3/step5b_v2/predictions_{val,test}.pt
# ★ 输出：experiment3/step5b_v2/metrics_report_val_test_v2.txt
# ★ L = 6.0（Exp3 固定）
# ============================================================

import os, sys, logging, warnings, json

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
warnings.filterwarnings("ignore")

# ── 路径常量 ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = r"C:\Users\T-Cat\Desktop\DiffCSP-main"
EXP3_ROOT    = os.path.join(PROJECT_ROOT, "experiment3")
STEP5B_DIR   = os.path.join(EXP3_ROOT, "step5b_v3")         # ★ 改为 step5b_v3
VOCAB_PATH   = os.path.join(EXP3_ROOT, "step3b", "elem_vocab.json")
L            = 6.0

# ── 壳层定义 ──────────────────────────────────────────────────────────────────
SHELL_BINS = [
    (0.0, 2.5,  "第一壳层 (≤2.5Å)"),
    (2.5, 3.5,  "第二壳层 (2.5-3.5Å)"),
    (3.5, 4.0,  "第三壳层 (3.5-4.0Å)"),
]

# ── 全猜O基线（O的原子序数=8；按壳层统计训练集中O的频率）────────────────────
SHELL_O_BASELINE = {
    "第一壳层 (≤2.5Å)":   0.595,
    "第二壳层 (2.5-3.5Å)": 0.162,
    "第三壳层 (3.5-4.0Å)": 0.339,
}


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def load_vocab(vocab_path):
    """返回 {str(Z): class_idx} 以及反向映射 {class_idx: Z}。"""
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    inv_vocab = {v: int(k) for k, v in vocab.items()}
    return vocab, inv_vocab


# ─────────────────────────────────────────────────────────────────────────────
# 指标 A：Set-Level TypeAcc（完全解耦坐标）
# ─────────────────────────────────────────────────────────────────────────────

def type_set_accuracy(pred_types, true_types, pred_logits=None, inv_vocab=None):
    """
    纯类型匈牙利匹配（0/1 代价矩阵），计算 Top-1 / Top-3 类型准确率。

    参数
    ----
    pred_types  : (20,) int array — argmax 预测的原子序数
    true_types  : (20,) int array — 真实原子序数
    pred_logits : (20, N_elem) float array or None — 类型 logits
    inv_vocab   : dict {class_idx: Z} or None

    返回
    ----
    top1_acc : float  — 纯类型匹配下的 Top-1 准确率
    top3_acc : float  — 纯类型匹配下的 Top-3 准确率（需要 logits）
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    pred_types = np.array(pred_types, dtype=np.int64)
    true_types = np.array(true_types, dtype=np.int64)
    n = len(pred_types)

    # ── Top-1 代价矩阵：0 = 类型匹配，1 = 类型不匹配 ─────────────────────────
    cost1 = (pred_types[:, None] != true_types[None, :]).astype(np.float64)
    r1, c1 = linear_sum_assignment(cost1)
    top1_acc = float((pred_types[r1] == true_types[c1]).mean())

    # ── Top-3 代价矩阵：0 = true_type 在 top3 中，1 = 不在 ───────────────────
    top3_acc = float('nan')
    if pred_logits is not None and inv_vocab is not None:
        pred_logits = np.array(pred_logits, dtype=np.float64)
        # 每个预测位置的 top-3 原子序数集合
        top3_sets = []
        for i in range(n):
            top3_class = np.argsort(pred_logits[i])[-3:]
            top3_z     = set(inv_vocab.get(int(c), -1) for c in top3_class)
            top3_sets.append(top3_z)

        # cost3[i, j] = 0 if true_types[j] in top3_sets[i] else 1
        cost3 = np.ones((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                if true_types[j] in top3_sets[i]:
                    cost3[i, j] = 0.0

        r3, c3   = linear_sum_assignment(cost3)
        top3_acc = float((cost3[r3, c3] == 0.0).mean())

    return top1_acc, top3_acc


# ─────────────────────────────────────────────────────────────────────────────
# 指标 B：Multiset F1
# ─────────────────────────────────────────────────────────────────────────────

def multiset_f1(pred_types, true_types):
    """
    基于元素计数分布（多重集合）计算 Precision / Recall / F1。
    与坐标和匹配顺序完全无关。

    Intersection = sum_z  min(count_pred(z), count_true(z))
    Precision = intersection / len(pred_types)
    Recall    = intersection / len(true_types)
    F1        = 2 * P * R / (P + R)

    参数
    ----
    pred_types : (20,) int — 预测原子序数（来自 argmax 或已匹配）
    true_types : (20,) int — 真实原子序数

    返回
    ----
    precision, recall, f1 : float
    """
    import numpy as np
    from collections import Counter

    pred_c = Counter(int(z) for z in pred_types)
    true_c = Counter(int(z) for z in true_types)

    all_z     = set(pred_c.keys()) | set(true_c.keys())
    intersect = sum(min(pred_c.get(z, 0), true_c.get(z, 0)) for z in all_z)

    total_pred = len(pred_types)
    total_true = len(true_types)

    precision = float(intersect / total_pred) if total_pred > 0 else 0.0
    recall    = float(intersect / total_true) if total_true > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return precision, recall, f1


# ─────────────────────────────────────────────────────────────────────────────
# 指标 C：近配对 TypeAcc（坐标配对距离 < 0.5 Å）
# ─────────────────────────────────────────────────────────────────────────────

def near_match_type_acc(row_ind, col_ind, pred_frac, true_frac,
                        pred_types, true_types, L=6.0, thresh_ang=0.5):
    """
    在坐标匈牙利匹配结果的基础上，
    只统计配对距离 < thresh_ang Å 的配对，计算这部分的类型准确率。

    参数
    ----
    row_ind, col_ind : 匈牙利匹配索引（来自 evaluate_sample）
    pred_frac  : (20, 3) float — 预测分数坐标 [-0.5, 0.5]
    true_frac  : (20, 3) float — 真实分数坐标 [-0.5, 0.5]
    pred_types : (20,) int  — 预测原子序数
    true_types : (20,) int  — 真实原子序数
    L          : float      — 虚拟晶格边长
    thresh_ang : float      — 距离阈值（Å）

    返回
    ----
    near_acc : float  — 近配对类型准确率（若无近配对则为 nan）
    near_n   : int    — 近配对数量
    """
    import numpy as np

    near_correct = 0
    near_total   = 0

    for ri, ci in zip(row_ind, col_ind):
        delta = pred_frac[ri] - true_frac[ci]
        delta -= np.round(delta)
        dist  = float(np.linalg.norm(delta * L))
        if dist < thresh_ang:
            near_total += 1
            if int(pred_types[ri]) == int(true_types[ci]):
                near_correct += 1

    near_acc = (float(near_correct / near_total)
                if near_total > 0 else float('nan'))
    return near_acc, near_total


# ─────────────────────────────────────────────────────────────────────────────
# 核心评估函数（含坐标匈牙利匹配）
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_sample(pred_frac, pred_types, true_frac, true_types, eval_cutoff,
                    pred_logits=None, inv_vocab=None, L=6.0):
    """
    20 对原子最小镜像匈牙利匹配，返回坐标+类型综合评估 dict。

    返回 dict 新增键（v2）：
      type_acc_top3      : float — 匈牙利后 Top-3 准确率
      true_dists         : (20,) ndarray — 真实原子到 Fe 原点距离（Å）
      row_ind, col_ind   : 匈牙利匹配索引
      matched_dists      : (20,) ndarray — 每个配对的坐标距离（Å）

    坐标约定（Exp3 v3）：
      pred_frac, true_frac ∈ [-0.5, 0.5]
      Fe 原点 = (0, 0, 0)
      最小镜像：delta -= round(delta)；dist = ||delta * L||
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    pred_frac  = np.array(pred_frac,  dtype=np.float64)
    true_frac  = np.array(true_frac,  dtype=np.float64)
    pred_types = np.array(pred_types, dtype=np.int64)
    true_types = np.array(true_types, dtype=np.int64)
    n = pred_frac.shape[0]  # 20

    # ── 20×20 坐标代价矩阵（最小镜像距离）─────────────────────────────────────
    cost_matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        delta = pred_frac[i] - true_frac
        delta -= np.round(delta)
        cost_matrix[i] = np.linalg.norm(delta * L, axis=1)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # ── 每个配对的距离（Å）────────────────────────────────────────────────────
    matched_dists = np.zeros(n, dtype=np.float64)
    for k, (ri, ci) in enumerate(zip(row_ind, col_ind)):
        delta = pred_frac[ri] - true_frac[ci]
        delta -= np.round(delta)
        matched_dists[k] = float(np.linalg.norm(delta * L))

    rmsd = float(np.sqrt(np.mean(matched_dists ** 2)))

    # ── Type Accuracy Top-1（坐标匹配后）──────────────────────────────────────
    type_acc_top1 = float((pred_types[row_ind] == true_types[col_ind]).mean())

    # ── Type Accuracy Top-3（坐标匹配后，需要 logits）─────────────────────────
    type_acc_top3 = float('nan')
    if pred_logits is not None and inv_vocab is not None:
        top3_list = []
        for ri, ci in zip(row_ind, col_ind):
            logits_i   = pred_logits[ri]
            top3_class = np.argsort(logits_i)[-3:]
            top3_z     = [inv_vocab.get(int(c), -1) for c in top3_class]
            top3_list.append(float(int(true_types[ci]) in top3_z))
        type_acc_top3 = float(np.mean(top3_list))

    # ── pred/true 在 eval_cutoff 内的原子数 ───────────────────────────────────
    pred_dists = np.linalg.norm(pred_frac * L, axis=1)
    true_dists = np.linalg.norm(true_frac * L, axis=1)
    n_pred_in  = int((pred_dists <= eval_cutoff).sum())
    n_true_in  = int((true_dists <= eval_cutoff).sum())

    return {
        'rmsd':           rmsd,
        'type_acc':       type_acc_top1,
        'type_acc_top3':  type_acc_top3,
        'n_pred_in':      n_pred_in,
        'n_true_in':      n_true_in,
        'eval_cutoff':    eval_cutoff,
        'true_dists':     true_dists,       # (20,) 真实原子到 Fe 原点距离
        'matched_dists':  matched_dists,    # (20,) 每配对距离（Å）
        'row_ind':        row_ind,
        'col_ind':        col_ind,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 壳层统计（指标 D）
# ─────────────────────────────────────────────────────────────────────────────

def shell_stats(all_results, shell_bins, L=6.0):
    """
    按 true atom 到 Fe 原点距离对匹配对分壳层统计。
    需要结果 dict 中保存有 _pred_frac、_true_frac、_pred_types、_true_types 等字段。
    """
    import numpy as np

    bin_data = {
        label: {
            'rmsd_sq_sum': 0.0, 'n_matched': 0,
            'type1_correct': 0, 'type3_correct': 0, 'type_valid': 0
        }
        for _, _, label in shell_bins
    }

    for res in all_results:
        true_dists  = res['true_dists']
        row_ind     = res['row_ind']
        col_ind     = res['col_ind']
        pred_frac   = res.get('_pred_frac')
        true_frac   = res.get('_true_frac')
        pred_types  = res.get('_pred_types')
        true_types  = res.get('_true_types')
        pred_logits = res.get('_pred_logits')
        inv_vocab   = res.get('_inv_vocab')

        if pred_frac is None:
            continue

        for ri, ci in zip(row_ind, col_ind):
            d_true = float(true_dists[ci])  # true atom distance to Fe origin

            shell_label = None
            for lo, hi, lbl in shell_bins:
                if lo <= d_true < hi:
                    shell_label = lbl
                    break
            if shell_label is None:
                continue

            bd = bin_data[shell_label]

            delta = pred_frac[ri] - true_frac[ci]
            delta -= np.round(delta)
            bd['rmsd_sq_sum'] += float(np.sum((delta * L) ** 2))
            bd['n_matched']   += 1

            if int(pred_types[ri]) == int(true_types[ci]):
                bd['type1_correct'] += 1

            if pred_logits is not None and inv_vocab is not None:
                logits_i   = pred_logits[ri]
                top3_class = np.argsort(logits_i)[-3:]
                top3_z     = [inv_vocab.get(int(c), -1) for c in top3_class]
                if int(true_types[ci]) in top3_z:
                    bd['type3_correct'] += 1
                bd['type_valid'] += 1

    stats = []
    for lo, hi, label in shell_bins:
        bd  = bin_data[label]
        n   = bd['n_matched']
        if n == 0:
            stats.append({'label': label, 'n': 0,
                          'rmsd': float('nan'), 'type1': float('nan'),
                          'type3': float('nan')})
            continue
        rmsd  = float(np.sqrt(bd['rmsd_sq_sum'] / n))
        type1 = float(bd['type1_correct'] / n)
        type3 = (float(bd['type3_correct'] / bd['type_valid'])
                 if bd['type_valid'] > 0 else float('nan'))
        stats.append({'label': label, 'n': n,
                      'rmsd': rmsd, 'type1': type1, 'type3': type3})
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# 子群统计（继承 v5）
# ─────────────────────────────────────────────────────────────────────────────

def subgroup_stats(results, key, bins):
    import numpy as np
    groups = []
    for lo, hi, label in bins:
        sub = [r for r in results
               if (lo is None or r[key] >= lo) and
                  (hi is None or r[key] < hi)]
        if not sub:
            groups.append((label, 0, float('nan'), float('nan')))
            continue
        groups.append((label, len(sub),
                       float(np.mean([r['rmsd']      for r in sub])),
                       float(np.mean([r['type_acc']  for r in sub]))))
    return groups


# ─────────────────────────────────────────────────────────────────────────────
# 主评估函数
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(pred_path, split_name, report_lines, vocab, inv_vocab):
    import numpy as np
    import torch
    logger = logging.getLogger(__name__)

    if not os.path.exists(pred_path):
        logger.error(f"❌ 找不到 {pred_path}")
        report_lines.append(f"\n❌ {split_name}: 文件不存在 {pred_path}")
        return

    preds      = torch.load(pred_path, map_location="cpu")
    n          = len(preds['mp_id'])
    has_logits = 'pred_type_logits' in preds

    logger.info(f"\n计算 {split_name} 集指标（{n} 样本, L={L}, 最小镜像匈牙利匹配）...")
    logger.info(f"  pred_type_logits: {'✅' if has_logits else '❌（跳过 Top-3 系列指标）'}")

    results = []
    skipped = 0

    for i in range(n):
        pf  = preds['pred_frac_coords'][i].numpy()   # (20, 3)
        pt  = preds['pred_atom_types'][i].numpy()    # (20,)
        tf  = preds['true_frac_coords'][i].numpy()   # (20, 3)
        tt  = preds['true_atom_types'][i].numpy()    # (20,)
        ec  = float(preds['eval_cutoff'][i])

        if pf.shape[0] != 20 or tf.shape[0] != 20:
            skipped += 1
            continue

        pl = preds['pred_type_logits'][i].numpy() if has_logits else None

        # ── 坐标匈牙利匹配评估（继承） ────────────────────────────────────
        r = evaluate_sample(pf, pt, tf, tt, ec,
                            pred_logits=pl, inv_vocab=inv_vocab, L=L)

        # ── 指标 A：Set-Level TypeAcc（纯类型匹配）─────────────────────────
        t1_set, t3_set = type_set_accuracy(pt, tt, pl, inv_vocab)
        r['type_acc_top1_set'] = t1_set
        r['type_acc_top3_set'] = t3_set

        # ── 指标 B：Multiset F1 ────────────────────────────────────────────
        prec, rec, f1 = multiset_f1(pt, tt)
        r['multiset_precision'] = prec
        r['multiset_recall']    = rec
        r['multiset_f1']        = f1

        # ── 指标 C：近配对 TypeAcc（dist < 0.5Å）──────────────────────────
        near_acc, near_n = near_match_type_acc(
            r['row_ind'], r['col_ind'],
            pf, tf, pt, tt, L=L, thresh_ang=0.5
        )
        r['near_type_acc'] = near_acc
        r['near_n']        = near_n

        # 保存原始数据供壳层统计（指标 D）
        r['_pred_frac']   = pf
        r['_true_frac']   = tf
        r['_pred_types']  = pt
        r['_true_types']  = tt
        r['_pred_logits'] = pl
        r['_inv_vocab']   = inv_vocab

        results.append(r)

    logger.info(f"  有效样本：{len(results)}/{n}（跳过 {skipped}）")

    if not results:
        report_lines.append(f"\n⚠ {split_name}: 无有效样本")
        return

    # ── 汇总数组 ───────────────────────────────────────────────────────────
    import numpy as np

    rmsds      = np.array([r['rmsd']          for r in results])
    type_accs  = np.array([r['type_acc']       for r in results])
    type_accs3 = np.array([r['type_acc_top3']  for r in results
                           if not np.isnan(r['type_acc_top3'])])
    n_pred_in  = np.array([r['n_pred_in']      for r in results])
    n_true_in  = np.array([r['n_true_in']      for r in results])

    # 指标 A
    t1_set_arr = np.array([r['type_acc_top1_set'] for r in results])
    t3_set_arr = np.array([r['type_acc_top3_set'] for r in results
                           if not np.isnan(r['type_acc_top3_set'])])

    # 指标 B
    prec_arr  = np.array([r['multiset_precision'] for r in results])
    rec_arr   = np.array([r['multiset_recall']    for r in results])
    f1_arr    = np.array([r['multiset_f1']         for r in results])

    # 指标 C
    near_valid = [r for r in results if not np.isnan(r['near_type_acc'])]
    near_acc_arr = np.array([r['near_type_acc'] for r in near_valid]) \
        if near_valid else np.array([])
    total_near_n = sum(r['near_n'] for r in results)

    # ── 验收标准 ──────────────────────────────────────────────────────────
    rb       = L / 2 * (3 / 5) ** 0.5    # 随机基线 ≈ 2.32 Å
    rmsd_ok  = rmsds.mean() <= 1.6
    type1_ok = type_accs.mean() >= 0.40
    type3_ok = (type_accs3.mean() >= 0.65) if len(type_accs3) > 0 else False
    in_cut_ok = n_pred_in.mean() >= 15.0

    lines = [
        "",
        f"{'='*70}",
        f"=== {split_name} Set Metrics — Exp3 Step5b v2 (Step4f ckpt, L={L}Å) ===",
        f"{'='*70}",
        f"N_samples        : {len(results)}",
        "",
        f"┌─── 核心评估指标（验收标准）─────────────────────────────────────────┐",
        "",
        f"  [坐标]",
        f"  RMSD (Å)        : mean={rmsds.mean():.4f}  "
            f"median={np.median(rmsds):.4f}  std={rmsds.std():.4f}"
            f"  {'✅' if rmsd_ok else '❌'} (目标 ≤ 1.6Å)",
        "",
        f"  [类型 — 坐标匹配后]",
        f"  Type Acc Top-1  : mean={type_accs.mean():.4f}  "
            f"median={np.median(type_accs):.4f}"
            f"  {'✅' if type1_ok else '❌'} (目标 ≥ 0.40)",
    ]

    if len(type_accs3) > 0:
        lines.append(
            f"  Type Acc Top-3  : mean={type_accs3.mean():.4f}  "
            f"N_valid={len(type_accs3)}"
            f"  {'✅' if type3_ok else '❌'} (目标 ≥ 0.65)"
        )
    else:
        lines.append("  Type Acc Top-3  : N/A（pred_type_logits 缺失）")

    lines += [
        "",
        f"  [原子密度（eval_cutoff 内）]",
        f"  pred_in_cutoff  : mean={n_pred_in.mean():.2f} / 20"
            f"  {'✅' if in_cut_ok else '❌'} (目标 ≥ 15)",
        f"  true_in_cutoff  : mean={n_true_in.mean():.2f} / 20",
        "",
        f"└────────────────────────────────────────────────────────────────────┘",
        "",
        f"┌─── 指标 A：Set-Level TypeAcc（完全解耦坐标）─────────────────────────┐",
        f"  纯类型匈牙利匹配，与坐标误差无关",
        f"  Set-Level Top-1 : mean={t1_set_arr.mean():.4f}  "
            f"median={np.median(t1_set_arr):.4f}",
    ]

    if len(t3_set_arr) > 0:
        lines.append(
            f"  Set-Level Top-3 : mean={t3_set_arr.mean():.4f}  "
            f"N_valid={len(t3_set_arr)}"
        )
    else:
        lines.append("  Set-Level Top-3 : N/A（pred_type_logits 缺失）")

    # 诊断提示：坐标误差对类型准确率的贡献
    delta_t1 = float(t1_set_arr.mean() - type_accs.mean())
    lines += [
        f"",
        f"  ▶ Set-Level Top-1 - 匈牙利后 Top-1 = {delta_t1:+.4f}",
        f"    （正值 = 坐标误差导致匹配错位，损失了这部分类型准确率）",
        f"└────────────────────────────────────────────────────────────────────┘",
        "",
        f"┌─── 指标 B：Multiset F1（元素计数分布，与位置无关）────────────────────┐",
        f"  Precision       : mean={prec_arr.mean():.4f}  median={np.median(prec_arr):.4f}",
        f"  Recall          : mean={rec_arr.mean():.4f}  median={np.median(rec_arr):.4f}",
        f"  F1              : mean={f1_arr.mean():.4f}  median={np.median(f1_arr):.4f}",
        f"└────────────────────────────────────────────────────────────────────┘",
        "",
        f"┌─── 指标 C：近配对 TypeAcc（坐标配对距离 < 0.5Å）─────────────────────┐",
    ]

    if len(near_acc_arr) > 0:
        lines += [
            f"  Near-match TypeAcc: mean={near_acc_arr.mean():.4f}  "
            f"N_samples_with_near={len(near_valid)}/{len(results)}",
            f"  Total near pairs  : {total_near_n}"
            f"  (avg per sample = {total_near_n/len(results):.1f})",
            f"  说明：坐标配对距离 < 0.5Å 的原子对中，类型预测准确率",
        ]
    else:
        lines.append("  无满足 dist < 0.5Å 的近配对（模型坐标误差较大）")

    lines.append(f"└────────────────────────────────────────────────────────────────────┘")

    # ── 子群：eval_cutoff ─────────────────────────────────────────────────────
    lines += ["", "── 子群：eval_cutoff ──"]
    for label, cnt, mr, mt in subgroup_stats(
            results, 'eval_cutoff',
            [(None, 3.0, "< 3.0 Å"), (3.0, 4.0, "3.0–4.0 Å")]):
        lines.append(
            f"  {label:12s}  N={cnt:5d}  RMSD={mr:.4f}  TypeAcc={mt:.4f}")

    # ── 子群：n_true_in ───────────────────────────────────────────────────────
    lines += ["", "── 子群：n_true_in（eval_cutoff 内真实原子数）──"]
    for label, cnt, mr, mt in subgroup_stats(
            results, 'n_true_in',
            [(None, 9,  "≤ 8"),
             (9,   15,  "9–14"),
             (15,  None,"15–20")]):
        lines.append(
            f"  {label:20s}  N={cnt:5d}  RMSD={mr:.4f}  TypeAcc={mt:.4f}")

    # ── 指标 D：分壳层统计 vs 全猜O基线 ─────────────────────────────────────
    lines += [
        "",
        f"┌─── 指标 D：分壳层准确率 vs 全猜O基线 ───────────────────────────────┐",
        f"  壳层定义：按匹配后 true atom 到 Fe 原点的最小镜像距离",
        "",
        f"  {'壳层':<22}  {'N_pairs':>8}  {'TypeAcc_Top1':>12}  {'全猜O_baseline':>14}  "
        f"{'超出baseline':>12}  {'TypeAcc_Top3':>12}",
        f"  {'-'*85}",
    ]

    shell_results = shell_stats(results, SHELL_BINS, L=L)
    for sr in shell_results:
        lbl      = sr['label']
        n_p      = sr['n']
        t1       = sr['type1']
        t3       = sr['type3']
        baseline = SHELL_O_BASELINE.get(lbl, float('nan'))

        if not np.isnan(t1) and not np.isnan(baseline):
            exceed = t1 - baseline
            exceed_str = f"{exceed:+.4f}"
            exceed_flag = "✅" if exceed > 0 else "❌"
        else:
            exceed_str = "  N/A  "
            exceed_flag = " "

        t1_str   = f"{t1:.4f}" if not np.isnan(t1) else "  N/A "
        t3_str   = f"{t3:.4f}" if not np.isnan(t3) else "  N/A "
        base_str = f"{baseline:.3f}" if not np.isnan(baseline) else "  N/A "

        lines.append(
            f"  {lbl:<22}  {n_p:>8d}  {t1_str:>12}  {base_str:>14}  "
            f"{exceed_str:>12} {exceed_flag}  {t3_str:>12}"
        )

    # 第二壳层关键诊断
    shell2 = next((s for s in shell_results
                   if "第二壳层" in s['label']), None)
    if shell2 and not np.isnan(shell2['type1']):
        shell2_ok = shell2['type1'] > SHELL_O_BASELINE["第二壳层 (2.5-3.5Å)"]
        lines += [
            f"",
            f"  ★ 关键诊断 — 第二壳层（2.5-3.5Å）类型准确率超出全猜O基线：",
            f"    TypeAcc={shell2['type1']:.4f}  baseline={SHELL_O_BASELINE['第二壳层 (2.5-3.5Å)']:.3f}"
            f"  {'✅ 已超出' if shell2_ok else '❌ 未超出'}",
        ]

    lines.append(f"└────────────────────────────────────────────────────────────────────┘")

    # ── 总结 ─────────────────────────────────────────────────────────────────
    all_ok = rmsd_ok and type1_ok and type3_ok and in_cut_ok
    _t3_mean_str = f"{type_accs3.mean():.4f}" if len(type_accs3) > 0 else "N/A"
    lines += [
        "",
        f"{'='*70}",
        f"随机基线 RMSD ≈ {rb:.2f} Å（[-L/2, L/2] 均匀分布，L={L}）",
        "",
        f"Exp3 Step5b v2 验收标准（Step4f checkpoint）：",
        f"  RMSD ≤ 1.6 Å           : {'✅' if rmsd_ok  else '❌'}"
            f"  (mean={rmsds.mean():.4f})",
        f"  Type Acc Top-1 ≥ 0.40  : {'✅' if type1_ok else '❌'}"
            f"  (mean={type_accs.mean():.4f})",
        f"  Type Acc Top-3 ≥ 0.65  : {'✅' if type3_ok else '❌'}"
            f"  (mean={_t3_mean_str})",
        f"  pred_in_cutoff ≥ 15    : {'✅' if in_cut_ok else '❌'}"
            f"  (mean={n_pred_in.mean():.2f})",
        "",
        f"{'✅ 所有主要验收指标达标！' if all_ok else '❌ 存在未达标指标（见上）'}",
        f"{'='*70}",
    ]

    for line in lines:
        logger.info(line)
    report_lines.extend(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import numpy as np
    logger = logging.getLogger(__name__)

    logger.info("=" * 70)
    logger.info("Step5b v2 — Exp3 解耦诊断指标计算（val + test）")
    logger.info("=" * 70)

    if not os.path.exists(VOCAB_PATH):
        logger.error(f"❌ 词表不存在：{VOCAB_PATH}")
        sys.exit(1)

    vocab, inv_vocab = load_vocab(VOCAB_PATH)
    logger.info(f"词表加载：{len(vocab)} 个元素类别")

    os.makedirs(STEP5B_DIR, exist_ok=True)

    report_lines = [
        "Step5b v3 Metrics Report — Experiment 3 (Step4f ckpt, epoch 57, dist-sort fix)",
        f"L={L} Å，坐标系 [-0.5, 0.5]，最小镜像匈牙利匹配",
        "新增指标：A=Set-Level TypeAcc  B=Multiset F1  C=近配对TypeAcc  D=分壳层vs全猜O",
        "=" * 70,
    ]

    compute_metrics(
        pred_path    = os.path.join(STEP5B_DIR, "predictions_val.pt"),
        split_name   = "Val",
        report_lines = report_lines,
        vocab        = vocab,
        inv_vocab    = inv_vocab,
    )

    compute_metrics(
        pred_path    = os.path.join(STEP5B_DIR, "predictions_test.pt"),
        split_name   = "Test",
        report_lines = report_lines,
        vocab        = vocab,
        inv_vocab    = inv_vocab,
    )

    report_path = os.path.join(STEP5B_DIR, "metrics_report_val_test_v3.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    logger.info(f"\n完整报告 → {report_path}")
    logger.info("=" * 70)
    logger.info("Step5b_2 v2 完成")
