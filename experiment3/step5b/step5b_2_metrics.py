# step5b_2_metrics.py
# Step 5b.2 — Experiment 3 指标计算（val + test）
# ============================================================
# 基于 step4b_4_compute_metrics.py (v5)，扩展以下两项：
#
#   1. Type Accuracy Top-3
#      从 predictions.pt 中的 pred_type_logits (20, N_elem) 计算，
#      反向映射 class_index → 原子序数后与真实类型比较。
#
#   2. 按壳层分组统计
#      以匈牙利匹配后的 true 原子到 Fe 原点（分数坐标原点）的最小镜像距离
#      作为壳层分组依据：
#        第一壳层 ≤ 2.5 Å
#        第二壳层 2.5–3.5 Å
#        第三壳层 3.5–4.0 Å
#      每壳层分别报告：样本数、匹配原子数均值、RMSD、Top-1 TypeAcc、Top-3 TypeAcc
#
# 其余逻辑（最小镜像匈牙利匹配、eval_cutoff 子群统计）完全继承 v5。
#
# ★ L = 6.0（Exp3 固定，Exp2 是 12.0）
# ★ 坐标系：[-0.5, 0.5]（Exp3 v3；最小镜像公式不变，通用于任意坐标系）
# ============================================================

import os, sys, logging, warnings, json

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
warnings.filterwarnings("ignore")

# ── 路径常量 ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = r"C:\Users\T-Cat\Desktop\DiffCSP-main"
EXP3_ROOT    = os.path.join(PROJECT_ROOT, "experiment3")
STEP5B_DIR   = os.path.join(EXP3_ROOT, "step5b")
VOCAB_PATH   = os.path.join(EXP3_ROOT, "step3b", "elem_vocab.json")
L            = 6.0

# ── 壳层定义 ──────────────────────────────────────────────────────────────────
SHELL_BINS = [
    (0.0, 2.5, "第一壳层 (≤2.5Å)"),
    (2.5, 3.5, "第二壳层 (2.5-3.5Å)"),
    (3.5, 4.0, "第三壳层 (3.5-4.0Å)"),
]


# ── 工具：加载词表 ─────────────────────────────────────────────────────────────

def load_vocab(vocab_path):
    """返回 {class_index: atomic_number} 的反向映射。"""
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)                  # {str(Z): class_idx}
    inv_vocab = {v: int(k) for k, v in vocab.items()}   # {class_idx: Z}
    return vocab, inv_vocab


# ── 核心评估函数 ───────────────────────────────────────────────────────────────

def evaluate_sample(pred_frac, pred_types, true_frac, true_types, eval_cutoff,
                    pred_logits=None, inv_vocab=None, L=6.0):
    """
    20 个预测原子 vs 20 个真实原子做最小镜像匈牙利匹配。

    坐标约定（Exp3 v3）：
      pred_frac, true_frac ∈ [-0.5, 0.5]
      Fe 原点对应 (0, 0, 0)
      最小镜像距离：delta = p - t; delta -= round(delta); dist = ||delta * L||

    返回 dict，新增键：
      type_acc_top3  : float，如果 pred_logits 不为 None
      true_dists     : ndarray (20,)，每个真实原子到 Fe 原点的最小镜像距离
      matched_row    : row_ind（预测侧匹配索引）
      matched_col    : col_ind（真实侧匹配索引）
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    pred_frac  = np.array(pred_frac,  dtype=np.float64)
    true_frac  = np.array(true_frac,  dtype=np.float64)
    pred_types = np.array(pred_types, dtype=np.int64)
    true_types = np.array(true_types, dtype=np.int64)

    n = pred_frac.shape[0]   # 固定 20

    # ── 20×20 最小镜像 cost matrix ────────────────────────────────────────────
    cost_matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        delta = pred_frac[i] - true_frac           # (20, 3)
        delta -= np.round(delta)
        cost_matrix[i] = np.linalg.norm(delta * L, axis=1)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # ── RMSD（匹配后最小镜像距离）────────────────────────────────────────────
    matched_dists_sq = []
    for ri, ci in zip(row_ind, col_ind):
        delta = pred_frac[ri] - true_frac[ci]
        delta -= np.round(delta)
        matched_dists_sq.append(np.sum((delta * L) ** 2))
    rmsd = float(np.sqrt(np.mean(matched_dists_sq)))

    # ── Type Accuracy Top-1 ───────────────────────────────────────────────────
    type_acc_top1 = float((pred_types[row_ind] == true_types[col_ind]).mean())

    # ── Type Accuracy Top-3（需要 logits）────────────────────────────────────
    type_acc_top3 = float('nan')
    if pred_logits is not None and inv_vocab is not None:
        # pred_logits: (20, N_elem) numpy array，对应匹配中的行（预测侧）
        # row_ind[i] 是预测侧第 i 个匹配的原子，col_ind[i] 是真实侧
        top3_acc_list = []
        for i, (ri, ci) in enumerate(zip(row_ind, col_ind)):
            logits_i = pred_logits[ri]              # (N_elem,) for predicted atom ri
            # Top-3 class indices（按 logit 从小到大排序，取最后三个）
            top3_class = np.argsort(logits_i)[-3:]  # (3,) class indices
            # 转换为原子序数
            top3_z     = [inv_vocab.get(int(c), -1) for c in top3_class]
            true_z_i   = int(true_types[ci])
            top3_acc_list.append(float(true_z_i in top3_z))
        type_acc_top3 = float(np.mean(top3_acc_list)) if top3_acc_list else float('nan')

    # ── pred/true in cutoff（最小镜像距离到 Fe 原点，即 norm(frac*L)）──────────
    # 坐标系已经是 [-0.5, 0.5]，所以直接用欧氏距离（最小镜像已内置）
    pred_dists = np.linalg.norm(pred_frac * L, axis=1)  # (20,)
    true_dists = np.linalg.norm(true_frac * L, axis=1)  # (20,)
    n_pred_in  = int((pred_dists <= eval_cutoff).sum())
    n_true_in  = int((true_dists <= eval_cutoff).sum())

    return {
        'rmsd':           rmsd,
        'type_acc':       type_acc_top1,
        'type_acc_top3':  type_acc_top3,
        'n_pred_in':      n_pred_in,
        'n_true_in':      n_true_in,
        'eval_cutoff':    eval_cutoff,
        'true_dists':     true_dists,    # (20,) all 20 true-atom distances
        'row_ind':        row_ind,
        'col_ind':        col_ind,
    }


# ── 壳层统计 ─────────────────────────────────────────────────────────────────

def shell_stats(all_results, shell_bins, L=6.0):
    """
    对所有样本的匹配对按 true atom 到 Fe 原点距离分组统计。

    参数
    ----
    all_results : list of dict（每个样本的 evaluate_sample 返回值）
    shell_bins  : list of (lo, hi, label)

    返回
    ----
    list of dict：每个 bin 的统计量
    """
    import numpy as np

    bin_data = {label: {'rmsd_sq_sum': 0.0, 'n_matched': 0,
                        'type1_correct': 0, 'type3_correct': 0,
                        'type_valid': 0}
                for _, _, label in shell_bins}

    for res in all_results:
        true_dists = res['true_dists']  # (20,) all true atoms
        row_ind    = res['row_ind']
        col_ind    = res['col_ind']
        # 对每个匹配对，以 true 原子的距离作为壳层归属
        # （col_ind[i] 是 true 侧索引）
        from scipy.optimize import linear_sum_assignment
        # row_ind, col_ind are already from evaluate_sample

        # Per-pair stats are not directly in result; reconstruct from res
        # We store matched pairwise metrics below:
        # matched_dist_sq[i] was computed in evaluate_sample but not stored.
        # Recompute per pair here for shell grouping.
        pred_frac  = res.get('_pred_frac')
        true_frac  = res.get('_true_frac')
        pred_types = res.get('_pred_types')
        true_types = res.get('_true_types')
        pred_logits = res.get('_pred_logits')
        inv_vocab   = res.get('_inv_vocab')

        if pred_frac is None:
            continue  # no per-pair data stored

        for i, (ri, ci) in enumerate(zip(row_ind, col_ind)):
            d_true = true_dists[ci]   # distance of the matched true atom

            # find which shell
            shell_label = None
            for lo, hi, lbl in shell_bins:
                if lo <= d_true < hi:
                    shell_label = lbl
                    break
            if shell_label is None:
                continue

            bd = bin_data[shell_label]
            # RMSD contribution
            delta = pred_frac[ri] - true_frac[ci]
            delta -= np.round(delta)
            bd['rmsd_sq_sum'] += float(np.sum((delta * L) ** 2))
            bd['n_matched']   += 1

            # Type acc Top-1
            if int(pred_types[ri]) == int(true_types[ci]):
                bd['type1_correct'] += 1

            # Type acc Top-3
            if pred_logits is not None and inv_vocab is not None:
                logits_i  = pred_logits[ri]
                top3_class = np.argsort(logits_i)[-3:]
                top3_z     = [inv_vocab.get(int(c), -1) for c in top3_class]
                if int(true_types[ci]) in top3_z:
                    bd['type3_correct'] += 1
                bd['type_valid'] += 1

    # 汇总
    stats = []
    for lo, hi, label in shell_bins:
        bd  = bin_data[label]
        n   = bd['n_matched']
        if n == 0:
            stats.append({'label': label, 'n': 0,
                          'rmsd': float('nan'), 'type1': float('nan'),
                          'type3': float('nan')})
            continue
        rmsd   = float(np.sqrt(bd['rmsd_sq_sum'] / n))
        type1  = float(bd['type1_correct'] / n)
        type3  = (float(bd['type3_correct'] / bd['type_valid'])
                  if bd['type_valid'] > 0 else float('nan'))
        stats.append({'label': label, 'n': n,
                      'rmsd': rmsd, 'type1': type1, 'type3': type3})
    return stats


# ── 子群统计（继承 v5）────────────────────────────────────────────────────────

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
                       float(np.mean([r['rmsd'] for r in sub])),
                       float(np.mean([r['type_acc'] for r in sub]))))
    return groups


# ── 主评估函数 ────────────────────────────────────────────────────────────────

def compute_metrics(pred_path, split_name, report_lines, vocab, inv_vocab):
    import numpy as np
    import torch
    logger = logging.getLogger(__name__)

    if not os.path.exists(pred_path):
        logger.error(f"❌ 找不到 {pred_path}")
        report_lines.append(f"\n❌ {split_name}: 文件不存在 {pred_path}")
        return

    preds = torch.load(pred_path, map_location="cpu")
    n     = len(preds['mp_id'])
    has_logits = 'pred_type_logits' in preds

    logger.info(f"\n计算 {split_name} 集指标（{n} 个样本，L={L}，最小镜像匈牙利匹配）...")
    logger.info(f"  pred_type_logits 存在：{'✅' if has_logits else '❌（将跳过 Top-3）'}")

    results  = []
    skipped  = 0

    for i in range(n):
        pf  = preds['pred_frac_coords'][i].numpy()       # (20, 3)
        pt  = preds['pred_atom_types'][i].numpy()         # (20,)
        tf  = preds['true_frac_coords'][i].numpy()        # (20, 3)
        tt  = preds['true_atom_types'][i].numpy()         # (20,)
        ec  = float(preds['eval_cutoff'][i])

        if pf.shape[0] != 20 or tf.shape[0] != 20:
            skipped += 1
            continue

        pl = None
        if has_logits:
            pl = preds['pred_type_logits'][i].numpy()    # (20, N_elem)

        r = evaluate_sample(pf, pt, tf, tt, ec,
                            pred_logits=pl, inv_vocab=inv_vocab, L=L)

        # 保存用于壳层统计的原始数据
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

    rmsds      = np.array([r['rmsd']          for r in results])
    type_accs  = np.array([r['type_acc']       for r in results])
    type_accs3 = np.array([r['type_acc_top3']  for r in results
                           if not np.isnan(r['type_acc_top3'])])
    n_pred_in  = np.array([r['n_pred_in']      for r in results])
    n_true_in  = np.array([r['n_true_in']      for r in results])

    rb = L / 2 * (3 / 5) ** 0.5   # 随机基线 ≈ 2.32 Å（L=6）

    # ── 验收标准判断 ──────────────────────────────────────────────────────────
    rmsd_ok    = rmsds.mean() <= 1.6
    type1_ok   = type_accs.mean() >= 0.40
    type3_ok   = (type_accs3.mean() >= 0.65) if len(type_accs3) > 0 else False
    in_cut_ok  = n_pred_in.mean() >= 15.0

    lines = [
        "",
        f"=== {split_name} Set Metrics (Step5b Exp3, min-image Hungarian, L={L}Å) ===",
        f"N_samples        : {len(results)}",
        f"",
        f"── 坐标（RMSD）──",
        f"RMSD (Å)         : mean={rmsds.mean():.4f}  "
            f"median={np.median(rmsds):.4f}  std={rmsds.std():.4f}"
            f"  {'✅' if rmsd_ok else '❌'} (目标 ≤ 1.6Å)",
        f"",
        f"── 原子类型 ──",
        f"Type Acc Top-1   : mean={type_accs.mean():.4f}  "
            f"median={np.median(type_accs):.4f}"
            f"  {'✅' if type1_ok else '❌'} (目标 ≥ 0.40)",
    ]

    if len(type_accs3) > 0:
        lines.append(
            f"Type Acc Top-3   : mean={type_accs3.mean():.4f}  "
            f"N_valid={len(type_accs3)}"
            f"  {'✅' if type3_ok else '❌'} (目标 ≥ 0.65)"
        )
    else:
        lines.append("Type Acc Top-3   : N/A（pred_type_logits 缺失）")

    lines += [
        "",
        f"── 原子密度（eval_cutoff 内） ──",
        f"pred_in_cutoff   : mean={n_pred_in.mean():.2f} / 20"
            f"  {'✅' if in_cut_ok else '❌'} (目标 ≥ 15)",
        f"true_in_cutoff   : mean={n_true_in.mean():.2f} / 20",
    ]

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

    # ── 壳层分组（新增）──────────────────────────────────────────────────────
    lines += ["", "── 壳层分组统计（按 true 原子到 Fe 原点距离，匹配对级别）──"]
    shell_results = shell_stats(results, SHELL_BINS, L=L)
    for sr in shell_results:
        top3_str = f"{sr['type3']:.4f}" if not np.isnan(sr.get('type3', float('nan'))) else "N/A"
        lines.append(
            f"  {sr['label']:22s}  "
            f"N_pairs={sr['n']:5d}  "
            f"RMSD={sr['rmsd']:.4f}  "
            f"TypeAcc_Top1={sr['type1']:.4f}  "
            f"TypeAcc_Top3={top3_str}"
        )

    # ── 总结 ─────────────────────────────────────────────────────────────────
    all_ok = rmsd_ok and type1_ok and type3_ok and in_cut_ok
    lines += [
        "",
        f"随机基线 RMSD ≈ {rb:.2f} Å（[-L/2, L/2] 均匀分布，L={L}）",
        f"",
        f"Exp3 验收标准：",
        f"  RMSD ≤ 1.6 Å        : {'✅' if rmsd_ok else '❌'}",
        f"  Type Acc Top-1 ≥ 0.40: {'✅' if type1_ok else '❌'}",
        f"  Type Acc Top-3 ≥ 0.65: {'✅' if type3_ok else '❌'}",
        f"  pred_in_cutoff ≥ 15  : {'✅' if in_cut_ok else '❌'}",
        f"",
        f"{'✅ 所有验收指标达标！' if all_ok else '❌ 存在未达标指标（见上）'}",
    ]

    for line in lines:
        logger.info(line)
    report_lines.extend(lines)


# ── 入口 ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import numpy as np
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("Step5b_2  Exp3 指标计算（val + test）")
    logger.info("=" * 60)

    # 加载词表
    if not os.path.exists(VOCAB_PATH):
        logger.error(f"❌ 词表不存在：{VOCAB_PATH}")
        sys.exit(1)
    vocab, inv_vocab = load_vocab(VOCAB_PATH)
    logger.info(f"词表加载：{len(vocab)} 个元素类别")

    os.makedirs(STEP5B_DIR, exist_ok=True)

    report_lines = [
        "Step5b Metrics Report — Experiment 3",
        f"L={L} Å，坐标系 [-0.5, 0.5]，最小镜像匈牙利匹配",
        f"新增：Type Accuracy Top-3，壳层分组统计",
        "=" * 60,
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

    report_path = os.path.join(STEP5B_DIR, "metrics_report_val_test.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    logger.info(f"\n完整报告 → {report_path}")
    logger.info("=" * 60)
    logger.info("Step5b_2 完成")
