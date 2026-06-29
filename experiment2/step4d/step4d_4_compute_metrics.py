# step4d_4_compute_metrics.py
# Step4d — 评估指标（L=6 版本）
# ============================================================
# 直接复用 step4b_4_compute_metrics.py v5 逻辑（最小镜像匈牙利匹配）
# Step4d 修改：
#   L = 12.0 → 6.0
#   STEP4b_DIR → STEP4d_DIR（路径）
#   随机基线 RMSD 重新计算（L=6: sqrt(3/5)*3 ≈ 2.32 Å）
# ============================================================

import os, sys, logging, warnings

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
warnings.filterwarnings("ignore")

PROJECT_ROOT = r"C:\Users\T-Cat\Desktop\DiffCSP-main"
EXP2_ROOT    = os.path.join(PROJECT_ROOT, "experiment2")
STEP4d_DIR   = os.path.join(EXP2_ROOT, "step4d")          # ← step4d
L = 6.0                                                    # ★ L=6


def evaluate_sample(pred_frac, pred_types, true_frac, true_types, eval_cutoff, L=6.0):
    """
    全部 20 个预测原子 vs 20 个真实原子做匈牙利匹配。

    坐标约定（Step4d / diffusion v3）：
      pred_frac, true_frac 均在 [-0.5, 0.5]（Dataset v5-L6 输出）

    匹配和 RMSD：使用最小镜像距离（周期边界条件）
      delta = p - t;  delta -= round(delta);  dist = ||delta * L||
    对 [-0.5, 0.5] 坐标系通用，最小镜像折叠对坐标系无依赖。
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    pred_frac = np.array(pred_frac, dtype=np.float64)
    true_frac = np.array(true_frac, dtype=np.float64)

    n = pred_frac.shape[0]   # 固定为 20

    # ── 构建 20×20 最小镜像距离矩阵 ──────────────────────────────────────
    cost_matrix = np.zeros((n, n))
    for i in range(n):
        delta = pred_frac[i] - true_frac          # (20, 3)
        delta -= np.round(delta)                   # 最小镜像
        cost_matrix[i] = np.linalg.norm(delta * L, axis=1)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # ── RMSD：用匹配后的最小镜像距离 ─────────────────────────────────────
    matched_dists_sq = []
    for ri, ci in zip(row_ind, col_ind):
        delta = pred_frac[ri] - true_frac[ci]
        delta -= np.round(delta)
        matched_dists_sq.append(np.sum((delta * L) ** 2))

    rmsd     = float(np.sqrt(np.mean(matched_dists_sq)))
    type_acc = float((pred_types[row_ind] == true_types[col_ind]).mean())

    # ── 子群统计：eval_cutoff 范围内有多少原子（最小镜像距离到 Fe 原点）──
    pred_mi = pred_frac.copy()
    true_mi = true_frac.copy()
    # 对 [-0.5, 0.5] 坐标系而言最小镜像折叠不改变值，但保持通用性
    pred_mi -= np.round(pred_mi)
    true_mi -= np.round(true_mi)
    pred_dists = np.linalg.norm(pred_mi * L, axis=1)
    true_dists = np.linalg.norm(true_mi * L, axis=1)
    n_pred_in  = int((pred_dists <= eval_cutoff).sum())
    n_true_in  = int((true_dists <= eval_cutoff).sum())

    return {
        'rmsd':        rmsd,
        'type_acc':    type_acc,
        'n_pred_in':   n_pred_in,
        'n_true_in':   n_true_in,
        'eval_cutoff': eval_cutoff,
    }


def subgroup_stats(results, key, bins):
    import numpy as np
    groups = []
    for lo, hi, label in bins:
        sub = [r for r in results
               if (lo is None or r[key] >= lo) and (hi is None or r[key] < hi)]
        if not sub:
            groups.append((label, 0, float('nan'), float('nan')))
            continue
        groups.append((label, len(sub),
                       float(np.mean([r['rmsd'] for r in sub])),
                       float(np.mean([r['type_acc'] for r in sub]))))
    return groups


def compute_metrics(pred_path, split_name, report_lines):
    import numpy as np, torch
    logger = logging.getLogger(__name__)

    if not os.path.exists(pred_path):
        logger.error(f"❌ 找不到 {pred_path}，请先运行采样脚本")
        return

    preds = torch.load(pred_path, map_location="cpu")
    n     = len(preds['mp_id'])

    # 验证预测文件中记录的 L 值
    pred_L = preds.get('L', 'unknown')
    if pred_L != L:
        logger.warning(f"⚠️  predictions 文件 L={pred_L}，脚本 L={L}，请确认一致性！")

    logger.info(f"\n计算 {split_name} 集指标（{n} 个样本，L={L}Å，最小镜像匈牙利匹配）...")

    results, skipped = [], 0
    for i in range(n):
        pf = preds['pred_frac_coords'][i].numpy()
        pt = preds['pred_atom_types'][i].numpy()
        tf = preds['true_frac_coords'][i].numpy()
        tt = preds['true_atom_types'][i].numpy()
        ec = float(preds['eval_cutoff'][i])

        if pf.shape[0] != 20 or tf.shape[0] != 20:
            skipped += 1
            continue

        r = evaluate_sample(pf, pt, tf, tt, ec, L=L)
        results.append(r)

    logger.info(f"  有效样本：{len(results)}/{n}（跳过 {skipped}）")

    rmsds     = [r['rmsd']     for r in results]
    type_accs = [r['type_acc'] for r in results]
    n_pred_in = [r['n_pred_in'] for r in results]
    n_true_in = [r['n_true_in'] for r in results]

    import numpy as np
    rmsds     = np.array(rmsds)
    type_accs = np.array(type_accs)
    n_pred_in = np.array(n_pred_in)
    n_true_in = np.array(n_true_in)

    # L=6 随机基线：[-L/2, L/2]³ 均匀分布期望 RMSD = L/2 * sqrt(3/5) ≈ 2.32 Å
    rb = (L / 2) * (3 / 5) ** 0.5

    lines = [
        "",
        f"=== {split_name} Set Metrics (Step4d, L=6Å, min-image Hungarian matching) ===",
        f"N_samples        : {len(results)}",
        f"RMSD (Å)         : mean={rmsds.mean():.4f}  "
            f"median={np.median(rmsds):.4f}  std={rmsds.std():.4f}",
        f"Type Accuracy    : mean={type_accs.mean():.4f}  "
            f"median={np.median(type_accs):.4f}",
        f"",
        f"原子密度参考（eval_cutoff 内，最小镜像到 Fe 原点）：",
        f"  pred_in_cutoff : mean={n_pred_in.mean():.2f} / 20",
        f"  true_in_cutoff : mean={n_true_in.mean():.2f} / 20",
        f"  （验收要求：pred_in_cutoff > 10/20）",
    ]

    lines += ["", "── 子群：eval_cutoff ──"]
    for label, cnt, mr, mt in subgroup_stats(
            results, 'eval_cutoff',
            [(None, 3.0, "< 3.0 Å"), (3.0, 4.0, "3.0–4.0 Å")]):
        lines.append(f"  {label:12s}  N={cnt:5d}  RMSD={mr:.4f}  TypeAcc={mt:.4f}")

    lines += ["", "── 子群：n_true_in（eval_cutoff 内真实原子数）──"]
    for label, cnt, mr, mt in subgroup_stats(
            results, 'n_true_in',
            [(None, 9,  "≤ 8（第一壳层）"),
             (9,   15,  "9–14（第二壳层）"),
             (15,  None,"15–20（第三壳层）")]):
        lines.append(f"  {label:20s}  N={cnt:5d}  RMSD={mr:.4f}  TypeAcc={mt:.4f}")

    lines += [
        "",
        f"随机基线 RMSD ≈ {rb:.2f} Å（[-L/2, L/2]³ 均匀分布，L=6Å）",
        f"Step4d 验收标准：RMSD < 2.0 Å，pred_in_cutoff > 10/20，Type Acc ≥ 0.27",
    ]
    if rmsds.mean() < 2.0:
        lines.append(f"✅✅ RMSD={rmsds.mean():.4f} Å < 2.0 Å，达到 Step4d 验收标准！")
    elif rmsds.mean() < rb * 0.7:
        lines.append(f"✅ RMSD={rmsds.mean():.4f} Å，显著优于随机基线（< 70%），未达 2.0 Å 目标")
    elif rmsds.mean() < rb * 0.9:
        lines.append(f"⚠️  RMSD={rmsds.mean():.4f} Å，略优于随机基线（< 90%），需进一步分析")
    else:
        lines.append(f"❌ RMSD={rmsds.mean():.4f} Å，未有效优于随机基线，需汇报 Main Agent 2")

    for line in lines:
        logger.info(line)
    report_lines.extend(lines)


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("Step4d_4  评估指标（L=6Å，最小镜像匈牙利匹配）")
    logger.info("=" * 60)

    report_lines = [
        "Step4d Metrics Report",
        f"L={L}Å，坐标系 [-0.5, 0.5]，匹配/RMSD 使用最小镜像周期边界距离",
        "=" * 60,
    ]

    compute_metrics(
        os.path.join(STEP4d_DIR, "predictions_val.pt"),  "Val",  report_lines)
    compute_metrics(
        os.path.join(STEP4d_DIR, "predictions_test.pt"), "Test", report_lines)

    report_path = os.path.join(STEP4d_DIR, "metrics_report.txt")
    os.makedirs(STEP4d_DIR, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    logger.info(f"\n完整报告 → {report_path}")
    logger.info("=" * 60)
    logger.info("Step4d_4 完成")
