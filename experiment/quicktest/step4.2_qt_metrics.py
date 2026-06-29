# =============================================================================
# 脚本编号: step4.2_qt
# 脚本名称: step4.2_qt_metrics.py
# 输入:
#   - experiment/quicktest/qt_step4/predictions_val.pt
#   - experiment/quicktest/qt_step4/predictions_test.pt
#   - experiment/quicktest/qt_inventory.csv
# 输出:
#   - experiment/quicktest/qt_step4/metrics_report.txt
#   - experiment/quicktest/qt_step4/metrics_detail.csv
# 说明:
#   正式服 step4.2_compute_metrics_v2.py 的 qt 版本。
#   改动：
#     - 所有路径指向 quicktest/qt_step4/
#     - inventory 读 qt_inventory.csv
#     - 去掉 is_ionic / n_sites / quality_tier 分层分析（qt 样本全是
#       单位点非离子 Fe 化合物，分层无意义）
#     - 其余评估逻辑（lattice_match、valid 检查、报告格式）完全不变
# =============================================================================

import os
import sys
import torch
import numpy as np
import pandas as pd

PROJECT_ROOT   = r"C:\Users\T-Cat\Desktop\DiffCSP-main"
EXPERIMENT_DIR = os.path.join(PROJECT_ROOT, "experiment")
QT_DIR         = os.path.join(EXPERIMENT_DIR, "quicktest")
QT_STEP4_DIR   = os.path.join(QT_DIR, "qt_step4")

os.environ["PROJECT_ROOT"] = PROJECT_ROOT
sys.path.insert(0, PROJECT_ROOT)

PRED_VAL_PATH   = os.path.join(QT_STEP4_DIR, "predictions_val.pt")
PRED_TEST_PATH  = os.path.join(QT_STEP4_DIR, "predictions_test.pt")
INVENTORY_CSV   = os.path.join(QT_DIR, "qt_inventory.csv")
REPORT_PATH     = os.path.join(QT_STEP4_DIR, "metrics_report.txt")
DETAIL_CSV_PATH = os.path.join(QT_STEP4_DIR, "metrics_detail.csv")

LTOL      = 0.3    # lengths 相对容差（与正式服相同）
ANGLE_TOL = 10.0   # angles 绝对容差（度）


def safe_np(t):
    return t.numpy() if hasattr(t, "numpy") else np.asarray(t, dtype=float)


def is_valid(lengths, angles) -> bool:
    if any(l <= 0 for l in lengths): return False
    if any(np.isnan(v) for v in list(lengths) + list(angles)): return False
    if any(a <= 0 or a >= 180 for a in angles): return False
    return True


def lattice_match(pred_l, pred_a, gt_l, gt_a):
    pl = np.sort(pred_l);  gl = np.sort(gt_l)
    pa = np.sort(pred_a);  ga = np.sort(gt_a)
    rel_err   = np.abs(pl - gl) / (gl + 1e-8)
    angle_err = np.abs(pa - ga)
    matched     = bool(rel_err.max() < LTOL and angle_err.max() < ANGLE_TOL)
    length_rmse = float(np.sqrt(np.mean((pl - gl) ** 2)))
    angle_mae   = float(np.mean(angle_err))
    return matched, length_rmse, angle_mae


def evaluate_split(split_name, predictions, inventory_df):
    print(f"\n{'='*60}\n  {split_name}  ({len(predictions)} 个化合物)\n{'='*60}")
    rows = []

    for mp_id, entry in predictions.items():
        pred_l = safe_np(entry["pred_lengths"])
        pred_a = safe_np(entry["pred_angles"])
        gt_l   = safe_np(entry["gt_lengths"])
        gt_a   = safe_np(entry["gt_angles"])

        pred_valid = is_valid(pred_l, pred_a)
        gt_valid   = is_valid(gt_l,   gt_a)

        if pred_valid and gt_valid:
            matched, len_rmse, ang_mae = lattice_match(pred_l, pred_a, gt_l, gt_a)
        else:
            matched, len_rmse, ang_mae = False, float("nan"), float("nan")

        rows.append({
            "mp_id":           str(mp_id),
            "split":           split_name,
            "lattice_matched": matched,
            "length_rmse":     len_rmse,
            "angle_mae":       ang_mae,
            "pred_valid":      pred_valid,
            "gt_valid":        gt_valid,
            "n_atoms":         entry["n_atoms"],
            "pred_len_max":    float(pred_l.max()),
            "pred_len_mean":   float(pred_l.mean()),
            "gt_len_mean":     float(gt_l.mean()),
        })

    df = pd.DataFrame(rows)

    if inventory_df is not None:
        inv = inventory_df[["mp_id"]].copy()
        inv["mp_id"] = inv["mp_id"].astype(str)
        df = df.merge(inv, on="mp_id", how="left")

    return df


def generate_report(val_df, test_df) -> str:
    lines = [
        "=" * 65,
        "  QuickTest Step 4 评估指标报告（纯晶格参数版）",
        "",
        f"  Lattice Match：lengths 相对误差 < {LTOL*100:.0f}%",
        f"                 AND angles 绝对误差 < {ANGLE_TOL:.0f}°",
        "  Bond Violation Rate：N/A（超胞太慢，Skip）",
        "  注：qt 只有 15 个 val / 15 个 test 化合物，",
        "      结果仅供 pipeline 验证，不代表模型真实性能",
        "=" * 65,
    ]

    for name, df in [("Val", val_df), ("Test", test_df)]:
        n         = len(df)
        mr        = df["lattice_matched"].sum() / n if n > 0 else 0
        lr        = df["length_rmse"].dropna().mean()
        am        = df["angle_mae"].dropna().mean()
        n_invalid = (~df["pred_valid"]).sum()
        n_extreme = (df["pred_len_max"] > 100.0).sum()

        lines += [
            f"\n{'─'*60}",
            f"  {name} 集（N={n}）",
            f"{'─'*60}",
            f"预测结构无效（负/NaN晶格）    : {n_invalid}",
            f"Pred lengths > 100 Å（outlier）: {n_extreme}"
            + ("  ← 扩散未收敛，需关注" if n_extreme > 0 else "  (OK)"),
            f"Lattice Match Rate            : {mr:.1%}  ({df['lattice_matched'].sum()}/{n})",
            f"Mean Length RMSE              : {lr:.4f} Å",
            f"Mean Angle MAE                : {am:.3f} °",
            f"Bond Viol Rate                : N/A",
            "",
            "  [结构健康检查]",
            f"  Length RMSE > 5 Å  : {(df['length_rmse'] > 5.0).sum()}",
            f"  Angle MAE > 30°    : {(df['angle_mae'] > 30.0).sum()}",
            f"  pred_len_mean (avg): {df['pred_len_mean'].mean():.2f} Å",
            f"  gt_len_mean   (avg): {df['gt_len_mean'].mean():.2f} Å",
        ]

        # 逐化合物明细（qt 样本少，直接打印）
        lines.append("\n  [逐样本明细]")
        lines.append(f"  {'mp_id':>10}  {'matched':>7}  {'len_rmse':>9}  {'ang_mae':>8}  {'pred_valid':>10}")
        for _, row in df.iterrows():
            lines.append(
                f"  {str(row['mp_id']):>10}  "
                f"{'✓' if row['lattice_matched'] else '✗':>7}  "
                f"{row['length_rmse']:>9.3f}  "
                f"{row['angle_mae']:>8.2f}  "
                f"{'✓' if row['pred_valid'] else '✗':>10}"
            )

    lines += [
        "\n" + "="*65,
        "  QuickTest 验收参考（非强制）：",
        "  Lattice Match Rate > 0% 即说明扩散有意义的输出",
        "  pred_len_max 无大量 >100 Å 的 outlier",
        "="*65,
    ]
    return "\n".join(lines)


def main():
    print("[Load] qt_inventory.csv ...")
    inventory_df = pd.read_csv(INVENTORY_CSV) if os.path.exists(INVENTORY_CSV) else None

    print("[Load] predictions ...")
    val_preds  = torch.load(PRED_VAL_PATH,  map_location="cpu", weights_only=False)
    test_preds = torch.load(PRED_TEST_PATH, map_location="cpu", weights_only=False)
    print(f"  val={len(val_preds)}, test={len(test_preds)}")

    val_df  = evaluate_split("val",  val_preds,  inventory_df)
    test_df = evaluate_split("test", test_preds, inventory_df)

    pd.concat([val_df, test_df], ignore_index=True).to_csv(DETAIL_CSV_PATH, index=False)
    print(f"\n[Saved] {DETAIL_CSV_PATH}")

    report = generate_report(val_df, test_df)
    print("\n" + report)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n[Saved] {REPORT_PATH}")

    for name, df in [("Val", val_df), ("Test", test_df)]:
        n = len(df)
        print(f"[SUMMARY] {name}: LatticeMatch={df['lattice_matched'].sum()/n:.1%} "
              f"({df['lattice_matched'].sum()}/{n})")


if __name__ == "__main__":
    main()