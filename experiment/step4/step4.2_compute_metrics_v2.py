# =============================================================================
# 脚本编号: step4.2
# 脚本名称: step4.2_compute_metrics.py  (v5 — 纯晶格参数版，无结构构建)
# 输入:
#   - experiment/step4/predictions_val.pt
#   - experiment/step4/predictions_test.pt
#   - experiment/step1/data_inventory.csv
#   - experiment/step1/bond_length_constraints.json (本版本不调用，仅加载备用)
# 输出:
#   - experiment/step4/metrics_report.txt
#   - experiment/step4/metrics_detail.csv
# 说明:
#   瓶颈定位：get_all_neighbors(4.0) 在 70 原子超胞上每个 ~18s。
#   本版本完全跳过结构构建和 get_all_neighbors，
#   仅做纯数值的晶格参数比较，运行时间 < 5 秒。
#   键长违规率标记为 N/A，在报告中注明需要 Step 5 中针对原胞单独计算。
# =============================================================================

import os, sys, json, torch
import numpy as np
import pandas as pd

PROJECT_ROOT = r"C:\Users\T-Cat\Desktop\DiffCSP-main"
EXPERIMENT_DIR = os.path.join(PROJECT_ROOT, "experiment")
STEP4_DIR  = os.path.join(EXPERIMENT_DIR, "step4")
STEP1_DIR  = os.path.join(EXPERIMENT_DIR, "step1")

os.environ["PROJECT_ROOT"] = PROJECT_ROOT
sys.path.insert(0, PROJECT_ROOT)

PRED_VAL_PATH   = os.path.join(STEP4_DIR, "predictions_val.pt")
PRED_TEST_PATH  = os.path.join(STEP4_DIR, "predictions_test.pt")
INVENTORY_CSV   = os.path.join(STEP1_DIR, "data_inventory.csv")
BOND_JSON       = os.path.join(STEP1_DIR, "bond_length_constraints.json")
REPORT_PATH     = os.path.join(STEP4_DIR, "metrics_report.txt")
DETAIL_CSV_PATH = os.path.join(STEP4_DIR, "metrics_detail.csv")

LTOL      = 0.3   # lengths 相对容差（与 StructureMatcher 一致）
ANGLE_TOL = 10.0  # angles 绝对容差（度）


def safe_np(t):
    return t.numpy() if hasattr(t, "numpy") else np.asarray(t, dtype=float)


def is_valid(lengths, angles) -> bool:
    if any(l <= 0 for l in lengths):
        return False
    if any(np.isnan(v) for v in list(lengths) + list(angles)):
        return False
    if any(a <= 0 or a >= 180 for a in angles):
        return False
    return True


def lattice_match(pred_l, pred_a, gt_l, gt_a):
    """
    纯数值比较，无需构建 Structure 对象。
    排序后比较（旋转不变近似）。
    返回 (matched, length_rmse, angle_mae)
    """
    pl = np.sort(pred_l)
    gl = np.sort(gt_l)
    pa = np.sort(pred_a)
    ga = np.sort(gt_a)

    rel_err   = np.abs(pl - gl) / (gl + 1e-8)
    angle_err = np.abs(pa - ga)

    matched    = bool(rel_err.max() < LTOL and angle_err.max() < ANGLE_TOL)
    length_rmse = float(np.sqrt(np.mean((pl - gl) ** 2)))
    angle_mae   = float(np.mean(angle_err))
    return matched, length_rmse, angle_mae


def evaluate_split(split_name, predictions, inventory_df):
    print(f"\n{'='*60}\n  {split_name}  ({len(predictions)} compounds)\n{'='*60}")
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
            "mp_id":           mp_id,
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
    print(f"  Done: {len(df)} rows computed.")

    if inventory_df is not None:
        inv_cols = ["mp_id"] + [c for c in
                    ["is_ionic","quality_tier","n_sites","element"]
                    if c in inventory_df.columns]
        inv = inventory_df[inv_cols].drop_duplicates("mp_id").copy()
        inv["mp_id"] = inv["mp_id"].astype(str)
        df["mp_id"]  = df["mp_id"].astype(str)
        df = df.merge(inv, on="mp_id", how="left")
        n_joined = df["mp_id"].notna().sum()
        print(f"  Inventory joined: {n_joined}/{len(df)}")
    return df


def sg(df, mask, label):
    sub = df[mask]
    n   = len(sub)
    if n == 0:
        return f"  {label}: N=0"
    mr  = sub["lattice_matched"].sum() / n
    lr  = sub["length_rmse"].dropna().mean()
    am  = sub["angle_mae"].dropna().mean()
    return (f"  {label}: LatticeMatch={mr:.1%}  "
            f"LenRMSE={lr:.3f} Å  AngleMAE={am:.2f}°  (N={n})")


def generate_report(val_df, test_df) -> str:
    lines = [
        "=" * 65,
        "  Step 4 评估指标报告（v5 纯晶格参数版）",
        "",
        "  Lattice Match Rate = lengths 相对误差 < 30%",
        "                       AND angles 绝对误差 < 10°",
        "  （StructureMatcher 晶格步骤等价，是 match_rate 的上界）",
        "  Bond Violation Rate = N/A（超胞 get_all_neighbors 太慢，",
        "                        将在 Step 5 针对原胞单独计算）",
        "=" * 65,
    ]

    for name, df in [("Val", val_df), ("Test", test_df)]:
        n  = len(df)
        mr = df["lattice_matched"].sum() / n
        lr = df["length_rmse"].dropna().mean()
        am = df["angle_mae"].dropna().mean()
        n_invalid = (~df["pred_valid"]).sum()
        n_extreme = (df["pred_len_max"] > 100.0).sum()
        pct_extreme = n_extreme / n

        lines += [
            f"\n{'─'*60}", f"  {name} 集", f"{'─'*60}",
            f"总样本数                      : {n}",
            f"预测结构无效（负/NaN晶格）    : {n_invalid}",
            f"Pred lengths > 100 Å          : {n_extreme} ({pct_extreme:.1%})"
            + ("  ← 扩散未收敛 outlier，需关注" if pct_extreme > 0.05 else "  (OK)"),
            f"Lattice Match Rate (上界)     : {mr:.1%}  ({df['lattice_matched'].sum()}/{n})",
            f"Mean Length RMSE              : {lr:.4f} Å",
            f"Mean Angle MAE                : {am:.3f} °",
            f"Bond Viol Rate                : N/A（见上方说明）",
        ]

        for col, label in [("is_ionic",     "is_ionic 分析"),
                            ("n_sites",      "位点数分析"),
                            ("quality_tier", "质量分级分析")]:
            if col not in df.columns:
                lines.append(f"\n  [{label}] 列不可用")
                continue
            lines.append(f"\n  [{label}]")
            if col == "is_ionic":
                lines.append(sg(df, df["is_ionic"] == False, "纯共价 (is_ionic=False)"))
                lines.append(sg(df, df["is_ionic"] == True,  "含 ionic (is_ionic=True)"))
            elif col == "n_sites":
                lines.append(sg(df, df["n_sites"] == 1,          "1 个位点"))
                lines.append(sg(df, df["n_sites"].between(2, 3),  "2–3 个位点"))
                lines.append(sg(df, df["n_sites"] >= 4,           "4+ 个位点"))
            elif col == "quality_tier":
                for t in ["A", "B", "C"]:
                    lines.append(sg(df, df["quality_tier"] == t, f"Quality {t}"))

        if "is_ionic" in df.columns and "element" in df.columns:
            lines += ["\n  [元素种类分析]",
                      sg(df,
                         df["element"].str.upper().str.contains("FE", na=False)
                         & (df["is_ionic"] == False), "含 Fe 且纯共价"),
                      sg(df,
                         df["element"].str.upper().str.contains("FE", na=False)
                         & (df["is_ionic"] == True),  "含 Fe 且含 ionic")]

        lines += [
            "\n  [预测结构健康检查]",
            f"  Length RMSE > 5 Å            : {(df['length_rmse'] > 5.0).sum()}",
            f"  Angle MAE > 30°              : {(df['angle_mae'] > 30.0).sum()}",
            f"  pred_len_mean (avg)          : {df['pred_len_mean'].mean():.2f} Å",
            f"  gt_len_mean   (avg)          : {df['gt_len_mean'].mean():.2f} Å",
        ]

    lines += [
        "\n" + "="*65,
        "  后续建议：",
        "  1. Fine-tune (Step 4.3) 完成后重新采样，用原胞输出计算 StructureMatcher",
        "  2. 键长违规率在 Step 5 盲测时对原胞结构计算",
        "="*65,
    ]
    return "\n".join(lines)


def main():
    print("[Inventory] loading ...")
    if os.path.exists(INVENTORY_CSV):
        inventory_df = pd.read_csv(INVENTORY_CSV)
        print(f"  {len(inventory_df)} rows, cols: {list(inventory_df.columns)}")
    else:
        print("  WARNING: not found.")
        inventory_df = None

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
    print(f"[Saved] {REPORT_PATH}")

    for name, df in [("Val", val_df), ("Test", test_df)]:
        n = len(df)
        print(f"[SUMMARY] {name}: LatticeMatch={df['lattice_matched'].sum()/n:.1%} "
              f"({df['lattice_matched'].sum()}/{n})")


if __name__ == "__main__":
    main()