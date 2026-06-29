# =============================================================================
# 脚本编号: step1.1_qt
# 脚本名称: step1.1_qt_select_100.py
# 输入:
#   - C:\Users\T-Cat\Desktop\DiffCSP-main\experiment\step1\data_inventory.csv
# 输出:
#   - C:\Users\T-Cat\Desktop\DiffCSP-main\experiment\quicktest\qt_inventory.csv
# 说明:
#   从正式服已生成的 data_inventory.csv 中筛选出：
#     - is_ionic=False
#     - files_complete=True
#     - element='Fe'
#   每个 mp_id 只保留 site_id 最小的那一行（第一个 Fe 位点），
#   再随机抽取 100 个 mp_id（seed=42），输出 qt_inventory.csv。
# =============================================================================

import os
import random
import pandas as pd

# ── 路径常量 ──────────────────────────────────────────────────────────────────
PROJECT_ROOT    = r"C:\Users\T-Cat\Desktop\DiffCSP-main"
EXPERIMENT_DIR  = os.path.join(PROJECT_ROOT, "experiment")
STEP1_DIR       = os.path.join(EXPERIMENT_DIR, "step1")
QT_DIR          = os.path.join(EXPERIMENT_DIR, "quicktest")
os.makedirs(QT_DIR, exist_ok=True)

INPUT_INVENTORY  = os.path.join(STEP1_DIR, "data_inventory.csv")
OUTPUT_INVENTORY = os.path.join(QT_DIR, "qt_inventory.csv")

RANDOM_SEED = 42
N_COMPOUNDS = 100

# ── 输出列（只保留必要列）────────────────────────────────────────────────────
OUTPUT_COLUMNS = [
    "folder_name", "mp_id", "element", "site_id",
    "is_ionic", "source_path", "quality_tier",
]


def main():
    print(f"读取数据清单: {INPUT_INVENTORY}")
    df = pd.read_csv(INPUT_INVENTORY, dtype={"mp_id": str, "site_id": str})
    print(f"  原始总行数: {len(df)}")

    # ── 1. 过滤条件 ───────────────────────────────────────────────────────────
    # files_complete 列可能是布尔或字符串，统一转换
    df["files_complete"] = df["files_complete"].astype(str).str.strip().str.lower()
    mask = (
        (df["is_ionic"].astype(str).str.lower().isin(["false", "0"])) &
        (df["files_complete"] == "true") &
        (df["element"].str.strip() == "Fe")
    )
    df_fe = df[mask].copy()
    print(f"  过滤后（is_ionic=False, files_complete=True, element=Fe）: {len(df_fe)} 行")

    if len(df_fe) == 0:
        raise ValueError("过滤后无任何数据，请检查 data_inventory.csv 是否已生成。")

    # ── 2. 每个 mp_id 只保留 site_id 最小的行 ────────────────────────────────
    df_fe["site_id_int"] = df_fe["site_id"].astype(int)
    df_first = (
        df_fe.sort_values("site_id_int")
             .groupby("mp_id", as_index=False)
             .first()
    )
    print(f"  每个 mp_id 保留第一个 Fe 位点后: {len(df_first)} 个 mp_id")

    if len(df_first) < N_COMPOUNDS:
        raise ValueError(
            f"可用 mp_id 数量 ({len(df_first)}) 少于所需的 {N_COMPOUNDS} 个，"
            f"请放宽过滤条件或检查数据集。"
        )

    # ── 3. 随机抽取 100 个 mp_id ──────────────────────────────────────────────
    random.seed(RANDOM_SEED)
    all_mp_ids = df_first["mp_id"].tolist()
    selected_mp_ids = set(random.sample(all_mp_ids, N_COMPOUNDS))

    df_selected = df_first[df_first["mp_id"].isin(selected_mp_ids)].copy()
    print(f"  随机抽取后: {len(df_selected)} 行（seed={RANDOM_SEED}）")

    # ── 4. 输出必要列 ──────────────────────────────────────────────────────────
    # 确保所有输出列都存在（quality_tier 可能是 NaN，保留原值）
    for col in OUTPUT_COLUMNS:
        if col not in df_selected.columns:
            df_selected[col] = float("nan")

    df_out = df_selected[OUTPUT_COLUMNS].reset_index(drop=True)
    df_out.to_csv(OUTPUT_INVENTORY, index=False, encoding="utf-8")
    print(f"\n✓ qt_inventory.csv 已写入: {OUTPUT_INVENTORY}")

    # ── 5. 摘要 ────────────────────────────────────────────────────────────────
    print("\n══ step1.1_qt 执行摘要 ══")
    print(f"  输入总行数:         {len(df)}")
    print(f"  过滤后 Fe 位点数:   {len(df_fe)}")
    print(f"  唯一 mp_id 数:      {len(df_first)}")
    print(f"  最终选取 mp_id 数:  {len(df_out)}")
    print(f"  输出文件:           {OUTPUT_INVENTORY}")
    print("step1.1_qt 完成。")


if __name__ == "__main__":
    main()