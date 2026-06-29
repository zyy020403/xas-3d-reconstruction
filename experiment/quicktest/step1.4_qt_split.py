# =============================================================================
# 脚本编号: step1.4_qt
# 脚本名称: step1.4_qt_split.py
# 输入:
#   - C:\Users\T-Cat\Desktop\DiffCSP-main\experiment\quicktest\qt_inventory.csv
# 输出:
#   - C:\Users\T-Cat\Desktop\DiffCSP-main\experiment\quicktest\qt_train_ids.txt  (~70)
#   - C:\Users\T-Cat\Desktop\DiffCSP-main\experiment\quicktest\qt_val_ids.txt   (~15)
#   - C:\Users\T-Cat\Desktop\DiffCSP-main\experiment\quicktest\qt_test_ids.txt  (~15)
# 说明:
#   QuickTest 简化版划分（对应正式服 step1.4_split_dataset.py）。
#   差异：
#     - 无保留集（holdout）
#     - 无分层采样，直接随机划分
#     - 无质量过滤（qt_inventory 已经只有 100 个 Fe 位点，质量可接受）
#     - 比例 70 / 15 / 15（random.seed=42）
# =============================================================================

import os
import random
import pandas as pd

# ── 路径常量 ──────────────────────────────────────────────────────────────────
PROJECT_ROOT   = r"C:\Users\T-Cat\Desktop\DiffCSP-main"
EXPERIMENT_DIR = os.path.join(PROJECT_ROOT, "experiment")
QT_DIR         = os.path.join(EXPERIMENT_DIR, "quicktest")

INPUT_INVENTORY = os.path.join(QT_DIR, "qt_inventory.csv")
OUTPUT_TRAIN    = os.path.join(QT_DIR, "qt_train_ids.txt")
OUTPUT_VAL      = os.path.join(QT_DIR, "qt_val_ids.txt")
OUTPUT_TEST     = os.path.join(QT_DIR, "qt_test_ids.txt")

RANDOM_SEED  = 42
TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.15
# TEST = 剩余（约 0.15）


def write_ids(filepath: str, ids: list):
    with open(filepath, "w", encoding="utf-8") as f:
        for mid in sorted(ids, key=str):
            f.write(f"{mid}\n")
    print(f"  → 写入 {len(ids)} 个 mp_id: {filepath}")


def main():
    print("=" * 60)
    print("step1.4_qt 开始：QuickTest 数据集划分（70/15/15）")
    print("=" * 60)

    # ── 1. 读取 qt_inventory.csv ──────────────────────────────────────────────
    if not os.path.exists(INPUT_INVENTORY):
        raise FileNotFoundError(
            f"qt_inventory.csv 不存在，请先运行 step1.1_qt_select_100.py:\n{INPUT_INVENTORY}"
        )

    df = pd.read_csv(INPUT_INVENTORY, dtype={"mp_id": str})
    mp_ids = df["mp_id"].unique().tolist()
    print(f"读入 mp_id 数: {len(mp_ids)}")

    # ── 2. 随机打乱 ───────────────────────────────────────────────────────────
    random.seed(RANDOM_SEED)
    random.shuffle(mp_ids)

    # ── 3. 按比例切分 ─────────────────────────────────────────────────────────
    n_total = len(mp_ids)
    n_train = round(n_total * TRAIN_RATIO)   # 70
    n_val   = round(n_total * VAL_RATIO)     # 15
    # test 取剩余，避免因舍入导致遗漏

    train_ids = mp_ids[:n_train]
    val_ids   = mp_ids[n_train : n_train + n_val]
    test_ids  = mp_ids[n_train + n_val :]

    print(f"划分结果: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

    # ── 4. 完整性校验 ─────────────────────────────────────────────────────────
    all_out = set(train_ids) | set(val_ids) | set(test_ids)
    assert len(train_ids) + len(val_ids) + len(test_ids) == n_total, \
        "❌ 划分后总数不等于输入总数！"
    assert len(all_out) == n_total, \
        "❌ 划分后存在重复 mp_id！"
    print("✓ 校验通过：无重复、无遗漏")

    # ── 5. 写入文件 ───────────────────────────────────────────────────────────
    write_ids(OUTPUT_TRAIN, train_ids)
    write_ids(OUTPUT_VAL,   val_ids)
    write_ids(OUTPUT_TEST,  test_ids)

    print("\n══ step1.4_qt 执行摘要 ══")
    print(f"  输入 mp_id 总数:  {n_total}")
    print(f"  train:           {len(train_ids)}")
    print(f"  val:             {len(val_ids)}")
    print(f"  test:            {len(test_ids)}")
    print("step1.4_qt 完成。")


if __name__ == "__main__":
    main()