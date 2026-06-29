# =============================================================================
# 脚本编号: step1.4
# 脚本名称: step1.4_split_dataset.py
# 输入:
#   - C:\Users\T-Cat\Desktop\DiffCSP-main\experiment\step1\data_inventory.csv
# 输出:
#   - C:\Users\T-Cat\Desktop\DiffCSP-main\experiment\step1\holdout_1000_ids.txt
#   - C:\Users\T-Cat\Desktop\DiffCSP-main\experiment\step1\train_ids.txt
#   - C:\Users\T-Cat\Desktop\DiffCSP-main\experiment\step1\val_ids.txt
#   - C:\Users\T-Cat\Desktop\DiffCSP-main\experiment\step1\test_ids.txt
#   - C:\Users\T-Cat\Desktop\DiffCSP-main\experiment\step1\split_summary.txt
# 说明:
#   以 mp_id 为单位进行数据集划分：
#   1. 仅使用 files_complete=True 的条目
#   2. 排除所有位点均为 C 级的 mp_id
#   3. 按元素组合标签进行分层采样，生成 1000 个保留集（holdout）
#   4. 剩余按 80/10/10 划分 train/val/test
#   全程 random_seed=42，同一 mp_id 的所有位点严格在同一集合中
# =============================================================================

import os
import random
import logging
import math
import pandas as pd
import numpy as np
from collections import defaultdict

# ── 路径常量 ──────────────────────────────────────────────────────────────────
PROJECT_ROOT   = r"C:\Users\T-Cat\Desktop\DiffCSP-main"
EXPERIMENT_DIR = os.path.join(PROJECT_ROOT, "experiment")
STEP_DIR       = os.path.join(EXPERIMENT_DIR, "step1")
os.makedirs(STEP_DIR, exist_ok=True)

INPUT_INVENTORY = os.path.join(STEP_DIR, "data_inventory.csv")

OUTPUT_HOLDOUT  = os.path.join(STEP_DIR, "holdout_1000_ids.txt")
OUTPUT_TRAIN    = os.path.join(STEP_DIR, "train_ids.txt")
OUTPUT_VAL      = os.path.join(STEP_DIR, "val_ids.txt")
OUTPUT_TEST     = os.path.join(STEP_DIR, "test_ids.txt")
OUTPUT_SUMMARY  = os.path.join(STEP_DIR, "split_summary.txt")

# ── 超参数 ────────────────────────────────────────────────────────────────────
RANDOM_SEED     = 42
HOLDOUT_TARGET  = 1000
TRAIN_RATIO     = 0.80
VAL_RATIO       = 0.10
TEST_RATIO      = 0.10

# ── 日志 ──────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(STEP_DIR, "step1.4.log"), encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def write_ids(filepath: str, ids: list):
    """每行写一个 mp_id。"""
    with open(filepath, "w", encoding="utf-8") as f:
        for mid in sorted(ids, key=lambda x: str(x)):
            f.write(f"{mid}\n")
    log.info(f"  → 写入 {len(ids)} 个 mp_id: {filepath}")


def stratified_sample_by_group(
    group_to_ids: dict[str, list],
    target_n: int,
    min_remaining: int = 2,
) -> list:
    """
    从各分组中按比例采样，总计约 target_n 个。
    约束：
    - 若某组只有 1 个 → 不放入采样
    - 若某组 2-3 个 → 最多放 1 个入采样（保证训练集至少剩 1 个）
    - 若某组 > 3 个 → 按比例，但该组在训练集中至少保留 min_remaining 个

    返回被采样的 id 列表。
    """
    total_eligible = sum(
        len(ids) for ids in group_to_ids.values() if len(ids) >= 2
    )
    if total_eligible == 0:
        log.warning("没有 ≥2 个成员的分组，无法构建保留集")
        return []

    sample_ratio = target_n / total_eligible

    sampled = []
    for group_label, ids in group_to_ids.items():
        n = len(ids)
        if n == 1:
            continue

        if n <= 3:
            # 最多取 1 个
            quota = 1
        else:
            # 按比例，但保证剩余 ≥ min_remaining
            quota = max(0, min(
                math.ceil(n * sample_ratio),
                n - min_remaining
            ))

        if quota > 0:
            chosen = random.sample(ids, quota)
            sampled.extend(chosen)

    return sampled


def adjust_to_target(sampled: list, group_to_ids: dict,
                      all_sampled_set: set, target_n: int) -> list:
    """
    若 stratified_sample 结果 < target_n，从最大组中补充；
    若 > target_n，随机删除到 target_n。
    """
    if len(sampled) >= target_n:
        random.shuffle(sampled)
        return sampled[:target_n]

    # 补充：从还未被采样的 id 中选
    candidates = []
    for group_label, ids in sorted(group_to_ids.items(),
                                    key=lambda x: -len(x[1])):
        for mid in ids:
            if mid not in all_sampled_set and mid not in set(sampled):
                candidates.append(mid)

    needed = target_n - len(sampled)
    extra = random.sample(candidates, min(needed, len(candidates)))
    sampled.extend(extra)
    log.info(f"  补充 {len(extra)} 个使保留集达到目标数量")
    return sampled


def stratified_train_val_test_split(
    ids: list,
    group_labels: dict,  # mp_id → group_label
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[list, list, list]:
    """
    按 group_label 分层划分 train/val/test。
    在每个组内，按 train:val:test 比例随机分配。
    """
    group_to_ids = defaultdict(list)
    for mid in ids:
        group_to_ids[group_labels.get(mid, "UNKNOWN")].append(mid)

    train_ids, val_ids, test_ids = [], [], []

    for group_label, group_ids in group_to_ids.items():
        random.shuffle(group_ids)
        n = len(group_ids)

        n_train = max(1, round(n * train_ratio))
        n_val   = max(0, round(n * val_ratio))
        # test gets the rest
        n_train = min(n_train, n)
        n_val   = min(n_val, n - n_train)
        n_test  = n - n_train - n_val

        train_ids.extend(group_ids[:n_train])
        val_ids.extend(group_ids[n_train:n_train + n_val])
        test_ids.extend(group_ids[n_train + n_val:])

    return train_ids, val_ids, test_ids


def main():
    log.info("=" * 60)
    log.info("Step 1.4 开始：数据集划分（保留集 + train/val/test）")
    log.info("=" * 60)

    # ── 1. 加载清单 ────────────────────────────────────────────────────────────
    if not os.path.exists(INPUT_INVENTORY):
        log.error(f"data_inventory.csv 不存在，请先运行 step1.1/1.3: {INPUT_INVENTORY}")
        return

    df = pd.read_csv(INPUT_INVENTORY, dtype={"mp_id": str, "site_id": str})
    log.info(f"清单总条目数: {len(df)}")

    # ── 2. 筛选有效条目：files_complete=True ──────────────────────────────────
    df_valid = df[df["files_complete"] == True].copy()
    log.info(f"files_complete=True 条目数: {len(df_valid)}")

    # ── 3. 按 mp_id 聚合，确定每个 mp_id 的质量状态 ──────────────────────────
    mp_tiers = df_valid.groupby("mp_id")["quality_tier"].apply(list)
    mp_elements = df_valid.groupby("mp_id")["element"].apply(
        lambda x: sorted(set(x.astype(str)))
    )

    # 排除所有位点均为 C 级的 mp_id
    excluded_mpids = set()
    valid_mpids    = []

    for mp_id, tiers in mp_tiers.items():
        non_c = [t for t in tiers if t != "C"]
        if len(non_c) == 0:
            # 全为 C 级 → 排除
            excluded_mpids.add(mp_id)
        else:
            valid_mpids.append(mp_id)

    log.info(f"有效 mp_id 总数（排除全 C 级）: {len(valid_mpids)}")
    log.info(f"排除的全 C 级 mp_id 数: {len(excluded_mpids)}")

    # ── 4. 生成元素组合标签（用于分层）────────────────────────────────────────
    group_labels = {}  # mp_id → 元素组合字符串（如 "Fe-Sc"）

    for mp_id in valid_mpids:
        elems = mp_elements.get(mp_id, [])
        # 过滤 UNKNOWN，排序后拼接
        clean_elems = sorted(set(e for e in elems if e != "UNKNOWN"))
        label = "-".join(clean_elems) if clean_elems else "UNKNOWN"
        group_labels[mp_id] = label

    # 按 group_label 分组
    group_to_ids = defaultdict(list)
    for mp_id in valid_mpids:
        group_to_ids[group_labels[mp_id]].append(mp_id)

    log.info(f"元素组合分组数: {len(group_to_ids)}")
    # 打印最大的10个分组
    top_groups = sorted(group_to_ids.items(), key=lambda x: -len(x[1]))[:10]
    for label, ids in top_groups:
        log.info(f"  {label}: {len(ids)} 个 mp_id")

    # ── 5. 生成保留集 ──────────────────────────────────────────────────────────
    log.info(f"生成保留集（目标 {HOLDOUT_TARGET} 个 mp_id）...")
    holdout_ids = stratified_sample_by_group(
        group_to_ids, HOLDOUT_TARGET, min_remaining=2
    )
    log.info(f"  分层采样初始结果: {len(holdout_ids)} 个"
             f"（{'超出' if len(holdout_ids) > HOLDOUT_TARGET else '不足'}目标，将自动修正）"
             if len(holdout_ids) != HOLDOUT_TARGET else
             f"  分层采样初始结果: {len(holdout_ids)} 个（恰好达标）")
    holdout_set = set(holdout_ids)

    # 无论超出还是不足，统一通过 adjust_to_target 修正到精确目标数
    # adjust_to_target 内部：超出时随机截断，不足时从大组补充
    holdout_ids = adjust_to_target(
        holdout_ids, group_to_ids, holdout_set, HOLDOUT_TARGET
    )
    holdout_set = set(holdout_ids)

    log.info(f"保留集最终数量: {len(holdout_ids)}")

    # ── 6. 剩余 mp_id → train/val/test ────────────────────────────────────────
    remaining_ids = [mid for mid in valid_mpids if mid not in holdout_set]
    log.info(f"剩余（非保留集）mp_id 数: {len(remaining_ids)}")

    train_ids, val_ids, test_ids = stratified_train_val_test_split(
        remaining_ids, group_labels, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
    )

    log.info(f"划分结果: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

    # ── 7. 验证：无交叉、无遗漏 ───────────────────────────────────────────────
    all_split_ids = holdout_ids + train_ids + val_ids + test_ids
    all_split_set = set(all_split_ids)

    # 检查重复
    assert len(all_split_ids) == len(all_split_set), \
        "❌ 集合划分存在重复 mp_id！请检查逻辑。"

    # 检查是否覆盖全部有效 mp_id
    valid_set = set(valid_mpids)
    missing = valid_set - all_split_set
    extra   = all_split_set - valid_set
    if missing:
        log.warning(f"⚠️  {len(missing)} 个有效 mp_id 未被分配到任何集合！")
    if extra:
        log.warning(f"⚠️  {len(extra)} 个 mp_id 在分配中出现但不在有效列表中！")

    assert not missing, "❌ 存在有效 mp_id 未被分配！"
    assert not extra,   "❌ 存在多余 mp_id！"
    log.info("✓ 集合覆盖验证通过：无重复、无遗漏")

    # ── 8. 写入 4 个 ID 文件 ──────────────────────────────────────────────────
    write_ids(OUTPUT_HOLDOUT, holdout_ids)
    write_ids(OUTPUT_TRAIN,   train_ids)
    write_ids(OUTPUT_VAL,     val_ids)
    write_ids(OUTPUT_TEST,    test_ids)

    # ── 9. 写入划分统计报告 ────────────────────────────────────────────────────
    # 计算保留集元素组合覆盖率
    holdout_groups  = {group_labels[mid] for mid in holdout_ids}
    all_groups      = set(group_to_ids.keys())
    coverage_pct    = len(holdout_groups) / len(all_groups) * 100 if all_groups else 0

    # 训练集中与保留集有相同元素组合的化合物数
    holdout_groups_set = holdout_groups
    train_neighbors = sum(
        1 for mid in train_ids
        if group_labels.get(mid, "") in holdout_groups_set
    )

    with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as f:
        f.write("Step 1.4 数据集划分统计报告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"总有效 mp_id 数（排除全 C 级）:  {len(valid_mpids)}\n")
        f.write(f"排除的全 C 级 mp_id 数:          {len(excluded_mpids)}\n\n")
        f.write(f"保留集（holdout）:               {len(holdout_ids)}\n")
        f.write(f"训练集:                          {len(train_ids)}\n")
        f.write(f"验证集:                          {len(val_ids)}\n")
        f.write(f"测试集:                          {len(test_ids)}\n\n")
        f.write(f"合计（验证）:                    "
                f"{len(holdout_ids)+len(train_ids)+len(val_ids)+len(test_ids)}"
                f" = {len(valid_mpids)}\n\n")

        f.write(f"元素组合分组总数:                {len(all_groups)}\n")
        f.write(f"保留集元素组合覆盖率:            "
                f"{len(holdout_groups)}/{len(all_groups)} = {coverage_pct:.1f}%\n")
        f.write(f"训练集中与保留集相邻的化合物数:   {train_neighbors}\n\n")

        f.write("── 验收状态 ──\n")
        holdout_ok = 990 <= len(holdout_ids) <= 1010
        f.write(f"  保留集大小 990-1010: {'✓ PASS' if holdout_ok else '✗ FAIL'} ({len(holdout_ids)})\n")
        f.write(f"  无重复 / 无遗漏:    ✓ PASS\n")

    log.info(f"✓ split_summary.txt 已写入: {OUTPUT_SUMMARY}")

    # ── 10. 最终摘要 ────────────────────────────────────────────────────────────
    log.info("\n══ Step 1.4 执行摘要 ══")
    log.info(f"  总有效 mp_id:   {len(valid_mpids)}")
    log.info(f"  holdout:        {len(holdout_ids)}")
    log.info(f"  train:          {len(train_ids)}")
    log.info(f"  val:            {len(val_ids)}")
    log.info(f"  test:           {len(test_ids)}")
    log.info(f"  保留集覆盖率:    {coverage_pct:.1f}%")
    log.info("Step 1.4 完成。")


if __name__ == "__main__":
    main()