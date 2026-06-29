# =============================================================================
# 脚本编号: step1.3
# 脚本名称: step1.3_quality_grading.py
# 输入:
#   - C:\Users\T-Cat\Desktop\DiffCSP-main\experiment\step1\data_inventory.csv
#   - C:\Users\T-Cat\Desktop\XAS-FeO\feff_features_all_site_v2.csv
#   - C:\Users\T-Cat\Desktop\XAS-FeO\feff_features_all_ionic_v3.csv
#   - C:\Users\T-Cat\Desktop\XAS-FeO\all_center_neighbors_summary.csv
# 输出:
#   - C:\Users\T-Cat\Desktop\DiffCSP-main\experiment\step1\data_inventory.csv  (更新 quality_tier)
#   - C:\Users\T-Cat\Desktop\DiffCSP-main\experiment\step1\quality_summary.txt
#   - C:\Users\T-Cat\Desktop\DiffCSP-main\experiment\step1\bond_length_constraints.json
# 说明:
#   合并两个特征表（site_v2 优先，ionic_v3 去重），按 flag_*_valid 列分级（A/B/C）；
#   同时解析键长约束表并序列化为 JSON。所有操作不修改原始数据文件。
# =============================================================================

import os
import json
import logging
import re
import pandas as pd
import numpy as np
from typing import Optional, Tuple

# ── 路径常量 ──────────────────────────────────────────────────────────────────
PROJECT_ROOT   = r"C:\Users\T-Cat\Desktop\DiffCSP-main"
EXPERIMENT_DIR = os.path.join(PROJECT_ROOT, "experiment")
STEP_DIR       = os.path.join(EXPERIMENT_DIR, "step1")
os.makedirs(STEP_DIR, exist_ok=True)

INPUT_INVENTORY     = os.path.join(STEP_DIR, "data_inventory.csv")
OUTPUT_INVENTORY    = INPUT_INVENTORY   # 原地更新
OUTPUT_QUALITY_SUMMARY  = os.path.join(STEP_DIR, "quality_summary.txt")
OUTPUT_BOND_JSON        = os.path.join(STEP_DIR, "bond_length_constraints.json")

FEATURE_SITE_CSV    = r"C:\Users\T-Cat\Desktop\XAS-FeO\feff_features_all_site_v2.csv"
FEATURE_IONIC_CSV   = r"C:\Users\T-Cat\Desktop\XAS-FeO\feff_features_all_ionic_v3.csv"
BOND_CONSTRAINT_CSV = r"C:\Users\T-Cat\Desktop\XAS-FeO\all_center_neighbors_summary.csv"

# ── 日志 ──────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(STEP_DIR, "step1.3.log"), encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ── 质量分级函数 ───────────────────────────────────────────────────────────────

def assign_quality_tier(row) -> str:
    """
    根据 flag_*_valid 列分配质量等级：
      A → 三个 flag 均为 1
      B → flag_white_valid=1（至少白线有效）
      C → 其余
    注意：离子元素（Li、Na 等）天然 flag_pre_valid=0，这是物理正常现象，
          质量仍可为 B 级，不应强制降至 C。
    """
    pre   = row.get("flag_pre_valid",   0)
    white = row.get("flag_white_valid", 0)
    post  = row.get("flag_post_valid",  0)

    # 处理 NaN
    pre   = 0 if pd.isna(pre)   else int(pre)
    white = 0 if pd.isna(white) else int(white)
    post  = 0 if pd.isna(post)  else int(post)

    if pre == 1 and white == 1 and post == 1:
        return "A"
    elif white == 1:
        return "B"
    else:
        return "C"


# ── 键长约束解析函数 ──────────────────────────────────────────────────────────

def parse_range_str(range_str: str) -> Optional[Tuple[float, float]]:
    """
    解析 "2.511-2.996" 格式为 (2.511, 2.996)。
    处理边界情况：
    - 可能含空格："2.511 - 2.996"
    - 负值（如 "-0.5-1.2"）：用正则处理
    """
    if pd.isna(range_str):
        return None
    s = str(range_str).strip()

    # 用正则匹配两个浮点数（支持负号）
    m = re.match(r"^(-?\d+\.?\d*)\s*-\s*(-?\d+\.?\d*)$", s)
    if m:
        try:
            v_min = float(m.group(1))
            v_max = float(m.group(2))
            if v_min <= v_max:
                return (v_min, v_max)
        except ValueError:
            pass

    # 备用方案：split by '-' 但处理负数
    parts = s.split("-")
    if len(parts) == 2:
        try:
            return (float(parts[0].strip()), float(parts[1].strip()))
        except ValueError:
            pass
    elif len(parts) == 3 and s.startswith("-"):
        try:
            return (float("-" + parts[1].strip()), float(parts[2].strip()))
        except ValueError:
            pass

    log.warning(f"无法解析键长范围字符串: '{range_str}'")
    return None


def read_csv_auto_encoding(filepath: str) -> Optional[pd.DataFrame]:
    """
    按优先顺序尝试多种编码读取 CSV。
    0xff 开头的文件通常是 UTF-16 LE BOM 编码。
    """
    for enc in ("utf-16", "utf-8-sig", "gbk", "latin-1"):
        try:
            df = pd.read_csv(filepath, encoding=enc)
            log.info(f"文件编码检测成功: {enc}  ({filepath})")
            return df
        except (UnicodeDecodeError, Exception):
            continue
    return None


def main():
    log.info("=" * 60)
    log.info("Step 1.3 开始：数据质量分级 + 键长约束序列化")
    log.info("=" * 60)

    # ── 1. 加载 data_inventory.csv ────────────────────────────────────────────
    if not os.path.exists(INPUT_INVENTORY):
        log.error(f"data_inventory.csv 不存在，请先运行 step1.1: {INPUT_INVENTORY}")
        return

    df_inv = pd.read_csv(INPUT_INVENTORY, dtype={"mp_id": str, "site_id": str})
    total_inventory = len(df_inv)
    log.info(f"清单条目数: {total_inventory}")

    # ── 2. 加载两个特征表 ──────────────────────────────────────────────────────
    log.info(f"加载 site 特征表: {FEATURE_SITE_CSV}")
    df_site = pd.read_csv(FEATURE_SITE_CSV)
    log.info(f"  → {len(df_site)} 行")

    log.info(f"加载 ionic 特征表: {FEATURE_IONIC_CSV}")
    df_ionic = pd.read_csv(FEATURE_IONIC_CSV)
    log.info(f"  → {len(df_ionic)} 行")

    # ── 3. 合并去重（site_v2 优先，删除 ionic_v3 中重复条目）─────────────────
    if "sample_name" not in df_site.columns:
        log.error("site 特征表中找不到 'sample_name' 列！请检查 CSV 列名。")
        log.info(f"site 表列名: {list(df_site.columns[:10])}")
        return

    if "sample_name" not in df_ionic.columns:
        log.error("ionic 特征表中找不到 'sample_name' 列！请检查 CSV 列名。")
        log.info(f"ionic 表列名: {list(df_ionic.columns[:10])}")
        return

    site_names_set  = set(df_site["sample_name"].astype(str))
    ionic_before    = len(df_ionic)
    df_ionic_dedup  = df_ionic[~df_ionic["sample_name"].astype(str).isin(site_names_set)].copy()
    ionic_removed   = ionic_before - len(df_ionic_dedup)
    log.info(f"从 ionic 特征表移除重复条目: {ionic_removed}")

    df_features = pd.concat([df_site, df_ionic_dedup], ignore_index=True)
    log.info(f"合并后特征表总行数: {len(df_features)}")

    feat_dict = {}
    for _, row in df_features.iterrows():
        name = str(row["sample_name"])
        if name not in feat_dict:
            feat_dict[name] = row

    log.info(f"特征字典唯一 sample_name 数: {len(feat_dict)}")

    # ── 4. 为每个 folder_name 分配 quality_tier ───────────────────────────────
    tiers = []
    matched   = 0
    unmatched = 0

    for _, inv_row in df_inv.iterrows():
        fname = str(inv_row["folder_name"])
        if fname in feat_dict:
            feat_row = feat_dict[fname]
            tier = assign_quality_tier(feat_row)
            matched += 1
        else:
            tier = "unknown"
            unmatched += 1
        tiers.append(tier)

    df_inv["quality_tier"] = tiers
    log.info(f"匹配到特征: {matched} 条，未匹配: {unmatched} 条（标记为 unknown）")

    # ── 5. 保存更新后的 data_inventory.csv ────────────────────────────────────
    df_inv.to_csv(OUTPUT_INVENTORY, index=False)
    log.info(f"✓ data_inventory.csv 已更新（quality_tier 列填充）: {OUTPUT_INVENTORY}")

    # ── 6. 生成质量分布统计 ────────────────────────────────────────────────────
    tier_counts = df_inv["quality_tier"].value_counts()
    total = len(df_inv)

    def pct(n):
        return f"{n / total * 100:.1f}%" if total > 0 else "N/A"

    n_A       = tier_counts.get("A",       0)
    n_B       = tier_counts.get("B",       0)
    n_C       = tier_counts.get("C",       0)
    n_unknown = tier_counts.get("unknown", 0)

    ab_total = n_A + n_B
    ab_pct   = ab_total / total * 100 if total > 0 else 0

    df_ionic_rows = df_inv[df_inv["is_ionic"] == True]
    df_coval_rows = df_inv[df_inv["is_ionic"] == False]

    def tier_dist(df_sub):
        n = len(df_sub)
        if n == 0:
            return "（无数据）"
        tc = df_sub["quality_tier"].value_counts()
        parts = []
        for t in ["A", "B", "C", "unknown"]:
            cnt = tc.get(t, 0)
            parts.append(f"{t}: {cnt} ({cnt/n*100:.1f}%)")
        return ",  ".join(parts)

    with open(OUTPUT_QUALITY_SUMMARY, "w", encoding="utf-8") as f:
        f.write("Step 1.3 质量分级统计报告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"总条目数:                 {total}\n")
        f.write(f"A 级（全有效）:            {n_A} ({pct(n_A)})\n")
        f.write(f"B 级（白线有效）:           {n_B} ({pct(n_B)})\n")
        f.write(f"C 级（异常）:              {n_C} ({pct(n_C)})\n")
        f.write(f"unknown（未找到特征）:      {n_unknown} ({pct(n_unknown)})\n\n")
        f.write(f"A+B 合计:                 {ab_total} ({ab_pct:.1f}%)\n\n")
        f.write(f"is_ionic=True  中的质量分布:\n  {tier_dist(df_ionic_rows)}\n\n")
        f.write(f"is_ionic=False 中的质量分布:\n  {tier_dist(df_coval_rows)}\n\n")
        f.write("── 验收状态 ──\n")
        pass_ab = ab_pct >= 60.0
        f.write(f"  A+B 合计 ≥ 60%: {'✓ PASS' if pass_ab else '✗ FAIL — 数据质量异常，请报告给用户'}\n")

    log.info(f"✓ quality_summary.txt 已写入: {OUTPUT_QUALITY_SUMMARY}")

    if ab_pct < 60.0:
        log.warning(f"⚠️  A+B 合计 {ab_pct:.1f}% < 60%！请向用户报告！")
    else:
        log.info(f"✓ A+B 合计 {ab_pct:.1f}% ≥ 60%，数据质量正常")

    # ── 7. 处理键长约束表 ─────────────────────────────────────────────────────
    log.info(f"加载键长约束表: {BOND_CONSTRAINT_CSV}")
    if not os.path.exists(BOND_CONSTRAINT_CSV):
        log.error(f"键长约束表不存在: {BOND_CONSTRAINT_CSV}")
        return

    df_bond = read_csv_auto_encoding(BOND_CONSTRAINT_CSV)
    if df_bond is None:
        log.error(f"无法以任何已知编码读取键长约束表: {BOND_CONSTRAINT_CSV}")
        return
    log.info(f"  → {len(df_bond)} 行，列名: {list(df_bond.columns)}")

    col_pair = None
    col_range = None

    for c in df_bond.columns:
        if "pair" in c.lower():
            col_pair = c
        if "raw_range" in c.lower() or "minmax" in c.lower():
            col_range = c

    if col_pair is None and len(df_bond.columns) > 1:
        col_pair = df_bond.columns[1]
        log.warning(f"未找到 'pair' 列，按位置使用第2列: '{col_pair}'")
    if col_range is None and len(df_bond.columns) > 5:
        col_range = df_bond.columns[5]
        log.warning(f"未找到 'raw_range' 列，按位置使用第6列: '{col_range}'")

    if col_pair is None or col_range is None:
        log.error(f"无法确定 pair 或 range 列，键长约束表列名: {list(df_bond.columns)}")
        return

    log.info(f"使用列: pair='{col_pair}', range='{col_range}'")

    bond_dict = {}
    parse_failed = 0
    for _, row in df_bond.iterrows():
        pair_str  = str(row[col_pair]).strip()
        range_str = row[col_range]
        parsed = parse_range_str(range_str)
        if parsed is not None:
            bond_dict[pair_str] = list(parsed)  # JSON 用 list 不用 tuple
        else:
            parse_failed += 1

    log.info(f"键长约束解析完成: {len(bond_dict)} 条成功，{parse_failed} 条解析失败")

    with open(OUTPUT_BOND_JSON, "w", encoding="utf-8") as f:
        json.dump(bond_dict, f, indent=2, ensure_ascii=False)

    log.info(f"✓ bond_length_constraints.json 已写入: {OUTPUT_BOND_JSON}")

    with open(OUTPUT_BOND_JSON, "r", encoding="utf-8") as f:
        test_load = json.load(f)
    assert len(test_load) == len(bond_dict), "JSON 写入验证失败！"
    log.info("✓ bond_length_constraints.json 验证通过（可正常 json.load）")

    # ── 8. 最终摘要 ────────────────────────────────────────────────────────────
    log.info("\n══ Step 1.3 执行摘要 ══")
    log.info(f"  清单总条目数: {total}")
    log.info(f"  A: {n_A} ({pct(n_A)}), B: {n_B} ({pct(n_B)}), "
             f"C: {n_C} ({pct(n_C)}), unknown: {n_unknown} ({pct(n_unknown)})")
    log.info(f"  A+B 合计: {ab_total} ({ab_pct:.1f}%)")
    log.info(f"  键长约束条数: {len(bond_dict)}")
    log.info("Step 1.3 完成。")


if __name__ == "__main__":
    main()