# =============================================================================
# 脚本编号: step1.1
# 脚本名称: step1.1_scan_and_inventory.py
# 输入:
#   - C:\Users\T-Cat\Desktop\XAS-FeO\site_dataset\               (主数据集)
#   - C:\Users\T-Cat\Desktop\XAS-FeO\test_missing_keep3_packed_A\ (离子数据集)
# 输出:
#   - C:\Users\T-Cat\Desktop\DiffCSP-main\experiment\step1\data_inventory.csv
#   - C:\Users\T-Cat\Desktop\DiffCSP-main\experiment\step1\dedup_report.txt
# 说明:
#   扫描两个数据文件夹，解析文件夹名（含特殊无元素格式），执行去重（仅清单操作，
#   不删磁盘文件），验证三文件完整性，生成数据清单 CSV。
#   ⚠️ 严禁将 formula 字段写入任何输出文件。
# =============================================================================

import os
import re
import csv
import logging
from pathlib import Path
from typing import Optional, List

# ── 路径常量 ──────────────────────────────────────────────────────────────────
PROJECT_ROOT   = r"C:\Users\T-Cat\Desktop\DiffCSP-main"
EXPERIMENT_DIR = os.path.join(PROJECT_ROOT, "experiment")
STEP_DIR       = os.path.join(EXPERIMENT_DIR, "step1")
os.makedirs(STEP_DIR, exist_ok=True)

SITE_DATASET_DIR  = r"C:\Users\T-Cat\Desktop\XAS-FeO\site_dataset"
IONIC_DATASET_DIR = r"C:\Users\T-Cat\Desktop\XAS-FeO\test_missing_keep3_packed_A"

OUTPUT_INVENTORY = os.path.join(STEP_DIR, "data_inventory.csv")
OUTPUT_DEDUP     = os.path.join(STEP_DIR, "dedup_report.txt")

# ── 日志 ──────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(STEP_DIR, "step1.1.log"), encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── 非金属元素集合（用于从 formula 推断吸收元素）──────────────────────────────
NON_METALS = {
    "O", "N", "C", "H", "S", "P", "F", "Cl", "Br", "I",
    "Se", "Te", "At", "B", "Si", "Ge", "As", "Sb", "Po",
    "He", "Ne", "Ar", "Kr", "Xe", "Rn",
}

# ── 必须存在的三个文件名 ───────────────────────────────────────────────────────
REQUIRED_FILES = {"chi.dat", "xmu.dat", "POSCAR_supercell_fixed"}


def extract_element_tokens(formula_str: str) -> List[str]:
    """从 formula 字符串中提取所有元素符号（大写字母开头+小写字母）。"""
    return re.findall(r"[A-Z][a-z]?", formula_str)


def infer_element_from_formula(formula_str: str) -> str:
    """
    从 formula 字符串推断唯一金属元素。
    返回元素符号字符串，或 'UNKNOWN' 若无法确定。
    """
    tokens = extract_element_tokens(formula_str)
    metals = [t for t in tokens if t not in NON_METALS]
    unique_metals = list(dict.fromkeys(metals))  # 保持顺序去重
    if len(unique_metals) == 1:
        return unique_metals[0]
    else:
        return "UNKNOWN"


def parse_folder_name(folder_name: str) -> Optional[dict]:
    """
    解析文件夹名，提取 mp_id、element、site_id。
    返回字典或 None（解析失败时）。
    ⚠️ formula 字段仅在推断 element 时临时使用，绝不写入输出。
    """
    # 标准格式: mp_{mp_id}_{formula}__feff_{element}_site_{nn}
    # 特殊格式: mp_{mp_id}_{formula}__feff_site_{nn}
    pattern = r"^mp_(\d+)_(.+?)__feff_(?:([A-Za-z]+)_)?site_(\d{2})$"
    m = re.match(pattern, folder_name)
    if not m:
        return None

    mp_id    = m.group(1)
    formula  = m.group(2)   # ⚠️ 仅临时使用，不输出
    element  = m.group(3)   # 可能为 None（特殊格式）
    site_id  = m.group(4)

    if element is None:
        # 特殊格式：从 formula 推断唯一金属元素
        element = infer_element_from_formula(formula)

    return {
        "mp_id":   mp_id,
        "element": element,
        "site_id": site_id,
        # formula 不存入字典，防止后续误用
    }


def check_files_complete(folder_path: str) -> bool:
    """检查文件夹内三个必要文件是否都存在。"""
    existing = set(os.listdir(folder_path))
    return REQUIRED_FILES.issubset(existing)


def scan_dataset(dataset_dir: str, is_ionic: bool) -> List[dict]:
    """扫描一个数据集目录，返回条目列表。"""
    records = []
    try:
        entries = os.listdir(dataset_dir)
    except FileNotFoundError:
        log.error(f"目录不存在: {dataset_dir}")
        return records

    for folder_name in entries:
        folder_path = os.path.join(dataset_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        parsed = parse_folder_name(folder_name)
        if parsed is None:
            log.warning(f"无法解析文件夹名，跳过: {folder_name}")
            continue

        complete = check_files_complete(folder_path)

        records.append({
            "folder_name":    folder_name,
            "mp_id":          parsed["mp_id"],
            "element":        parsed["element"],
            "site_id":        parsed["site_id"],
            "is_ionic":       is_ionic,
            "source_path":    folder_path,
            "source_dataset": "ionic" if is_ionic else "site_dataset",
            "files_complete": complete,
            "quality_tier":   float("nan"),  # Step 1.3 填充
        })

    return records


def main():
    log.info("=" * 60)
    log.info("Step 1.1 开始：扫描文件夹 + 去重 + 建立数据清单")
    log.info("=" * 60)

    # ── 1. 扫描两个数据集 ──────────────────────────────────────────────────────
    log.info(f"扫描主数据集: {SITE_DATASET_DIR}")
    site_records  = scan_dataset(SITE_DATASET_DIR,  is_ionic=False)
    log.info(f"  → {len(site_records)} 个文件夹")

    log.info(f"扫描离子数据集: {IONIC_DATASET_DIR}")
    ionic_records = scan_dataset(IONIC_DATASET_DIR, is_ionic=True)
    log.info(f"  → {len(ionic_records)} 个文件夹")

    # ── 2. 去重（以 folder_name 为 key，保留 site_dataset 中的条目）────────────
    site_names = {r["folder_name"] for r in site_records}
    deduplicated_ionic = []
    removed_duplicates  = []

    for r in ionic_records:
        if r["folder_name"] in site_names:
            removed_duplicates.append(r["folder_name"])
        else:
            deduplicated_ionic.append(r)

    log.info(f"去重：从 ionic 数据集移除 {len(removed_duplicates)} 条重复条目（仅清单，不删磁盘文件）")

    # ── 3. 合并清单 ────────────────────────────────────────────────────────────
    all_records = site_records + deduplicated_ionic

    # 最终验证唯一性
    all_names = [r["folder_name"] for r in all_records]
    assert len(all_names) == len(set(all_names)), "合并后仍有重复 folder_name，请检查逻辑！"
    log.info(f"合并后总条目数: {len(all_records)}")

    # ── 4. 统计 UNKNOWN element 和文件缺失情况 ────────────────────────────────
    unknown_count   = sum(1 for r in all_records if r["element"] == "UNKNOWN")
    incomplete_count = sum(1 for r in all_records if not r["files_complete"])
    total = len(all_records)

    unknown_pct = unknown_count / total * 100 if total > 0 else 0
    log.info(f"element=UNKNOWN 条目数: {unknown_count} ({unknown_pct:.2f}%)")
    log.info(f"files_complete=False 条目数: {incomplete_count}")

    if unknown_pct >= 1.0:
        log.warning(
            f"⚠️  UNKNOWN 比例 {unknown_pct:.2f}% >= 1%，请检查文件夹名解析逻辑！"
        )
    else:
        log.info("✓ UNKNOWN 比例 < 1%，解析逻辑正常")

    # 列出 UNKNOWN 示例（最多 20 个）
    unknown_examples = [r["folder_name"] for r in all_records if r["element"] == "UNKNOWN"][:20]
    if unknown_examples:
        log.warning(f"UNKNOWN 示例（最多20条）:\n" + "\n".join(f"  {x}" for x in unknown_examples))

    # ── 5. 写入 data_inventory.csv ────────────────────────────────────────────
    fieldnames = [
        "folder_name", "mp_id", "element", "site_id",
        "is_ionic", "source_path", "source_dataset",
        "files_complete", "quality_tier",
    ]
    with open(OUTPUT_INVENTORY, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_records:
            writer.writerow(r)
    log.info(f"✓ data_inventory.csv 已写入: {OUTPUT_INVENTORY}")

    # ── 6. 写入去重报告 ────────────────────────────────────────────────────────
    with open(OUTPUT_DEDUP, "w", encoding="utf-8") as f:
        f.write("Step 1.1 去重报告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"主数据集 (site_dataset) 文件夹总数:   {len(site_records)}\n")
        f.write(f"离子数据集 (ionic) 原始文件夹总数:    {len(ionic_records)}\n")
        f.write(f"从 ionic 移除的重复条目数:           {len(removed_duplicates)}\n")
        f.write(f"合并后总条目数:                     {total}\n\n")
        f.write(f"files_complete=False 条目数:        {incomplete_count}\n")
        f.write(f"element=UNKNOWN 条目数:             {unknown_count} ({unknown_pct:.2f}%)\n\n")

        if removed_duplicates:
            f.write("被移除的重复文件夹名列表（仅从清单删除，磁盘文件未动）:\n")
            for name in removed_duplicates:
                f.write(f"  {name}\n")

        if unknown_examples:
            f.write(f"\nelement=UNKNOWN 示例（最多20条）:\n")
            for name in unknown_examples:
                f.write(f"  {name}\n")

    log.info(f"✓ dedup_report.txt 已写入: {OUTPUT_DEDUP}")

    # ── 7. 最终摘要 ────────────────────────────────────────────────────────────
    is_ionic_count = sum(1 for r in all_records if r["is_ionic"])
    log.info("\n══ Step 1.1 执行摘要 ══")
    log.info(f"  总条目数（去重后）: {total}")
    log.info(f"  is_ionic=True:     {is_ionic_count}")
    log.info(f"  is_ionic=False:    {total - is_ionic_count}")
    log.info(f"  files_complete:    {total - incomplete_count} / {total}")
    log.info(f"  element=UNKNOWN:   {unknown_count} ({unknown_pct:.2f}%)")
    log.info("Step 1.1 完成。")


if __name__ == "__main__":
    main()