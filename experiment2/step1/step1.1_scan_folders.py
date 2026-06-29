# step1.1_scan_folders.py
# 任务：遍历 DATA_ROOT，检查三文件完整性，输出 step1_raw_scan.csv

import os
import csv
import re
from pathlib import Path
from tqdm import tqdm

# ── 路径常量 ──────────────────────────────────────────────────────────────────
DATA_ROOT  = r"C:\Users\T-Cat\Desktop\DiffCSP-main\site_dataset_Fe_only_oxide_one_site"
STEP1_DIR  = r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment2\step1"
OUTPUT_CSV = os.path.join(STEP1_DIR, "step1_raw_scan.csv")

# ── 文件夹名解析 ──────────────────────────────────────────────────────────────
# 格式：mp_{id}_{formula}_feff_Fe_site_{nn}
# 例如：mp_204_CeFe2_feff_Fe_site_02
#       mp_13494_Nd3Fe29_feff_Fe_site_04
FOLDER_RE = re.compile(
    r'^(mp_\d+)_(.+?)_feff_Fe_site_(\d{2})$'
)

def parse_folder_name(name: str):
    """返回 (mp_id, formula, site_nn) 或 None"""
    m = FOLDER_RE.match(name)
    if m:
        return m.group(1), m.group(2), m.group(3)
    return None, None, None

# ── 主扫描逻辑 ────────────────────────────────────────────────────────────────
def main():
    os.makedirs(STEP1_DIR, exist_ok=True)

    data_root = Path(DATA_ROOT)
    if not data_root.exists():
        raise FileNotFoundError(f"DATA_ROOT 不存在: {DATA_ROOT}")

    # 只取直接子目录
    all_entries = [e for e in data_root.iterdir() if e.is_dir()]
    print(f"发现子目录总数：{len(all_entries)}")

    rows = []
    unparsed = 0

    for entry in tqdm(all_entries, desc="扫描文件夹"):
        folder_name = entry.name
        mp_id, formula, site_nn = parse_folder_name(folder_name)

        if mp_id is None:
            unparsed += 1
            # 仍然记录，便于排查
            rows.append({
                "folder_name":  folder_name,
                "folder_path":  str(entry),
                "mp_id":        "PARSE_FAIL",
                "formula":      "PARSE_FAIL",
                "site_nn":      "PARSE_FAIL",
                "has_chi1":     False,
                "has_xmu":      False,
                "has_poscar":   False,
            })
            continue

        has_chi1   = (entry / "chi1.dat").is_file()
        has_xmu    = (entry / "xmu.dat").is_file()
        has_poscar = (entry / "POSCAR_supercell_fixed").is_file()

        rows.append({
            "folder_name":  folder_name,
            "folder_path":  str(entry),
            "mp_id":        mp_id,
            "formula":      formula,
            "site_nn":      site_nn,
            "has_chi1":     has_chi1,
            "has_xmu":      has_xmu,
            "has_poscar":   has_poscar,
        })

    # ── 写出 CSV ──
    fieldnames = ["folder_name", "folder_path", "mp_id", "formula",
                  "site_nn", "has_chi1", "has_xmu", "has_poscar"]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # ── 统计摘要 ──
    total      = len(rows)
    parsed_ok  = total - unparsed
    all_three  = sum(1 for r in rows
                     if r["has_chi1"] and r["has_xmu"] and r["has_poscar"])
    miss_chi1  = sum(1 for r in rows if not r["has_chi1"]  and r["mp_id"] != "PARSE_FAIL")
    miss_xmu   = sum(1 for r in rows if not r["has_xmu"]   and r["mp_id"] != "PARSE_FAIL")
    miss_poscar= sum(1 for r in rows if not r["has_poscar"] and r["mp_id"] != "PARSE_FAIL")

    print("\n======= Step 1.1 扫描结果 =======")
    print(f"子目录总数          : {total}")
    print(f"文件夹名解析成功    : {parsed_ok}  |  解析失败: {unparsed}")
    print(f"三文件全齐          : {all_three}")
    print(f"缺 chi1.dat         : {miss_chi1}")
    print(f"缺 xmu.dat          : {miss_xmu}")
    print(f"缺 POSCAR_supercell : {miss_poscar}")
    print(f"\n输出文件 → {OUTPUT_CSV}")

if __name__ == "__main__":
    main()