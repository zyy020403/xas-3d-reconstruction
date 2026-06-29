# =============================================================================
# 脚本编号: step1.2
# 脚本名称: step1.2_validate_file_formats.py
# 输入:
#   - C:\Users\T-Cat\Desktop\DiffCSP-main\experiment\step1\data_inventory.csv
# 输出:
#   - C:\Users\T-Cat\Desktop\DiffCSP-main\experiment\step1\format_validation_report.txt
# 说明:
#   抽样验证 chi.dat（k 空间谱）、xmu.dat（能量空间谱）、POSCAR_supercell_fixed
#   三类文件的格式可读性。发现异常时记录路径，不修改任何原始文件。
#   POSCAR 使用 pymatgen.core.Structure.from_file() 读取（与 DiffCSP 同库）。
# =============================================================================

import os
import random
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# ── 路径常量 ──────────────────────────────────────────────────────────────────
PROJECT_ROOT   = r"C:\Users\T-Cat\Desktop\DiffCSP-main"
EXPERIMENT_DIR = os.path.join(PROJECT_ROOT, "experiment")
STEP_DIR       = os.path.join(EXPERIMENT_DIR, "step1")
os.makedirs(STEP_DIR, exist_ok=True)

INPUT_INVENTORY = os.path.join(STEP_DIR, "data_inventory.csv")
OUTPUT_REPORT   = os.path.join(STEP_DIR, "format_validation_report.txt")

# ── 抽样大小 ──────────────────────────────────────────────────────────────────
N_CHI_SAMPLE   = 200
N_XMU_SAMPLE   = 200
N_POSCAR_SAMPLE = 100

# ── 随机种子 ──────────────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── 日志 ──────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(STEP_DIR, "step1.2.log"), encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ── 验证函数 ──────────────────────────────────────────────────────────────────

def validate_chi_dat(filepath: str) -> tuple[bool, str]:
    """
    验证 chi.dat 格式：
    - 跳过 '#' 注释行后，剩余行可解析为 ≥2 列浮点数
    - 第一列 k 值大致在 -2 到 22 Å⁻¹
    - 数据点数 > 100
    返回 (is_valid, reason)
    """
    try:
        data_lines = []
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith("#") or stripped == "":
                    continue
                data_lines.append(stripped)

        if len(data_lines) <= 100:
            return False, f"数据点数 {len(data_lines)} ≤ 100"

        # 尝试解析前 10 行验证格式
        for line in data_lines[:10]:
            parts = line.split()
            if len(parts) < 2:
                return False, f"列数不足2列: '{line[:60]}'"
            float(parts[0])  # k 值
            float(parts[1])  # chi 值

        # 检查 k 范围（用前/后几行）
        check_lines = data_lines[:5] + data_lines[-5:]
        k_vals = []
        for line in check_lines:
            parts = line.split()
            if len(parts) >= 1:
                try:
                    k_vals.append(float(parts[0]))
                except ValueError:
                    pass

        if k_vals:
            k_min, k_max = min(k_vals), max(k_vals)
            if k_min < -5 or k_max > 30:
                return False, f"k 范围异常: [{k_min:.2f}, {k_max:.2f}]"

        return True, "OK"

    except Exception as e:
        return False, str(e)


def validate_xmu_dat(filepath: str) -> tuple[bool, str]:
    """
    验证 xmu.dat 格式：
    - 跳过注释行后，剩余行可解析为 ≥2 列浮点数
    - 第一列为能量（eV），数值合理可读即可（不同元素 K 边位置差异很大）
    - 数据点数 > 50
    """
    try:
        data_lines = []
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith("#") or stripped == "":
                    continue
                data_lines.append(stripped)

        if len(data_lines) <= 50:
            return False, f"数据点数 {len(data_lines)} ≤ 50"

        # 验证前 10 行格式
        for line in data_lines[:10]:
            parts = line.split()
            if len(parts) < 2:
                return False, f"列数不足2列: '{line[:60]}'"
            float(parts[0])  # energy
            float(parts[1])  # xmu

        # 简单验证能量值是正数
        e_val = float(data_lines[0].split()[0])
        if e_val < 0:
            return False, f"能量首值为负数: {e_val}"

        return True, "OK"

    except Exception as e:
        return False, str(e)


def validate_poscar(filepath: str) -> tuple[bool, str]:
    """
    验证 POSCAR_supercell_fixed 格式：
    使用 pymatgen.core.Structure.from_file() 读取（与 DiffCSP data_utils.py 同库）。
    """
    try:
        from pymatgen.core.structure import Structure
        struct = Structure.from_file(filepath)
        if struct.num_sites == 0:
            return False, "Structure 读取成功但原子数为 0"
        return True, f"OK (num_sites={struct.num_sites})"
    except ImportError:
        return False, "pymatgen 未安装"
    except Exception as e:
        return False, str(e)


def sample_and_validate(df_complete: pd.DataFrame, n: int, file_name: str,
                        validate_fn, label: str) -> tuple[list[str], int]:
    """
    从 files_complete=True 的条目中随机抽取 n 条，对指定文件名运行验证函数。
    返回 (异常文件路径列表, 成功数量)
    """
    candidates = df_complete.sample(n=min(n, len(df_complete)),
                                    random_state=RANDOM_SEED).copy()
    errors = []
    ok_count = 0

    for _, row in candidates.iterrows():
        filepath = os.path.join(row["source_path"], file_name)
        if not os.path.exists(filepath):
            errors.append(f"[FILE_MISSING] {filepath}")
            continue

        is_valid, reason = validate_fn(filepath)
        if is_valid:
            ok_count += 1
        else:
            errors.append(f"[{reason}] {filepath}")

    log.info(f"  {label}: 抽样 {len(candidates)} 个，成功 {ok_count}，异常 {len(errors)}")
    return errors, ok_count


def main():
    log.info("=" * 60)
    log.info("Step 1.2 开始：抽样验证文件格式")
    log.info("=" * 60)

    # ── 加载清单 ───────────────────────────────────────────────────────────────
    if not os.path.exists(INPUT_INVENTORY):
        log.error(f"data_inventory.csv 不存在，请先运行 step1.1: {INPUT_INVENTORY}")
        return

    df = pd.read_csv(INPUT_INVENTORY)
    total = len(df)
    log.info(f"清单总条目数: {total}")

    df_complete = df[df["files_complete"] == True].copy()
    log.info(f"files_complete=True 的条目数: {len(df_complete)}")

    if len(df_complete) == 0:
        log.error("没有 files_complete=True 的条目，请检查 step1.1 输出！")
        return

    random.seed(RANDOM_SEED)

    # ── 验证 chi.dat ───────────────────────────────────────────────────────────
    log.info(f"验证 chi.dat（抽样 {N_CHI_SAMPLE} 个）...")
    chi_errors, chi_ok = sample_and_validate(
        df_complete, N_CHI_SAMPLE, "chi.dat", validate_chi_dat, "chi.dat")

    # ── 验证 xmu.dat ───────────────────────────────────────────────────────────
    log.info(f"验证 xmu.dat（抽样 {N_XMU_SAMPLE} 个）...")
    xmu_errors, xmu_ok = sample_and_validate(
        df_complete, N_XMU_SAMPLE, "xmu.dat", validate_xmu_dat, "xmu.dat")

    # ── 验证 POSCAR ────────────────────────────────────────────────────────────
    log.info(f"验证 POSCAR_supercell_fixed（抽样 {N_POSCAR_SAMPLE} 个）...")
    poscar_errors, poscar_ok = sample_and_validate(
        df_complete, N_POSCAR_SAMPLE, "POSCAR_supercell_fixed",
        validate_poscar, "POSCAR")

    # ── 计算异常率 ─────────────────────────────────────────────────────────────
    chi_sample_n   = min(N_CHI_SAMPLE, len(df_complete))
    xmu_sample_n   = min(N_XMU_SAMPLE, len(df_complete))
    poscar_sample_n = min(N_POSCAR_SAMPLE, len(df_complete))

    chi_err_rate   = len(chi_errors)   / chi_sample_n   * 100 if chi_sample_n   > 0 else 0
    xmu_err_rate   = len(xmu_errors)   / xmu_sample_n   * 100 if xmu_sample_n   > 0 else 0
    poscar_err_rate = len(poscar_errors) / poscar_sample_n * 100 if poscar_sample_n > 0 else 0

    # ── 写入报告 ───────────────────────────────────────────────────────────────
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write("Step 1.2 文件格式验证报告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"总文件夹数（清单）:            {total}\n")
        f.write(f"files_complete=True 条目数:   {len(df_complete)}\n\n")

        f.write(f"── chi.dat 验证 ──\n")
        f.write(f"  抽样数量:     {chi_sample_n}\n")
        f.write(f"  格式正常:     {chi_ok}\n")
        f.write(f"  格式异常数:   {len(chi_errors)} ({chi_err_rate:.2f}%)\n")
        if chi_errors:
            f.write("  异常文件:\n")
            for e in chi_errors:
                f.write(f"    {e}\n")
        f.write("\n")

        f.write(f"── xmu.dat 验证 ──\n")
        f.write(f"  抽样数量:     {xmu_sample_n}\n")
        f.write(f"  格式正常:     {xmu_ok}\n")
        f.write(f"  格式异常数:   {len(xmu_errors)} ({xmu_err_rate:.2f}%)\n")
        if xmu_errors:
            f.write("  异常文件:\n")
            for e in xmu_errors:
                f.write(f"    {e}\n")
        f.write("\n")

        f.write(f"── POSCAR_supercell_fixed 验证 ──\n")
        f.write(f"  抽样数量:     {poscar_sample_n}\n")
        f.write(f"  读取成功:     {poscar_ok}\n")
        f.write(f"  读取失败数:   {len(poscar_errors)} ({poscar_err_rate:.2f}%)\n")
        if poscar_errors:
            f.write("  失败文件:\n")
            for e in poscar_errors:
                f.write(f"    {e}\n")
        f.write("\n")

        f.write("── 验收状态 ──\n")
        chi_pass   = chi_err_rate   < 2.0
        xmu_pass   = xmu_err_rate   < 2.0
        poscar_pass = poscar_err_rate < 5.0

        f.write(f"  chi.dat   异常率 < 2%:  {'✓ PASS' if chi_pass   else '✗ FAIL'}\n")
        f.write(f"  xmu.dat   异常率 < 2%:  {'✓ PASS' if xmu_pass   else '✗ FAIL'}\n")
        f.write(f"  POSCAR    失败率 < 5%:  {'✓ PASS' if poscar_pass else '✗ FAIL — 请立即停止并报告'}\n")

    log.info(f"✓ format_validation_report.txt 已写入: {OUTPUT_REPORT}")

    # ── 控制台摘要 ─────────────────────────────────────────────────────────────
    log.info("\n══ Step 1.2 执行摘要 ══")
    log.info(f"  chi.dat   异常率: {chi_err_rate:.2f}%   {'✓' if chi_err_rate < 2   else '✗ 超标！'}")
    log.info(f"  xmu.dat   异常率: {xmu_err_rate:.2f}%   {'✓' if xmu_err_rate < 2   else '✗ 超标！'}")
    log.info(f"  POSCAR    失败率: {poscar_err_rate:.2f}% {'✓' if poscar_err_rate < 5 else '✗ 超标，请停止！'}")

    if poscar_err_rate >= 5.0:
        log.error("⚠️  POSCAR 失败率 ≥ 5%！请立即停止并向用户报告！")

    log.info("Step 1.2 完成。")


if __name__ == "__main__":
    main()