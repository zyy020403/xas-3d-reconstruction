"""
Step 3b — Build Element Vocabulary for TypeClassifier
======================================================
Scans the training set via XASLocalStructureDataset, collects all neighbor
atom_types (atomic numbers), and outputs:
  - experiment3/step3b/elem_vocab.json   {str(Z): class_index}
  - experiment3/step3b/elem_freq.csv     frequency table

Run from DiffCSP-main root (so xas_local_dataset_L6.py is importable):
  python experiment3/step3b/step3b_1_build_vocab.py
"""

import sys
import os
import json
import collections
import warnings

import pandas as pd

# ── 压制已知无害的 UserWarning（xmu 能量窗口超出，代码自动兜底，不影响结果）────
warnings.filterwarnings(
    "ignore",
    message="xmu.dat 能量窗口超出数据范围",
    category=UserWarning,
)

# ── 路径常量（硬编码） ─────────────────────────────────────────────────────────
DATA_ROOT     = r"C:\Users\T-Cat\Desktop\DiffCSP-main\site_dataset_Fe_only_oxide_one_site"
EXP2_ROOT     = r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment2"
EXP3_ROOT     = r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment3"
STEP1_DIR     = EXP2_ROOT + r"\step1"
FEFF_FEAT_CSV = r"C:\Users\T-Cat\Desktop\DiffCSP-main\tesst_feff_features_all_full_v4.csv"
TRAIN_IDS     = STEP1_DIR + r"\train_ids.txt"
INVENTORY_CSV = STEP1_DIR + r"\data_inventory.csv"
OUTPUT_DIR    = EXP3_ROOT + r"\step3b"

# ── 确保 xas_local_dataset_L6.py 可导入 ───────────────────────────────────────
DATASET_DIR = r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment2\step3"
if DATASET_DIR not in sys.path:
    sys.path.insert(0, DATASET_DIR)
sys.path.insert(0, r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment2\step3")
from xas_local_dataset_L6 import XASLocalStructureDataset

# 元素符号查询表（仅用于打印，不用于模型）
_Z_TO_SYM = {
    1: 'H',  2: 'He', 3: 'Li', 4: 'Be', 5: 'B',  6: 'C',  7: 'N',
    8: 'O',  9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si',
    15: 'P', 16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca', 21: 'Sc',
    22: 'Ti', 23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni',
    29: 'Cu', 30: 'Zn', 31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br',
    36: 'Kr', 37: 'Rb', 38: 'Sr', 39: 'Y', 40: 'Zr', 41: 'Nb', 42: 'Mo',
    43: 'Tc', 44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd', 49: 'In',
    50: 'Sn', 51: 'Sb', 52: 'Te', 53: 'I', 54: 'Xe', 55: 'Cs', 56: 'Ba',
    57: 'La', 58: 'Ce', 59: 'Pr', 60: 'Nd', 61: 'Pm', 62: 'Sm', 63: 'Eu',
    64: 'Gd', 65: 'Tb', 66: 'Dy', 67: 'Ho', 68: 'Er', 69: 'Tm', 70: 'Yb',
    71: 'Lu', 72: 'Hf', 73: 'Ta', 74: 'W', 75: 'Re', 76: 'Os', 77: 'Ir',
    78: 'Pt', 79: 'Au', 80: 'Hg', 81: 'Tl', 82: 'Pb', 83: 'Bi', 84: 'Po',
    85: 'At', 86: 'Rn', 87: 'Fr', 88: 'Ra', 89: 'Ac', 90: 'Th', 91: 'Pa',
    92: 'U',  93: 'Np', 94: 'Pu',
}


def z_to_sym(z: int) -> str:
    return _Z_TO_SYM.get(z, f'Z{z}')


def main():
    # ── 输出目录 ──────────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Step 1：实例化 Dataset（仅 train） ────────────────────────────────────
    print("初始化 XASLocalStructureDataset（train_ids）...")
    dataset = XASLocalStructureDataset(
        data_root        = DATA_ROOT,
        inventory_csv    = INVENTORY_CSV,
        ids_file         = TRAIN_IDS,
        feff_feat_csv    = FEFF_FEAT_CSV,
        feff_scaler_path = None,   # 词表统计不需要 scaler
    )
    print(f"Dataset 长度（train_ids 映射后）: {len(dataset)}\n")

    # ── Step 2：遍历，收集所有邻居原子序数 ───────────────────────────────────
    counter     = collections.Counter()
    n_total     = len(dataset)
    n_skipped   = 0
    LOG_INTERVAL = 500

    print(f"开始遍历 {n_total} 个样本...")
    for i in range(n_total):
        sample = dataset[i]
        if sample is None:
            n_skipped += 1
            continue

        # atom_types: torch.Tensor shape (20,), dtype=long
        zs = sample.atom_types.tolist()   # list of int
        counter.update(zs)

        if (i + 1) % LOG_INTERVAL == 0:
            top3 = [(z_to_sym(z), freq) for z, freq in counter.most_common(3)]
            top3_str = "  ".join(f"{sym}={freq}" for sym, freq in top3)
            print(f"  已处理 {i+1:>5}/{n_total}  跳过={n_skipped}  "
                  f"已知元素种数={len(counter)}  前3高频: {top3_str}",
                  flush=True)

    print(f"\n遍历完成。总计：{n_total}，跳过：{n_skipped}，"
          f"有效：{n_total - n_skipped}")

    # ── Step 3：构建词表（按频率降序，class_index 从 0 起） ──────────────────
    sorted_elements = sorted(counter.items(), key=lambda x: -x[1])
    # sorted_elements: [(z, freq), ...]

    vocab    = {}   # {str(z): class_index}
    freq_rows = []  # for CSV
    for class_idx, (z, freq) in enumerate(sorted_elements):
        vocab[str(z)] = class_idx
        freq_rows.append({
            'atomic_number': z,
            'element_symbol': z_to_sym(z),
            'frequency':      freq,
            'class_index':    class_idx,
        })

    n_elem = len(vocab)

    # ── Step 4：验证 & 打印 ───────────────────────────────────────────────────
    print()
    print("=" * 40)
    print("     元素词表统计结果")
    print("=" * 40)
    print(f"扫描样本总数：{n_total}")
    print(f"跳过（None）：{n_skipped}")
    print(f"词表大小（N_elem）：{n_elem}")

    # 关键元素检查
    o_present  = '8'  in vocab
    fe_present = '26' in vocab
    if o_present:
        print(f"O  (Z=8)  ：YES，class_index={vocab['8']}，"
              f"频率={counter[8]}")
    else:
        print("O  (Z=8)  ：MISSING")

    if fe_present:
        print(f"Fe (Z=26) ：YES，class_index={vocab['26']}，"
              f"频率={counter[26]}")
    else:
        print("Fe (Z=26) ：MISSING")

    print("前10高频元素：")
    for rank, (z, freq) in enumerate(sorted_elements[:10]):
        print(f"  rank {rank}: Z={z:3d} ({z_to_sym(z):2s}), "
              f"freq={freq:10d}, class_index={vocab[str(z)]}")

    print("=" * 40)

    # 范围警告
    if not (30 <= n_elem <= 80):
        print(f"\nWARNING: N_elem={n_elem} 超出预期范围 [30, 80]，请汇报 Main Agent")

    # 关键元素缺失报警
    if not o_present or not fe_present:
        missing = []
        if not o_present:
            missing.append('O (Z=8)')
        if not fe_present:
            missing.append('Fe (Z=26)')
        print(f"\nERROR: 关键元素缺失 {missing}，请汇报 Main Agent")

    # ── 保存文件 ──────────────────────────────────────────────────────────────
    vocab_path = os.path.join(OUTPUT_DIR, 'elem_vocab.json')
    freq_path  = os.path.join(OUTPUT_DIR, 'elem_freq.csv')

    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, indent=2)
    print(f"\n✅ 词表已保存：{vocab_path}")

    freq_df = pd.DataFrame(freq_rows)
    freq_df.to_csv(freq_path, index=False)
    print(f"✅ 频率表已保存：{freq_path}")


if __name__ == '__main__':
    main()