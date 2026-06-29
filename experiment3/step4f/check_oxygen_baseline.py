"""
check_oxygen_baseline.py
========================
不需要模型，只读 val 集标签，计算：
1. 全猜O（原子序数=8）的 Top-1 baseline 准确率
2. 各壳层的 O 占比
3. 对比 Step4f 的 val_type_acc=0.601，判断模型是否有实质提升

在 DiffCSP-main 根目录执行：
  python check_oxygen_baseline.py
"""

import os, sys, warnings
import numpy as np

DIFFCSP_ROOT = r"C:\Users\T-Cat\Desktop\DiffCSP-main"
EXP2_ROOT    = os.path.join(DIFFCSP_ROOT, "experiment2")
STEP1_DIR    = os.path.join(EXP2_ROOT, "step1")

sys.path.insert(0, os.path.join(EXP2_ROOT, "step2"))
sys.path.insert(0, os.path.join(EXP2_ROOT, "step3"))
os.environ.setdefault('PROJECT_ROOT', DIFFCSP_ROOT)

from xas_local_dataset_L6 import XASLocalStructureDataset

val_ds = XASLocalStructureDataset(
    data_root        = os.path.join(DIFFCSP_ROOT, "site_dataset_Fe_only_oxide_one_site"),
    inventory_csv    = os.path.join(STEP1_DIR, "data_inventory.csv"),
    ids_file         = os.path.join(STEP1_DIR, "val_ids.txt"),
    feff_feat_csv    = os.path.join(DIFFCSP_ROOT, "tesst_feff_features_all_full_v4.csv"),
    feff_scaler_path = os.path.join(STEP1_DIR, "feff_feature_scaler.pkl"),
)

all_types  = []   # 所有邻居原子序数
all_dists  = []   # 对应的到Fe的距离（笛卡尔，用 frac*L 估算）
L = 6.0

print("读取 val 集标签中...")
for i in range(len(val_ds)):
    sample = val_ds[i]
    if sample is None:
        continue
    types  = sample.atom_types.numpy()           # (20,)
    fcoord = sample.frac_coords.numpy()          # (20, 3)
    dists  = np.linalg.norm(fcoord * L, axis=1) # 近似笛卡尔距离
    all_types.append(types)
    all_dists.append(dists)

all_types = np.concatenate(all_types)   # (N_total,)
all_dists = np.concatenate(all_dists)

total = len(all_types)
n_oxygen = (all_types == 8).sum()
print(f"\n总邻居数：{total}")
print(f"O（Z=8）数量：{n_oxygen}  占比：{n_oxygen/total:.3f}")
print(f"\n全猜O的 Top-1 baseline 准确率：{n_oxygen/total:.4f}")
print(f"Step4f val_type_acc（epoch 26）：0.6013")
print(f"模型超出 baseline：{0.6013 - n_oxygen/total:+.4f}")

# 按壳层统计
shells = [(0, 2.5, "第一壳层 ≤2.5Å"),
          (2.5, 3.5, "第二壳层 2.5~3.5Å"),
          (3.5, 4.0, "第三壳层 3.5~4.0Å")]

print("\n各壳层 O 占比（即全猜O的壳层baseline）：")
for lo, hi, name in shells:
    mask = (all_dists >= lo) & (all_dists < hi)
    n    = mask.sum()
    n_o  = (all_types[mask] == 8).sum()
    if n > 0:
        print(f"  {name}：{n} 个原子，O 占比 {n_o/n:.3f}  ← 全猜O baseline = {n_o/n:.3f}")
    else:
        print(f"  {name}：0 个原子（无数据）")
