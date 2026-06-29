# Step 2.5 — step2_5_determine_L.py
# 统计第20近邻距离分布，确定虚拟晶格边长 L
# 输出：d20_distribution.png + L_recommendation.txt
#
# L 确定原则：
#   L > 2 × d_max_99th_pct  →  保证所有邻居分数坐标落在 [-0.5, 0.5]
#   向上取整到最近整数

import sys
import os
STEP2_DIR = r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment2\step2"
STEP1_DIR = r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment2\step1"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import warnings
warnings.filterwarnings('ignore')

from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# ── 常量 ────────────────────────────────────────────────────
INVENTORY_CSV = os.path.join(STEP1_DIR, "data_inventory.csv")
OUT_DIR       = STEP2_DIR
N_SAMPLE      = 500
RANDOM_SEED   = 42
N_NEIGHBORS   = 20
SEARCH_RADIUS = 10.0   # Å，覆盖 10Å 内所有邻居

os.makedirs(OUT_DIR, exist_ok=True)

# ── 原胞转换函数（复用 Exp1 逻辑）──────────────────────────
def get_primitive_structure(poscar_path, symprec=0.1):
    structure = Structure.from_file(poscar_path)
    analyzer  = SpacegroupAnalyzer(structure, symprec=symprec)
    return analyzer.get_primitive_standard_structure()

# ── 主循环 ───────────────────────────────────────────────────
inventory = pd.read_csv(INVENTORY_CSV)
valid = inventory[
    (inventory['prim_n_atoms'] > 0) &        # poscar 可解析且原胞转换成功
    (inventory['flag_pre_valid'] == True) &   # XANES 有效
    (inventory['chi_npts'] > 0)               # chi 数据存在
].reset_index(drop=True)
print(f"有效样本数: {len(valid)}")

np.random.seed(RANDOM_SEED)
sample_idx = np.random.choice(len(valid), min(N_SAMPLE, len(valid)), replace=False)
samples    = valid.iloc[sample_idx].reset_index(drop=True)

d20_list   = []   # 第20近邻的距离
d_max_list = []   # 20个邻居中最远的距离（= d20，因为按距离排序取前20）
skipped    = []   # 邻居不足 20 个的样本

print(f"开始处理 {len(samples)} 个样本...")

for i, row in samples.iterrows():
    poscar_path = os.path.join(row['folder_path'], 'POSCAR_supercell_fixed')
    mp_id       = row['mp_id']

    try:
        primitive = get_primitive_structure(poscar_path)

        # 定位第一个 Fe 位点
        fe_indices = [idx for idx, s in enumerate(primitive)
                      if s.specie.symbol == 'Fe']
        if not fe_indices:
            skipped.append((mp_id, "no Fe in primitive"))
            continue
        fe_idx = fe_indices[0]

        # 获取 10Å 内所有邻居，按距离排序
        neighbors = primitive.get_neighbors(primitive[fe_idx], r=SEARCH_RADIUS)
        neighbors.sort(key=lambda n: n.nn_distance)

        if len(neighbors) < N_NEIGHBORS:
            skipped.append((mp_id, f"only {len(neighbors)} neighbors in {SEARCH_RADIUS}Å"))
            continue

        top20  = neighbors[:N_NEIGHBORS]
        d20    = top20[-1].nn_distance          # 第20个邻居距离
        d_max  = d20                            # 20个邻居中最远 = 第20个

        d20_list.append(d20)
        d_max_list.append(d_max)

        if (i + 1) % 50 == 0:
            print(f"  [{i+1:3d}/{len(samples)}] 已处理 {len(d20_list)} 个有效样本")

    except Exception as e:
        skipped.append((mp_id, str(e)))

d20_arr  = np.array(d20_list)
dmax_arr = np.array(d_max_list)

# ── 统计量 ───────────────────────────────────────────────────
p_vals = [50, 90, 95, 99, 99.9]

print(f"\n{'='*55}")
print(f"有效样本数：{len(d20_arr)} / {len(samples)}")
print(f"跳过样本数：{len(skipped)}")
if skipped:
    for mp_id, reason in skipped[:10]:
        print(f"  SKIP {mp_id}: {reason}")
    if len(skipped) > 10:
        print(f"  ... 及另外 {len(skipped)-10} 个")

print(f"\n{'─'*55}")
print(f"{'分位数':>10}  {'d20 (Å)':>12}  {'d_max (Å)':>12}")
print(f"{'─'*10}  {'─'*12}  {'─'*12}")
for p in p_vals:
    d20_p  = np.percentile(d20_arr, p)
    dmax_p = np.percentile(dmax_arr, p)
    print(f"  {p:6.1f}th   {d20_p:12.4f}  {dmax_p:12.4f}")

d20_mean   = d20_arr.mean()
d20_median = np.median(d20_arr)
d20_99     = np.percentile(d20_arr, 99)
dmax_99    = np.percentile(dmax_arr, 99)

print(f"\n  mean  (d20) = {d20_mean:.4f} Å")
print(f"  median(d20) = {d20_median:.4f} Å")
print(f"  99th  (d20) = {d20_99:.4f} Å")
print(f"  99th (dmax) = {dmax_99:.4f} Å")

# ── 推荐 L ───────────────────────────────────────────────────
L_raw  = 2.0 * dmax_99
L_rec  = math.ceil(L_raw)          # 向上取整到整数
L_safe = L_rec + 1 if L_rec < 12 else L_rec  # 保底 >= 12Å

print(f"\n{'='*55}")
print(f"推荐虚拟晶格边长 L：")
print(f"  L_raw  = 2 × d_max_99th = 2 × {dmax_99:.4f} = {L_raw:.4f} Å")
print(f"  L_rec  = ceil(L_raw)    = {L_rec} Å")
print(f"  L_final= {L_safe} Å  ← Step3 使用此值")
print(f"{'='*55}")

# ── 可视化 ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(d20_arr, bins=60, color='steelblue', edgecolor='white', lw=0.3)
axes[0].axvline(d20_99,  color='red',    lw=1.5, ls='--', label=f'99th={d20_99:.2f}Å')
axes[0].axvline(d20_mean, color='orange', lw=1.2, ls=':',  label=f'mean={d20_mean:.2f}Å')
axes[0].set_xlabel("d₂₀ — 第20近邻距离 (Å)", fontsize=11)
axes[0].set_ylabel("样本数",                  fontsize=11)
axes[0].set_title("第20近邻距离分布",          fontsize=12)
axes[0].legend(fontsize=9)

# CDF
x_sorted = np.sort(d20_arr)
cdf      = np.arange(1, len(x_sorted)+1) / len(x_sorted)
axes[1].plot(x_sorted, cdf, color='steelblue', lw=1.5)
axes[1].axvline(d20_99, color='red', lw=1.5, ls='--', label=f'99th={d20_99:.2f}Å')
axes[1].axhline(0.99,   color='red', lw=0.8, ls=':')
axes[1].set_xlabel("d₂₀ (Å)",     fontsize=11)
axes[1].set_ylabel("累积概率 CDF", fontsize=11)
axes[1].set_title("d₂₀ CDF",      fontsize=12)
axes[1].legend(fontsize=9)

fig.suptitle(f"Step 2.5 虚拟晶格边长分析  |  推荐 L = {L_safe} Å", fontsize=13)
plt.tight_layout()

png_path = os.path.join(OUT_DIR, "d20_distribution.png")
plt.savefig(png_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n可视化已保存：{png_path}")

# ── 输出 L_recommendation.txt ────────────────────────────────
txt_path = os.path.join(OUT_DIR, "L_recommendation.txt")
with open(txt_path, 'w', encoding='utf-8') as f:
    f.write("# Step 2.5 虚拟晶格边长推荐\n")
    f.write(f"# 生成时间：{pd.Timestamp.now()}\n\n")
    f.write(f"L = {L_safe}  # 单位 Å，Step3 Dataset 使用此值\n\n")
    f.write(f"# 统计依据\n")
    f.write(f"n_valid      = {len(d20_arr)}\n")
    f.write(f"n_skipped    = {len(skipped)}\n")
    f.write(f"d20_mean     = {d20_mean:.4f}\n")
    f.write(f"d20_median   = {d20_median:.4f}\n")
    f.write(f"d20_99th     = {d20_99:.4f}\n")
    f.write(f"dmax_99th    = {dmax_99:.4f}\n")
    f.write(f"L_raw        = {L_raw:.4f}  # 2 x dmax_99th\n")
    f.write(f"L_rec        = {L_rec}       # ceil(L_raw)\n")
    f.write(f"L_final      = {L_safe}      # Step3 使用值\n")
print(f"L 推荐已写入：{txt_path}")
print("\nStep 2.5 完成。")