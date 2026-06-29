# Step 2.2 — step2_2_preprocess_validation.py
# 从 data_inventory.csv 随机抽取 20 个样本，验证三路预处理函数
# 输出：step2_visualization.png + 控制台统计信息

import sys
import os

# ── 路径设置（将 step2 目录加入 import 路径）─────────────────
STEP2_DIR  = r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment2\step2"
STEP1_DIR  = r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment2\step1"
sys.path.insert(0, STEP2_DIR)

import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from spectrum_preprocessor import load_xmu_xanes, load_chi1, load_feff_features

# ── 中文字体设置（避免警告）────────────────────────────────
def _set_chinese_font():
    candidates = [
        'Microsoft YaHei', 'SimHei', 'SimSun',
        'WenQuanYi Micro Hei', 'Noto Sans CJK SC',
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams['font.family'] = name
            return
    # 找不到中文字体就把标题改成英文，不报错
    plt.rcParams['font.family'] = 'DejaVu Sans'

_set_chinese_font()

# ── 常量 ────────────────────────────────────────────────────
INVENTORY_CSV  = os.path.join(STEP1_DIR, "data_inventory.csv")
FEAT_CSV       = r"C:\Users\T-Cat\Desktop\DiffCSP-main\tesst_feff_features_all_full_v4.csv"
SCALER_PKL     = os.path.join(STEP1_DIR, "feff_feature_scaler.pkl")
FEAT_STATS_CSV = os.path.join(STEP1_DIR, "feff_feature_stats.csv")
OUT_DIR        = STEP2_DIR
N_SAMPLES      = 20
RANDOM_SEED    = 42

# ── 加载公共资源 ─────────────────────────────────────────────
print("加载资源...")
inventory = pd.read_csv(INVENTORY_CSV)

# ── 过滤有效 train 样本（用实际存在的列）────────────────────
valid = inventory[
    (inventory['split'] == 'train') &
    (inventory['has_feff_feat'] == True) &
    (inventory['has_nan_features'] == False) &
    (inventory['flag_pre_valid'] == True) &
    (inventory['flag_white_valid'] == True) &
    (inventory['chi_npts'] > 0)
].reset_index(drop=True)
print(f"  有效 train 样本数: {len(valid)}")

# 加载 feff_features 表
feat_df = pd.read_csv(FEAT_CSV)

# 构造匹配键：mp_id + site_nn
def parse_sample_name(s):
    parts = s.split('_')
    mp_id   = parts[0] + '_' + parts[1]
    site_nn = parts[-1]
    return mp_id, site_nn

feat_df['mp_id_key']   = feat_df.iloc[:, 1].apply(lambda s: parse_sample_name(s)[0])
feat_df['site_nn_key'] = feat_df.iloc[:, 1].apply(lambda s: parse_sample_name(s)[1])

# 排序索引以消除 PerformanceWarning
feat_df_sorted = feat_df.sort_values(['mp_id_key', 'site_nn_key'])
feat_lookup = feat_df_sorted.set_index(['mp_id_key', 'site_nn_key'])

# 加载 scaler 和 NaN 填充均值
with open(SCALER_PKL, 'rb') as f:
    scaler = pickle.load(f)
feat_stats = pd.read_csv(FEAT_STATS_CSV)
col_means  = feat_stats['mean'].values.astype(np.float32)  # shape (73,)

# ── 随机抽 N_SAMPLES 个样本 ──────────────────────────────────
np.random.seed(RANDOM_SEED)
sample_idx = np.random.choice(len(valid), N_SAMPLES, replace=False)
samples    = valid.iloc[sample_idx].reset_index(drop=True)

# ── 逐样本验证 ───────────────────────────────────────────────
xmu_list   = []
chi1_list  = []
feats_list = []
labels     = []
errors     = []

for i, row in samples.iterrows():
    mp_id   = row['mp_id']
    site_nn = row['site_nn']
    folder  = row['folder_path']

    xmu_path  = os.path.join(folder, 'xmu.dat')
    chi1_path = os.path.join(folder, 'chi1.dat')

    try:
        feat_row = feat_lookup.loc[(mp_id, str(site_nn).zfill(2))]
        if isinstance(feat_row, pd.DataFrame):
            feat_row = feat_row.iloc[0]

        # ✅ 修复：直接从 inventory 取 E0，不再从 feat_row 偏移推算
        E0 = float(row['E0'])

        # 三路预处理
        xmu_out   = load_xmu_xanes(xmu_path, E0)
        chi1_out  = load_chi1(chi1_path)
        feats_out = load_feff_features(feat_row, scaler, col_means)

        # ── 断言检查 ─────────────────────────────────────────
        assert xmu_out.shape   == (150,), f"xmu shape {xmu_out.shape}"
        assert chi1_out.shape  == (200,), f"chi1 shape {chi1_out.shape}"
        assert feats_out.shape == (73,),  f"feats shape {feats_out.shape}"
        assert not np.isnan(xmu_out).any(),   "xmu NaN"
        assert not np.isnan(chi1_out).any(),  "chi1 NaN"
        assert not np.isnan(feats_out).any(), "feats NaN"
        assert np.abs(xmu_out).max()  < 20,  f"xmu 异常值: {np.abs(xmu_out).max():.2f}"
        assert np.abs(chi1_out).max() < 20,  f"chi1 异常值: {np.abs(chi1_out).max():.2f}"

        xmu_list.append(xmu_out)
        chi1_list.append(chi1_out)
        feats_list.append(feats_out)
        labels.append(f"{mp_id}\nsite{site_nn}")
        print(f"  [{i+1:02d}] OK {mp_id} site{site_nn}  E0={E0:.1f} eV")

    except Exception as e:
        errors.append((mp_id, site_nn, str(e)))
        print(f"  [{i+1:02d}] FAIL {mp_id} site{site_nn} -- {e}")

# ── 统计摘要 ─────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"验证结果：{len(xmu_list)}/{N_SAMPLES} 样本通过，{len(errors)} 个失败")
if errors:
    for mp_id, site_nn, msg in errors:
        print(f"  FAIL  {mp_id} site{site_nn}: {msg}")

if xmu_list:
    xmu_arr   = np.stack(xmu_list)
    chi1_arr  = np.stack(chi1_list)
    feats_arr = np.stack(feats_list)

    xmu_mean_of_stds = xmu_arr.std(axis=1).mean()
    chi1_mean_of_stds = chi1_arr.std(axis=1).mean()
    feats_mean_of_stds = feats_arr.std(axis=1).mean()

    print(f"\n{'-'*55}")
    print(f"xmu_xanes : mean={xmu_arr.mean(1).mean():.4f}  std={xmu_mean_of_stds:.4f}"
          f"  {'OK' if 0.7 < xmu_mean_of_stds < 1.3 else 'WARNING: std 异常，期望接近 1.0'}")
    print(f"chi1      : mean={chi1_arr.mean(1).mean():.4f}  std={chi1_mean_of_stds:.4f}"
          f"  (max-abs 归一化，std 无需为 1)")
    print(f"feats     : mean={feats_arr.mean(1).mean():.4f}  std={feats_mean_of_stds:.4f}"
          f"  {'OK' if 0.7 < feats_mean_of_stds < 1.3 else 'WARNING: std 异常'}")
    print(f"{'-'*55}")

# ── 可视化 ───────────────────────────────────────────────────
n_ok = len(xmu_list)
if n_ok > 0:
    fig = plt.figure(figsize=(22, 14))
    outer = fig.add_gridspec(4, 5, hspace=0.7, wspace=0.4)

    for i in range(min(n_ok, 20)):
        row_g = i // 5
        col_g = i % 5
        inner = outer[row_g, col_g].subgridspec(2, 1, hspace=0.05)

        ax1 = fig.add_subplot(inner[0])
        ax2 = fig.add_subplot(inner[1])

        ax1.plot(xmu_list[i],  color='steelblue',  lw=0.8)
        ax2.plot(chi1_list[i], color='darkorange', lw=0.8)

        ax1.set_title(labels[i], fontsize=6.5)
        ax1.set_ylabel("XANES", fontsize=5)
        ax2.set_ylabel("chi1",  fontsize=5)
        for ax in (ax1, ax2):
            ax.tick_params(labelsize=5)
            ax.set_xticks([])

    fig.suptitle("Step 2.2 Validation — top: xmu XANES (150pt)  bottom: chi1 k1chi(k) (200pt)",
                 fontsize=11)

    out_path = os.path.join(OUT_DIR, "step2_visualization.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n可视化已保存：{out_path}")

print("\nStep 2.2 完成。")