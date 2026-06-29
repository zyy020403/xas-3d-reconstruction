# Step 2.1 — spectrum_preprocessor.py
# 三路谱预处理函数库：xmu_xanes / chi1 / feff_features
# ★ 纯库文件，不含执行代码，Step3 直接 import
#
# xmu.dat 列顺序（Step1 实测确认）：
#   col0=omega(绝对能量eV) / col1=e(相对能量) / col2=k / col3=mu / col4=mu0 / col5=chi
#   ✅ 能量列 = data[:,0]（绝对能量，与 E0 单位一致）
#   ✅ μ(E)列 = data[:,3]

import os
import logging
import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# 函数 1：XANES 窗口截取 + 插值 + z-score
# ──────────────────────────────────────────────────────────────

def load_xmu_xanes(xmu_path, E0, n_points=150, pre_eV=50, post_eV=150):
    """
    从 xmu.dat 截取 XANES 窗口 [E0-50, E0+150] eV，插值到 150 点，逐样本 z-score。

    xmu.dat 列顺序（实测）：
        col0 = omega  绝对能量 (eV)  ← 用这列
        col1 = e      相对能量 (E-E0)
        col2 = k
        col3 = mu     吸收系数       ← 用这列
        col4 = mu0
        col5 = chi

    Parameters
    ----------
    xmu_path : str
    E0 : float   吸收边绝对能量（来自 inventory['E0']，单位 eV）
    n_points : int
    pre_eV / post_eV : float

    Returns
    -------
    np.ndarray, shape (150,), dtype float32
    """
    data = np.loadtxt(xmu_path, comments='#')

    # ✅ 修复：使用绝对能量列 col0，与 E0 单位一致
    E  = data[:, 0]   # 绝对能量
    mu = data[:, 3]   # μ(E)

    E_lo, E_hi = E0 - pre_eV, E0 + post_eV
    mask = (E >= E_lo) & (E <= E_hi)

    if mask.sum() < 5:
        # 容错：窗口内点数不足，退化为全局范围插值
        E_win, mu_win = E, mu
    else:
        E_win, mu_win = E[mask], mu[mask]

    E_uniform = np.linspace(E_lo, E_hi, n_points)
    mu_interp = np.interp(E_uniform, E_win, mu_win,
                          left=mu_win[0], right=mu_win[-1])

    # 逐样本 z-score（防除零）
    mu_norm = (mu_interp - mu_interp.mean()) / (mu_interp.std() + 1e-8)
    return mu_norm.astype(np.float32)


# ──────────────────────────────────────────────────────────────
# 函数 2：chi1(k) 插值 + max-abs 归一化
# ──────────────────────────────────────────────────────────────

def load_chi1(chi1_path, n_points=200, min_valid_points=30):
    """
    读取 chi1.dat（已是 k¹χ(k)），插值到 200 点，max-abs 归一化。
    """
    zero_out = np.zeros(n_points, dtype=np.float32)

    if not os.path.isfile(chi1_path):
        logger.warning(f"load_chi1: 文件不存在: {chi1_path}")
        return zero_out

    rows = []
    try:
        with open(chi1_path, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                try:
                    rows.append([float(p) for p in parts])
                except ValueError:
                    continue
    except Exception as e:
        logger.warning(f"load_chi1: 读取失败 {chi1_path}: {e}")
        return zero_out

    if len(rows) == 0:
        logger.warning(f"load_chi1: 空文件或全注释: {chi1_path}")
        return zero_out

    rows = np.array(rows, dtype=np.float32)

    if rows.ndim == 1 or rows.shape[1] == 1:
        chi_raw = rows.ravel()
        k_raw   = np.linspace(0.0, 12.0, len(chi_raw), dtype=np.float32)
    else:
        k_raw   = rows[:, 0]
        chi_raw = rows[:, 1]

    mask    = k_raw >= 0.0
    k_raw   = k_raw[mask]
    chi_raw = chi_raw[mask]

    if len(k_raw) < min_valid_points:
        logger.warning(
            f"load_chi1: 有效点数不足 {min_valid_points}，"
            f"实际 {len(k_raw)} 点: {chi1_path}"
        )
        return zero_out

    k_uniform  = np.linspace(k_raw.min(), k_raw.max(), n_points, dtype=np.float32)
    chi_interp = np.interp(k_uniform, k_raw, chi_raw,
                           left=chi_raw[0], right=chi_raw[-1]).astype(np.float32)

    max_abs = np.max(np.abs(chi_interp))
    if max_abs > 1e-10:
        chi_interp = chi_interp / max_abs

    return chi_interp


# ──────────────────────────────────────────────────────────────
# 函数 3：feff_features NaN 填充 + StandardScaler 标准化
# ──────────────────────────────────────────────────────────────

def load_feff_features(features_row, scaler, col_means_for_nan):
    """
    从 feff_features 表的一行提取 73 个数值列，NaN 填充 + 标准化。
    """
    vals = features_row.iloc[3:76].values.astype(np.float32)

    nan_mask = np.isnan(vals)
    if nan_mask.any():
        vals[nan_mask] = col_means_for_nan[nan_mask]

    vals_scaled = scaler.transform(vals.reshape(1, -1)).flatten()
    return vals_scaled.astype(np.float32)