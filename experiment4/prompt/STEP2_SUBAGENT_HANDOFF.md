# STEP 2 SUB-AGENT 交接文档
# Experiment 4 — 谱预处理：xmu(XANES) + chi1(EXAFS) → 定长张量

> **发送对象**：DiffCSP-Exp4-Step2-SubAgent（新会话窗口）
> **撰写者**：DiffCSP-Exp4-Main-Agent 2
> **日期**：2026-04-23
> **执行环境**：本地 Windows（Python 3.9，无需 SSH）
> **前置依赖**：Step 1 已 PASS（输出全部在 `experiment4\step1\`）

---

## 1. 你的角色

你是 Experiment 4 的 **Step 2 Sub-Agent**。上游 Step 1 已完成数据清洗、切分和 RobustScaler 拟合，你的工作是**把每个样本的原始 xmu.csv 和 chi.csv 转成固定长度的张量数组**，供 Step 3 的 Dataset 直接 load。

**你要做的事**：
1. 读每个保留样本的 `xmu.csv`（能量吸收谱）和 `chi.csv`（EXAFS 振荡）
2. xmu 截取 `[E0-50, E0+150]` eV 窗口，插值到 **150 点**
3. chi 的 **chi1 列**截取 `k ∈ [0, 12]` Å⁻¹，插值到 **200 点**
4. 按 split（train/val/test/holdout）打包成 4 个 pkl 文件
5. 视觉抽检 + 统计自查

**你不做的事**：
- ❌ 不处理 POSCAR（Step 3 Dataset 的事）
- ❌ 不动 feff_features（Step 1 已搞定，存在 `feff_features_imputed.pkl`，Step 3 直接用）
- ❌ 不做任何 scaling / normalization（xmu 和 chi 原始尺度直接保留；feff 的 scaler 也不在你这里 apply，由 Step 3 Dataset 在 `__getitem__` 时 transform）
- ❌ 不构造目标 tensor（目标原子位置/类型由 Step 3 Dataset 从 POSCAR 现场构造）

**你不改决策**：本文档标 🔒 的是已锁死的规格，不要自由发挥。发现问题停下来汇报。

**你在本地 Windows 跑**，不需要 SSH。

---

## 2. 动手前：必须先做的三件事

### 事一：确认 Step 1 产出齐全

在 `C:\Users\T-Cat\Desktop\DiffCSP-main\experiment4\step1\` 检查以下 9 个文件存在且可 load：

```python
import pandas as pd, joblib
inv   = pd.read_csv(r"...\step1\data_inventory.csv")          # 128382 × 14
feff  = pd.read_pickle(r"...\step1\feff_features_imputed.pkl") # (128382, 74)
scal  = joblib.load(r"...\step1\feff_feature_scaler.pkl")      # RobustScaler
# train_ids.txt, val_ids.txt, test_ids.txt, holdout_ids.txt
# train_samples.csv, val_samples.csv, test_samples.csv, holdout_samples.csv
```

assert：
- `inv.shape[0] == 128382`
- `feff.shape == (128382, 74)` 且 `feff.index.is_unique`
- `inv['split'].value_counts()` = {train: 102660, val: 12912, test: 7696, holdout: 5114}
- 4 个 samples.csv 加起来 = 128,382，无交集

### 事二：读 3 条 xmu / chi 文件头部验证格式

```python
# 从 data_inventory 随机抽 3 个样本，打印：
# 1) xmu_path 指向的 CSV 前 5 行（应见表头 x,y + 4 行数值）
# 2) chi_path 指向的 CSV 前 5 行（应见表头 k,chi,chi1,chi2 + 4 行数值）
# 3) 行数是否 = 401（1 表头 + 400 数据）
# 4) 是否为逗号分隔
# 5) x 列（xmu）和 k 列（chi）是否单调递增
```

如果任一项不符，**停下来汇报**，不要猜测。

### 事三：读 5 条样本的 E0 值做 sanity check

```python
# 随机抽 5 个样本，从 feff_features_imputed.pkl 取 'E0' 列
# 打印 sample_name, center_element, E0
# 验证：E0 大致在该元素 K-edge 能量附近
#   H:   13.6 eV       C:   284 eV
#   O:   543 eV        Fe: 7112 eV
#   Cu:  8979 eV       U:  115606 eV
# 没有在 [10 eV, 130000 eV] 范围内的视为异常，汇报
```

三事都过后才开始写主脚本。

---

## 3. Step 1 交付物（你的唯一输入源）

所有路径在 `STEP1_DIR = C:\Users\T-Cat\Desktop\DiffCSP-main\experiment4\step1\`：

| 文件 | 你怎么用 |
|------|---------|
| `data_inventory.csv` | 主索引，读 `sample_name, mp_id, center_element, chi_path, xmu_path, split` 这 6 列 |
| `feff_features_imputed.pkl` | 只读 `E0` 列来定位 xmu 窗口 |
| `train_samples.csv` / `val_samples.csv` / `test_samples.csv` / `holdout_samples.csv` | 切分索引 |

其他 Step 1 产物（scaler、feff_names、ids.txt 等）**Step 2 用不到**，下游 Step 3 才用。

---

## 4. 路径常量（脚本顶部粘贴）

```python
import os

EXP4_ROOT   = r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment4"
STEP1_DIR   = os.path.join(EXP4_ROOT, "step1")
STEP2_DIR   = os.path.join(EXP4_ROOT, "step2")
os.makedirs(STEP2_DIR, exist_ok=True)

INVENTORY_CSV = os.path.join(STEP1_DIR, "data_inventory.csv")
FEFF_PKL      = os.path.join(STEP1_DIR, "feff_features_imputed.pkl")
```

建议拆成 2 个脚本（或合一）：
- `step2_1_preprocess_spectra.py` — 主预处理 + 打包
- `step2_2_visualize_samples.py` — 抽 5 个样本画 pre/post 插值图做 QC

---

## 5. 核心逻辑（🔒 LOCKED）

### 5.1 XMU 预处理（150 点）🔒

**输入**：`xmu.csv`，表头 `x,y`，401 行（1 表头 + 400 数据点）
- `x` 列 = 能量 E（eV）
- `y` 列 = 归一化吸收强度 μ(E)

**每个样本的 E0**：从 `feff_features_imputed.pkl.loc[sample_name, "E0"]` 读（eV 单位）

**处理流程（伪码）**：
```python
raw = pd.read_csv(xmu_path)                        # (400, 2)
E   = raw["x"].values                              # native energy grid
mu  = raw["y"].values                              # native intensity
E0  = feff.loc[sample_name, "E0"]                  # scalar

# 目标窗口和网格
E_target = np.linspace(E0 - 50.0, E0 + 150.0, 150) # (150,)

# 排序防御（FEFF 输出理论上已递增，但做一层保险）
order = np.argsort(E)
E, mu = E[order], mu[order]

# 线性插值
xmu_150 = np.interp(E_target, E, mu)               # (150,) float32
# np.interp 默认用 left=mu[0], right=mu[-1] 做常数外推 —— 我们要它
```

**边界情况记录**（不要跳过，要计数）：
- 如果 `E[0] > E0 - 50`：xmu 窗口左边界被外推 → `xmu_left_extrapolated += 1`
- 如果 `E[-1] < E0 + 150`：右边界被外推 → `xmu_right_extrapolated += 1`
- 两者都触发 → 两个计数器都 +1

外推本身不是 bug（常数外推对 XAS 谱远离边缘的部分影响小），但 `xmu_right_extrapolated > 5000` 要汇报。

### 5.2 CHI 预处理（200 点）🔒

**输入**：`chi.csv`，表头 `k,chi,chi1,chi2`，401 行（1 表头 + 400 数据点）

**⚠️ 关键**：**用 `chi1` 列**（第 3 列，索引 2）—— 这是 k¹χ(k) 加权信号，对应 Exp2 的输入。**不要用 `chi` 或 `chi2` 列**。

**处理流程（伪码）**：
```python
raw  = pd.read_csv(chi_path)                       # (400, 4)
k    = raw["k"].values                             # native k grid (Å⁻¹)
chi1 = raw["chi1"].values                          # native k¹χ(k) signal

# 目标网格（固定，与 E0 无关）
k_target = np.linspace(0.0, 12.0, 200)             # (200,)

# 排序 + 去负 k 防御（FEFF 输出可能包含 k<0 的数值伪影，强制裁掉）
mask   = k >= 0.0
k, chi1 = k[mask], chi1[mask]
order  = np.argsort(k)
k, chi1 = k[order], chi1[order]

chi_200 = np.interp(k_target, k, chi1).astype(np.float32)  # (200,)
```

**边界情况记录**：
- 如果 `k[-1] < 12.0`：k 空间右边界被外推 → `chi_right_extrapolated += 1`
- `chi_right_extrapolated > 5000` 要汇报

### 5.3 打包输出（🔒）

**不对 xmu 和 chi 做 normalization、standardization、scaling** —— 保持原始物理尺度。

**每个 split 产出一个 pkl 文件**，结构统一：

```python
# 例：spectra_train.pkl 内容（dict）
{
    "sample_names":  ["mp-10003__mp-10003-EXAFS-Co-K", ...],   # list[str], len = N_split
    "xmu":           np.ndarray(shape=(N_split, 150), dtype=float32),
    "chi1":          np.ndarray(shape=(N_split, 200), dtype=float32),
    "name_to_idx":   {"mp-10003__mp-10003-EXAFS-Co-K": 0, ...}, # dict[str, int], 快速索引
    "E0":            np.ndarray(shape=(N_split,), dtype=float32), # 每个样本的 E0，便于 Step 3 debug
    "meta": {
        "xmu_window_eV": [-50.0, 150.0],        # 相对 E0
        "xmu_n_points": 150,
        "chi_k_range": [0.0, 12.0],
        "chi_n_points": 200,
        "chi_column":   "chi1",
        "dtype":        "float32",
        "interp":       "np.interp (linear, constant extrapolation)",
    }
}
```

用 `pickle.dump(obj, f, protocol=4)` 保存。

**split 到文件名**：
- `spectra_train.pkl`
- `spectra_val.pkl`
- `spectra_test.pkl`
- `spectra_holdout.pkl`

### 5.4 执行规模控制 🔒

- 总样本数 128,382，串行 CSV 读 + 插值单线程 ≈ 25-40 分钟。可接受。
- **不要用多进程 / num_workers > 0**（Windows I/O 不稳定，和全流程保持一致）。
- 建议用 `tqdm` 打进度条，每 10,000 样本 print 一次内存占用（`psutil.Process().memory_info().rss / 1e9` GB）供监控。
- 处理顺序按 `data_inventory.csv` 行序，每行根据 `split` 字段写入对应 bucket，最后一次性 dump 四个 pkl。

### 5.5 dtype 规格 🔒

- xmu、chi1 数组：`np.float32`（不要用 float64，省一半空间，精度足够）
- E0 数组：`np.float32`
- sample_names：`list[str]`（不要 np.array(str)，pickle 反序列化后 dtype 奇怪）

---

## 6. 输出文件清单（严格按此）

全部放在 `STEP2_DIR = C:\Users\T-Cat\Desktop\DiffCSP-main\experiment4\step2\`：

| 文件 | 预估大小 | 内容 |
|------|---------|------|
| `spectra_train.pkl` | ~150 MB | 102,660 样本的 xmu+chi1+E0 |
| `spectra_val.pkl` | ~20 MB | 12,912 样本 |
| `spectra_test.pkl` | ~12 MB | 7,696 样本 |
| `spectra_holdout.pkl` | ~8 MB | 5,114 样本 |
| `step2_spectra_stats.csv` | <10 KB | 每 split 的 xmu 和 chi1 的 mean/std/min/max，供核对 |
| `step2_extrapolation_log.csv` | <5 KB | 每 split 的外推计数（左/右，xmu/chi） |
| `step2_qc_samples.png` | ~500 KB | 5 个随机样本的 pre/post 插值对比图（2×5 subplots：顶排 xmu，底排 chi1） |
| `step2_summary.txt` | <5 KB | 人类可读汇总 |

**绝对不要产出的东西**：
- ❌ 不要产出 scaled/normalized 的 xmu/chi（Step 3 用 raw 尺度）
- ❌ 不要把 feff 拼进 spectra pkl（Step 3 Dataset 自己从 `feff_features_imputed.pkl` 读）
- ❌ 不要保存 per-sample 小文件（128K 小文件 Windows NTFS 性能灾难）

---

## 7. 自查清单（汇报前必须全跑过）

打印到 console 和 `step2_summary.txt`：

1. **四个 pkl 样本数**：
   - train 应 = 102,660
   - val 应 = 12,912
   - test 应 = 7,696
   - holdout 应 = 5,114

2. **四个 pkl 加起来** = 128,382，与 `data_inventory.csv` 对齐

3. **shape 验证**：每个 split 的 `xmu.shape[1] == 150`，`chi1.shape[1] == 200`

4. **dtype 验证**：`xmu.dtype == np.float32`，`chi1.dtype == np.float32`

5. **无 NaN / Inf**：
   ```python
   assert np.isfinite(xmu).all(), "xmu has NaN/Inf"
   assert np.isfinite(chi1).all(), "chi1 has NaN/Inf"
   ```

6. **sample_names 与 name_to_idx 一致**：
   ```python
   assert all(name_to_idx[n] == i for i, n in enumerate(sample_names))
   ```

7. **跨 split 零重复**：所有 split 的 sample_names 合集应等于 128,382（无重复、无缺失）

8. **统计 sanity**（输出到 `step2_spectra_stats.csv`）：
   | split | xmu_mean | xmu_std | xmu_min | xmu_max | chi1_mean | chi1_std | chi1_min | chi1_max |
   - train vs val vs test vs holdout 的均值和方差应**高度接近**（相对差 < 20%）。若 holdout 的 xmu_std 比 train 大 2 倍以上，汇报（可能提示切分不均）。

9. **外推计数**（输出到 `step2_extrapolation_log.csv`）：
   - 总 `xmu_right_extrapolated`、`xmu_left_extrapolated`、`chi_right_extrapolated`
   - 若 `xmu_right_extrapolated > 5000` 或 `chi_right_extrapolated > 5000` 汇报

10. **视觉 QC**（`step2_qc_samples.png`）：
    - 随机抽 5 个样本（跨不同中心元素，如 O、Fe、Cu、La、U 各一个）
    - 画 2 行 × 5 列 subplot：
      - 上行：xmu 的 native (E, μ) 与 resampled (E_target, xmu_150) 叠加
      - 下行：chi 的 native (k, chi1) 与 resampled (k_target, chi_200) 叠加
    - 每个子图标题显示 `sample_name`、`center_element`、`E0`
    - **自己肉眼看**：resampled 曲线在窗口内应贴合 native，窗口外若外推应是常数平台
    - 若发现曲线爆炸、错位、非物理震荡，**停下来汇报不要继续**

---

## 8. 汇报模板

```markdown
## Step 2 完成报告

### 8.1 执行总览
- Wall-clock：? 分钟
- 处理样本数：? / 128,382（预期全部成功）
- 失败样本数：?（逐条列出 sample_name 和原因）

### 8.2 各 split 样本数与文件大小
| split | 样本数 | 预期 | spectra_{split}.pkl 大小 |
|-------|-------|-----|------------------------|
| train | ? | 102,660 | ? MB |
| val | ? | 12,912 | ? MB |
| test | ? | 7,696 | ? MB |
| holdout | ? | 5,114 | ? MB |

### 8.3 shape & dtype
| split | xmu shape | xmu dtype | chi1 shape | chi1 dtype |
|-------|-----------|-----------|------------|------------|
| ... | ... | ... | ... | ... |

### 8.4 统计 sanity
[粘贴 step2_spectra_stats.csv]

观察：train/val/test/holdout 的 mean/std 是否高度接近？（相对差 < 20% 为健康）

### 8.5 外推计数
| split | xmu_left_extrap | xmu_right_extrap | chi_right_extrap |
|-------|-----------------|------------------|------------------|
| train | ? | ? | ? |
| val | ? | ? | ? |
| test | ? | ? | ? |
| holdout | ? | ? | ? |
| 合计 | ? | ? | ? |

### 8.6 视觉 QC
- 5 个样本已画到 step2_qc_samples.png
- 肉眼检查：[OK / 发现异常：...]
- 采样的 5 个元素：[O, Fe, Cu, La, U]

### 8.7 产出文件列表与大小
[ls -lh 风格列出]

### 8.8 异常与发现
[任何偏离预期的情况；若无写"无"]

### 8.9 需要 Main Agent 决策的问题
[若无写"无"]
```

---

## 9. 不要做的事

1. ❌ 不要用 `chi` 列或 `chi2` 列 —— 只用 **chi1**
2. ❌ 不要对 xmu 或 chi 做 normalize / standardize / RobustScaler —— 保持原始尺度
3. ❌ 不要触碰 `feff_features_imputed.pkl` 的内容（只读 `E0` 列定位 xmu 窗口）
4. ❌ 不要处理 POSCAR
5. ❌ 不要构造目标 tensor（原子位置/类型是 Step 3 的事）
6. ❌ 不要用多进程 / num_workers > 0
7. ❌ 不要改 150 / 200 / [E0-50, E0+150] / [0, 12] 这些数字（🔒 LOCKED）
8. ❌ 不要生成 per-sample 的小文件（128K 个 NTFS 小文件会极慢）
9. ❌ 不要把 holdout 和其他 split 合并成一个大文件（holdout 必须独立存储，训练期禁止访问）
10. ❌ 不要修改 Step 1 的任何产出

---

## 10. 依赖

```
pandas
numpy
scipy       (可选，如果用 interp1d；np.interp 足够则无需)
matplotlib  (仅 QC 画图用)
tqdm
psutil      (可选，监控内存)
joblib      (仅读 Step 1 scaler 验证用)
```

建议复用 Step 1 的 `exp4_step1` conda env（加装 matplotlib 即可）：

```powershell
conda activate exp4_step1
pip install matplotlib psutil
```

---

## 11. 交付节奏

跑完后：
1. 按 §8 模板写汇报，贴给用户
2. 用户把汇报转发给 Main Agent
3. Main Agent 审查通过后下发 Step 3 交接文档（Dataset + SpectrumEncoder + 前向测试，在服务器上执行）

Step 3 Sub-Agent 会 load 你产出的 4 个 `spectra_*.pkl`，所以文件结构和 key 的稳定性很关键 —— 按本文档 §5.3 规格严格执行，**不要自创字段名**。

---

*DiffCSP-Exp4-Main-Agent 2 撰写，2026-04-23*
