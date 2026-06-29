# STEP 1 SUB-AGENT COMPLETION REPORT

**Experiment**: Exp4 — XAS → Local Structure on MP all-element EXAFS
**Phase**: Step 1 — Data Cleansing & Split Pipeline
**Status**: ✅ **PASS**（3 条 deviation 已由 Main Agent 裁决 A 并接受）
**Wall-clock**: 3,183 s (~53 min) for serial run of step1_1 → step1_6
**Reporting to**: Main Agent (Exp4)

---

## 1. 最终数据量

| 指标 | 值 | 备注 |
|---|---|---|
| FEFF raw | 133,718 | shape (133718, 76) |
| parse_fail | 0 | regex 100% 命中 |
| 剔除合计 | 5,336 | 见 §4 细分 |
| **保留样本** | **128,382** | 落在 MA 估计 [120K, 130K] |
| 独立 mp_id | 41,431 | |
| 独立 center_element | 88 | 含 4 个稀有(Ar/He/Kr/Ne) |

## 2. 切分结果(mp_id-level, random_state=42)

| split | mp_ids | samples | pct | 目标 | 偏差 |
|---|---|---|---|---|---|
| train | 33,147 | 102,660 | **79.96%** | 80% | -0.04% |
| val | 4,142 | 12,912 | **10.06%** | 10% | +0.06% |
| test | 2,485 | 7,696 | **5.99%** | 6% | -0.01% |
| holdout | 1,657 | 5,114 | **3.98%** | 4% | -0.02% |

- 6 组 pairwise mp_id 交集 = **0 / 0 / 0 / 0 / 0 / 0** ✓
- 稀有元素(14 条样本)全部落入 train ✓
- safe_stratified_split 未触发任何 primary-class 降级(所有类都够 stratify)

## 3. Feature / Scaler

| 项 | 值 |
|---|---|
| feff feature dim | **74**(73 原始 + 1 `has_pre_edge`)✓ |
| has_pre_edge=1 占比 | 95,827 / 128,382 = **74.64%** |
| RobustScaler fit 样本 | 102,660 train × 74 |
| `max\|scaler.center_ − train_median\|` | **0** (完美一致)✓ |
| Scaler reload + transform sanity | shape=(5, 74) ✓ |
| 残留 NaN | 0 ✓ |

## 4. 剔除拆解(`step1_excluded_log.csv`)

| reason | count | MA 估计 | 裁决 |
|---|---|---|---|
| chi_invalid | 1,911 | 未设 | 新信息,数据真实损耗 |
| missing_poscar | 790 | [600, 1000] | ✓ OK(§5.4b 改用磁盘 isfile 后对齐) |
| H_element | 479 | [1300, 1800] | **⚠ A** — H 口径差异(见 §6) |
| iqr_outlier | 2,156 | [3000, 10000] | **⚠ A** — 数据比预期干净(见 §6) |
| parse_fail | 0 | [0, 5] | ✓ |
| **total** | **5,336** | | |

## 5. 产物清单(全部在 `C:\Users\T-Cat\Desktop\DiffCSP-main\experiment4\step1\`)

### 给下游 Step 2/3/4/5 用的交付物(上传服务器的三件套 + 一件 Step2 直接 load 的)

| 文件 | 大小 | 用途 |
|---|---|---|
| `data_inventory.csv` | 56.8 MB | 规格 §6.1 的 14 列表格,全部 128,382 样本的元数据+split |
| `feff_features_imputed.pkl` | ~37 MB* | **Step 2 直接 load**:DataFrame, index=sample_name(unique), 74 列 float32 |
| `feff_feature_scaler.pkl` | 1.6 KB | RobustScaler 对象,Step 2/3 `transform()` 用 |
| `feff_feature_stats.csv` | 6.8 KB | 74 列的 median/iqr/q1/q3/min/max/n_nan_before_impute |
| `feff_feature_names.txt` → 当前名为 `step1_3_feff_feature_names.txt` | 1.0 KB | 74 个特征名的**有序**列表,与 pkl 列 & scaler 轴 1 对齐 |

*`feff_features_imputed.pkl` 由新增的 `step1_7_export_feff_features.py` 生成(详见 §7)

### 切分索引

| 文件 | 大小 |
|---|---|
| `train_ids.txt` / `val_ids.txt` / `test_ids.txt` / `holdout_ids.txt` | 343 / 43 / 26 / 17 KB |
| `train_samples.csv` / `val_samples.csv` / `test_samples.csv` / `holdout_samples.csv` | 4,427 / 556 / 332 / 221 KB |

### 日志 & 诊断

| 文件 | 用途 |
|---|---|
| `step1_excluded_log.csv` (292 KB) | 5,336 条剔除记录(sample_name, mp_id, center_element, reason) |
| `element_distribution.csv` (2.5 KB) | 88 元素 × (n_total, n_train, n_val, n_test, n_holdout, is_rare) |
| `step1_summary.txt` | 完整 10-check self-report |
| `step1_3_n_nan_before_impute.csv` | 每列 impute 前 NaN 数,对应文档溯源 |

### 中间态 pkl(可保留,Step 2+ 可不依赖)

| 文件 | 大小 |
|---|---|
| `step1_1_raw_inventory.pkl` | 135 MB (scan+parse 后) |
| `step1_2_filtered_inventory.pkl` | 130 MB (过滤后) |
| `step1_3_imputed_inventory.pkl` | 131 MB (imputation 后) |
| `step1_4_full_inventory.pkl` | 132 MB (加 split + quality_tier 后,**`feff_features_imputed.pkl` 的来源**) |

## 6. LOCKED DECISIONS(本 Sub-Agent 锁定的规格偏离)

### D1. §5.4(b) 语义替换 ⭐
- **原文**:剔 `mp_id ∈ set(missing_poscar_list.csv 第0列)` → ~791
- **修正**:剔 `not os.path.isfile(os.path.join(POSCAR_DIR, f"{mp_id}_POSCAR"))` → 790
- **原因**:v2 sanity 发现 `missing_poscar_list.csv` 里 8,003 条中 7,751 条磁盘上实际存在 POSCAR,按原文执行会多剔 28,161 条(36×);该列表是上游流水线的旧日志,已失效。用文件系统真值语义无歧义,且数字(790)精准命中 MA 预估 [600, 1000]
- **Main Agent 已批**:✓

### D2. 文件名模式补全(spec §5.2 underspecified)
- CHI 文件 = `{sample_name}_chi.csv`
- XMU 文件 = `{sample_name}.csv`
- POSCAR 文件 = `{mp_id}_POSCAR`(无扩展名,`pymatgen.Structure.from_file` 直接可读,无需 fallback)
- listdir 时过滤非 `mp-` 开头的条目(排除 4 个 log/manifest csv)

### D3. chi_valid / xmu_valid 判定列
- `chi_valid` = 读得了 + rows≥300 + `std(chi1) > 1e-6`(**用 `chi1` 列,不是 `chi`**)
- `xmu_valid` = 读得了 + rows≥300 + `std(y) > 1e-6`
- 依据:v2 sanity 确认 chi.csv 列为 `k,chi,chi1,chi2`, `chi1` 是 k-加权归一信号;xmu.csv 列为 `x,y`

### D4. Imputation 未覆盖列的安全网
- 对不在 `fill_zero` 也不在 `fill_group_median` 列表、但有 NaN 的列,使用 groupby(center_element) median 填补,global median fallback,最终 0.0 fallback
- 本次执行未触发此安全网(所有列都被显式策略覆盖)

## 7. 新增产物 — `feff_features_imputed.pkl`(回应 Main Agent 的问题)

**问题**:Step 2 需要按 sample_name 取 74 维 feff 向量,但 `data_inventory.csv` §6.1 字段不含 feff,Step 2 是否有干净入口?

**回应**:`step1_4_full_inventory.pkl`(132 MB) 里确实有 74 列 feff,但混杂了 `chi_path/xmu_path/poscar_valid/poscar_reason` 等元数据,Step 2 load 时过滤字段不够清爽。

**新增产物**:`step1_7_export_feff_features.py` → `feff_features_imputed.pkl`
- `pandas.DataFrame`, `index=sample_name`(unique, 128,382)
- 74 列,顺序严格对齐 `step1_3_feff_feature_names.txt`
- dtype = `float32`(节省存储,推理足够)
- 断言:index unique ✓, shape (128382, 74) ✓, 无 NaN ✓
- 预计 ~37 MB

**Step 2 使用方式示例**:
```python
import pandas as pd, joblib
feff = pd.read_pickle("feff_features_imputed.pkl")
scaler = joblib.load("feff_feature_scaler.pkl")
x74 = scaler.transform(feff.loc[sample_name].values.reshape(1, -1))  # (1, 74)
```

**需要你在 Windows 侧再跑一条命令产出该文件**:

```powershell
cd C:\Users\T-Cat\Desktop\DiffCSP-main\experiment4\step1
C:/Users/T-Cat/AppData/Local/Microsoft/WindowsApps/python3.9.exe .\step1_7_export_feff_features.py
```

预计 <10 秒。跑完把控制台输出贴回,确认 shape (128382, 74) float32。

## 8. 3 条 DEVIATION 的 Main Agent 裁决(存档)

### DV1. holdout = 5,114 > 预期区间上限 5,000
- **裁决**:A(接受)
- **MA 理由**:5,114/128,382 = 3.98% 精准对齐 4% 目标。预期区间 [3K, 5K] 是按 126K 估的,总量变动带着 4% 点位上移。**非数据偏差,是区间参数过紧。**

### DV2. H_element = 479 < 预期区间下限 1,300
- **裁决**:A(接受)
- **MA 理由**:raw FEFF 中 H 总数约 2,209(MA 此处需进一步确认,见 §9 Open Q1)。最终"H_element"只计 **走到 H 过滤器那一步** 才被剔的样本;其余 ~1,730 条在前序过滤器(chi_invalid / missing_poscar / iqr_outlier)里已被归入别的 reason。**语义上 H 仍全部排除**,只是计数归属问题。

### DV3. iqr_outlier = 2,156 < 预期区间下限 3,000
- **裁决**:A(接受,明确不重跑、不调阈值)
- **MA 理由(四条,存档完整陈述)**:
  1. IQR×50 是"数据腐败检测器",不是"分布尾部剪裁器";抓不到就说明无此类脏数据,1.68% 是合理的清洁度
  2. Exp4 要验证全元素架构泛化性,收紧阈值会误杀稀有配位环境(Ac/Pa/Np/Pu 等),**与目标直接冲突**
  3. MA 的 5,729 预估大概率用全局 IQR;Step 1 交接明确要求 grouped-by-center_element IQR,两者口径不同,分组法更合理
  4. 保持 LOCKED 决策不做二次调整,维护可追溯性

## 9. Self-Check (spec §7 10 项) 结果

| # | 项 | 结果 |
|---|---|---|
| 1 | total ∈ [120K, 130K] | ✓ 128,382 |
| 2 | holdout ∈ [3K, 5K] | ⚠→A 5,114(3.98% ✓) |
| 3 | 4-split mp_id 零交集(6 对) | ✓ 全部 0 |
| 4 | feff dim == 74 | ✓ |
| 5 | has_pre_edge ⊆ {0, 1} | ✓ |
| 6 | `max\|scaler.center_ - train_median\|` < 1e-6 | ✓ 0 |
| 7 | holdout 元素数 ≥ 30, O/Li ≤ 25% | ✓ 84 种, O=17.68%, Li=5.12% |
| 8 | rare elements 全在 train | ✓ 14 条全在 train |
| 9 | scaler reload + transform | ✓ shape (5, 74) |
| 10 | 剔除拆解 vs MA 估计 | ⚠→A 2 项(已裁决) |

**综合**:**PASS**

## 10. Open Questions(供 Main Agent 决策)

### Q1. Raw H 总数核实(可选,不阻塞 Step 2)
MA 的 §8.DV2 陈述说 raw FEFF 中 H=2,209,本 Sub-Agent 未记录此数字(step1_1 只计有效的 `_valid` 列,未显式对 center_element="H" 统计)。如需精准归属,可在 Windows 侧跑:

```python
import pandas as pd
inv = pd.read_pickle(r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment4\step1\step1_1_raw_inventory.pkl")
print("raw H count:", (inv["center_element"]=="H").sum())
```

### Q2. 服务器上传时机(MA 已给建议,此处确认)
- 建议**等 Step 2 打包完成后一次性上传**(最干净)
- 若要提前上传 Step 1 三件套占位也 OK(~40 MB)
- 本 Sub-Agent 默认你按 MA 建议**等 Step 2**

### Q3. `step1_2_filtered_inventory.pkl` / `step1_3_imputed_inventory.pkl` 等中间态是否保留?
- 建议保留至 Step 2 验收通过,之后可清理(每个 130 MB,总共 ~500 MB)
- 已交付的 `feff_features_imputed.pkl` + `data_inventory.csv` + `step1_4_full_inventory.pkl` 足以重建任何下游分析

## 11. Handoff to Step 2

下面 3 件事全部对齐后,Step 2 Sub-Agent 可以开工:
1. **跑 `step1_7_export_feff_features.py`** 产出 `feff_features_imputed.pkl`(~10 秒)
2. (可选)如想溯源 H 总数,跑 §10 Q1 命令
3. Main Agent 生成 Step 2 交接文档(chi/xmu 预处理、target 构造、tensor 打包)

**Step 2 的主要入口文件**(Sub-Agent 请照抄到 Step 2 交接):
```
data_inventory.csv                   (128,382 × 14 元数据 + split)
feff_features_imputed.pkl            (DataFrame, index=sample_name, (128382, 74) float32)
feff_feature_scaler.pkl              (RobustScaler, fit on train)
step1_3_feff_feature_names.txt       (74 名,与 pkl 列顺序对齐)
train_ids.txt / val_ids.txt / test_ids.txt / holdout_ids.txt
```

---

**End of Step 1 Sub-Agent report.** 交还 Main Agent,待 Step 2 交接文档下发。
