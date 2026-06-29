# SHARED_04_PHYSICAL_CONSTRAINTS.md
# 物理约束与先验知识 — 所有 Sub-Agent 必读

> **本文档版本**: v1.0（已完成）  
> **维护者**: Main Agent

---

## 1. 键长约束表（all_center_neighbors_summary.csv）

### 文件路径
```
C:\Users\T-Cat\Desktop\XAS-FeO\all_center_neighbors_summary.csv
```

### 列结构

| 列名 | 含义 | 是否使用 |
|------|------|----------|
| `center_element` (A) | 中心元素（如 Fe、Ag、Al） | 索引用 |
| `pair` (B) | 元素对（如 Fe-O、Ag-Ag） | ✅ **主键，当前使用** |
| `count` (C) | 统计样本数量 | 参考 |
| `main_range_A_5to95` (D) | 5-95 百分位键长范围（Å），格式 `min-max` | 备用 |
| `median_A` (E) | 键长中位数（Å） | 参考 |
| `raw_range_A_minmax` (F) | 真实统计全范围（Å），格式 `min-max` | ✅ **当前使用** |
| `cutoff_used_A` (G) | 截断距离（Å），通常 3.00 | 参考 |

**当前方案**：使用 B + F 列。若模型输出键长违规率高，改用 D 列作更严格约束。  
**应用时机**：Step 4 评估阶段后处理，不在训练中直接使用。

---

## 2. FEFF 特征参考表（feff_features_all_stable_v2.csv）

**仅供 Agent 理解物理含义，不作为模型输入，不需要脚本读取。**

---

## 3. 人工读谱特征表

### 文件路径
```
主数据集: C:\Users\T-Cat\Desktop\XAS-FeO\feff_features_all_site_v2.csv
离子数据集: C:\Users\T-Cat\Desktop\XAS-FeO\feff_features_all_ionic_v3.csv
```

### 使用规则
- 第一列 `sample_dir` **忽略**（路径失效）
- 用第二列 `sample_name` 匹配数据文件夹名
- 两表重叠时：删除 ionic 表中的重复条目，以 site 表为准
- **主要用途**：Step 1.3 数据质量筛查

### 完整特征字段说明

#### XANES 特征（来自 xmu.dat）

| 特征名 | 物理含义 | 脚本用途 |
|--------|----------|----------|
| `xmu_Emin` / `xmu_Emax` | 谱的能量范围（eV） | 质量检查 |
| `xmu_npts` | 能量点数 | 质量检查 |
| `E0` | 吸收边能量（eV），K 边位置 | 确定元素，归一化参考 |
| `mu_at_E0` | E0 处吸收截面值 | 归一化参考 |
| `pre_peak_E` / `pre_peak_I` | 前驱峰能量和强度 | 氧化态、对称性指标 |
| `white_line_E` / `white_line_I` | 白线峰能量和强度 | 配位环境指标 |
| `post_peak1_E` / `post_peak1_I` | 后峰1能量和强度 | 局部结构指标 |
| `d1_pre_E/I`, `d1_post_E/I` | 一阶导数极值位置和值 | 边形状描述 |
| `area_pre`, `area_edge`, `area_white_line`, `area_post1` | 各区域面积 | 积分特征 |
| `pre_white_ratio`, `post_white_ratio` | 峰面积比 | 化学环境综合指标 |
| `pre_centroid_E`, `white_centroid_E` | 峰质心能量 | 对称性指标 |
| `flag_pre_valid` | 前驱峰是否有效（0/1） | **质量筛查** |
| `flag_white_valid` | 白线峰是否有效（0/1） | **质量筛查** |
| `flag_post_valid` | 后峰是否有效（0/1） | **质量筛查** |

#### EXAFS 特征（来自 chi.dat，k 空间）

命名规则：`k{n}chi_{kmin}_{kmax}_{统计量}`  
- `n` = 2 或 3（k 权重）  
- k 区间：`2_5`（2-5 Å⁻¹）、`5_8`（5-8 Å⁻¹）、`8_12`（8-12 Å⁻¹）  
- 统计量：`mean`、`std`、`max`、`min`、`area_abs`（绝对面积）、`rms`

| k 区间 | 物理意义 |
|--------|----------|
| 2-5 Å⁻¹ | 近邻壳层（第一配位层），受多重散射影响小 |
| 5-8 Å⁻¹ | 中程结构，键角信息 |
| 8-12 Å⁻¹ | 远程结构，精确键长信息 |

#### R 空间特征（FFT 变换后）

| 特征名 | 物理含义 | 重要性 |
|--------|----------|--------|
| `R1_peak_pos` | 第一壳层峰位（Å），近似对应最近邻键长（需加相移校正约 +0.3-0.5 Å） | **极高** |
| `R1_peak_height` | 第一峰高度，配位数指标 | 高 |
| `R2_peak_pos` | 第二壳层峰位（Å），Fe 体系通常为 Fe-Fe 或第二层 O | 高 |
| `R2_peak_height` | 第二峰高度 | 中 |
| `R1_area`, `R2_area` | 各峰面积，配位数积分指标 | 中 |
| `R1_R2_ratio` | 两峰面积比，结构类型判断指标 | 中 |

---

## 4. 数据质量筛查规则（Step 1.3 执行）

### Flag 含义
- `1` = 峰成功检测，特征有效  
- `0` = 峰不存在或检测失败，对应列为 NaN

### 分级策略（打标签，不删除）

```
quality_tier 分级：
  "A" → flag_pre_valid=1 AND flag_white_valid=1 AND flag_post_valid=1
  "B" → flag_white_valid=1（至少白线有效）
  "C" → 其余（谱形异常，训练时降权或排除）
```

### ⚠️ 特别注意
- **不要因 flag=0 删除 ionic 位点**：Li、Na 等轻元素天然无明显 pre-peak，flag=0 是物理正常现象，质量仍可为 B 级
- 所有数据保留记录，通过 `quality_tier` 字段标记

---

## 5. Step 1 操作要点

1. 加载 site_v2 + ionic_v3 两表，合并去重，生成统一质量清单
2. 为每个 `sample_name` 记录 `quality_tier`（A/B/C）
3. 键长约束表加载为字典 `{pair_str: (float_min, float_max)}` 并序列化保存，供 Step 4 调用
4. Step 1 输出的 `data_inventory.csv` 必须包含 `quality_tier` 列
