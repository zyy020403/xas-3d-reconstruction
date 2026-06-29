# SHARED_01_DATA_MANIFEST.md
# 数据清单与格式说明 — 所有 Sub-Agent 必读

> **本文档版本**: v1.0  
> **维护者**: Main Agent

---

## 1. 两个主数据文件夹

### 1.1 主数据集（无离子元素）
```
路径: C:\Users\T-Cat\Desktop\XAS-FeO\site_dataset\
文件夹数量: 26608 个
```
包含内容：以共价键为主的化合物中各元素各位点的 FEFF 计算结果。

### 1.2 离子元素补充数据集
```
路径: C:\Users\T-Cat\Desktop\XAS-FeO\test_missing_keep3_packed_A\
文件夹数量: 17087 个
```
包含内容：Sr、Li、Na 等离子键元素的 XAS 谱（同步辐射较少计算，单独存放用于评估风险）。

### 1.3 去重规则（重要！）
两个文件夹**可能存在重复**（文件夹名完全相同的条目）：
- 如果 `test_missing_keep3_packed_A` 中某文件夹名在 `site_dataset` 中已存在 → **删除 ionic 文件夹中的副本**，以 `site_dataset` 为准
- 判断标准：文件夹名完全相同（字符串完全匹配）

---

## 2. 文件夹命名格式

### 标准格式（含元素名称）
```
mp_{mp_id}_{formula}__feff_{element}_site_{nn}
```

| 字段 | 说明 | 是否可用于模型输入 |
|------|------|-----------------|
| `mp_id` | Materials Project ID（无实际意义） | ❌ 不可用 |
| `formula` | 化学式（如 Fe2N, Sc3FeC4） | ❌ **严禁**作为模型输入，会泄露元素种类和数量 |
| `element` | 当前谱的吸收元素（如 Fe, Sc, Li） | ✅ 可用，作为元素类型 embedding 的标识符 |
| `site_nn` | 位点编号，格式为两位数字（01, 02, 03...） | ✅ 可用于标识同一化合物的不同位点 |

### 特殊格式（仅单一金属元素时无元素标识）
```
mp_{mp_id}_{formula}__feff_site_{nn}
```
**⚠️ 重要处理规则**：若文件夹名中 `__feff_` 后直接跟 `site_`（没有元素名称），则该位点的吸收元素默认为 `{formula}` 中**唯一的金属元素**。

示例：
```
mp_1456_Fe2O3__feff_site_02  →  吸收元素 = Fe（Fe2O3 中唯一金属）
```
脚本解析文件夹名时**必须处理此边界情况**，否则会解析失败。

---

## 3. 文件夹内容（每个位点文件夹包含 3 个文件）

```
{folder_name}/
├── chi.dat               # k空间谱（χ(k)，EXAFS信号）
├── xmu.dat               # 能量空间谱（μ(E)，XANES信号）
└── POSCAR_supercell_fixed  # 超胞晶体结构文件（VASP POSCAR格式，训练标签）
```

### 3.1 chi.dat 格式
- EXAFS 信号，k 空间
- 通常为两列：`k`（波矢，Å⁻¹）和 `chi(k)`（无量纲）
- 文件开头可能有注释行（`#` 开头）

### 3.2 xmu.dat 格式
- 总吸收截面，能量空间
- 通常列为：`energy`（eV）、`xmu`、`mu0`、`chi` 等
- 文件开头可能有注释行（`#` 开头）
- XANES 信息（边前区、白线峰）主要在此文件中

### 3.3 POSCAR_supercell_fixed 格式
标准 VASP POSCAR 格式：
```
{注释行}
{缩放系数}
{晶格向量a: ax ay az}
{晶格向量b: bx by bz}
{晶格向量c: cx cy cz}
{元素种类列表}
{各元素数量列表}
Direct / Cartesian
{原子分数坐标列表}
```
**注意**：这是超胞（supercell），不是原胞，尺寸可能较大。

---

## 4. 化合物与位点的对应关系

一个化合物（由 `mp_id` 唯一标识）可能对应**多个文件夹**（多个位点）：

```
示例：Sc3FeC4（mp_3155）
  → mp_3155_Sc3FeC4__feff_Fe_site_01  （Fe 的 1 个位点）
  → mp_3155_Sc3FeC4__feff_Sc_site_01  （Sc 的位点 1）
  → mp_3155_Sc3FeC4__feff_Sc_site_02  （Sc 的位点 2）
  共 3 个文件夹 → 3 条谱 → 1 个结构
```

**数据组织逻辑**：
- 同一 `mp_id` 的所有位点共享**同一个晶体结构**（POSCAR 相同）
- 训练时需要将同一 `mp_id` 的所有谱**打包为一个样本**

---

## 5. 离子元素分类参考

以下元素在 `test_missing_keep3_packed_A` 中单独存储，视为"离子类"谱：
- **典型离子元素**：Sr、Li、Na（以及其他碱金属、碱土金属）
- 判断方法：若文件夹在 `test_missing_keep3_packed_A` 中且去重后仍保留 → 标记 `is_ionic=True`
- 若文件夹在 `site_dataset` 中 → 标记 `is_ionic=False`

---

## 6. 参考/辅助 CSV 文件（非模型输入，供脚本编写参考）

| 文件 | 路径 | 用途 |
|------|------|------|
| 键长约束表 | `C:\Users\T-Cat\Desktop\XAS-FeO\all_center_neighbors_summary.csv` | A列=中心元素，B列=元素对（如Ag-Ag），F列=真实统计最小/最大值；**当前使用B+F列** |
| 特征详解表 | `C:\Users\T-Cat\Desktop\XAS-FeO\feff_features_all_stable_v2.csv` | **不作为脚本输入**，仅供编写脚本时理解物理先验 |
| 主数据集特征表 | `C:\Users\T-Cat\Desktop\XAS-FeO\feff_features_all_site_v2.csv` | 人工读谱特征，第一列文件路径**忽略**（路径与当前不一致）；可能与ionic表有overlap，以此表为准 |
| 离子数据集特征表 | `C:\Users\T-Cat\Desktop\XAS-FeO\feff_features_all_ionic_v3.csv` | 同上，离子版本；与主表重复的删除此表中的条目 |

> **物理约束详细说明**见 `SHARED_04_PHYSICAL_CONSTRAINTS.md`（待补充后发布）

---

## 7. 1000 结构保留集说明

### 保留集目的
用于**最终盲测评估**，在整个 pipeline 调试完成后才用于评估，确保模型从未见过这些结构。

### 选取策略（已确认）
1. 按化合物结构类型（元素组成类别、配位环境）对所有化合物聚类
2. 每个聚类按比例采样，确保每个聚类中**仍有类似结构留在训练集**（保证测试覆盖性而不造成训练集盲区）
3. 保留集大小：**1000 个 mp_id**（注意是按化合物而非按位点计数）
4. 保留集清单文件将由 Step 1.4 生成，存放于：
   ```
   C:\Users\T-Cat\Desktop\DiffCSP-main\experiment\step1\holdout_1000_ids.txt
   ```

### 注意
- 保留集一旦确定**严禁后续步骤读取其谱文件或 POSCAR**
- Step 1 之后所有 agent 必须先检查当前处理的 mp_id 是否在保留集中
