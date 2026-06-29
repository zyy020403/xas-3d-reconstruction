# SHARED_03_SPECTRAL_STRATEGY.md
# 多位点谱处理策略 — 所有 Sub-Agent 必读

> **本文档版本**: v1.0  
> **维护者**: Main Agent  
> **状态**: 已确认设计方向

---

## 1. 核心问题

一个化合物可能有多种元素、每种元素有多个不等价位点，因此对应**多条 XAS 谱**。
这些谱**绝对不能简单平均**，原因：

- 不同元素的谱描述的是完全不同的局部化学环境
- 即使同种元素的不同位点，其局部对称性也可能不同
- 离子键元素（Li、Na、Sr）的谱特征与共价键元素有系统性差异

---

## 2. 设计哲学："盲人摸象"聚合

> 每条谱 = 一个盲人从自己的视角摸到的局部结构描述  
> 模型目标 = 汇总所有视角，重建完整的晶体结构

### 2.1 关键性质要求
- **排列不变性（Permutation Invariance）**：同一化合物的谱集合，无论 site_01 和 site_02 谁先谁后，输出的结构 embedding 必须相同
- **元素区分性**：Fe 的谱和 Li 的谱必须被区别对待，不能混用同一编码权重（需要元素类型条件化）
- **离子/共价可区分性**：来自 ionic 数据集的谱需要特殊标记，让模型学习其置信权重

---

## 3. 架构设计（已确认）

### 3.1 整体数据流
```
输入: {(spectrum_i, element_i, is_ionic_i)} for i in all_sites_of_compound
         ↓
[单谱编码器] × N sites
         ↓
{embedding_1, embedding_2, ..., embedding_N}  (无序集合)
         ↓
[排列不变聚合器 (Set Transformer)]
         ↓
structure_embedding (固定维度向量)
         ↓
[条件化 DiffCSP 扩散模型]
         ↓
预测的晶体结构 (lattice + coordinates)
```

### 3.2 单谱编码器（Spectrum Encoder）
- **输入**: 谱信号（chi.dat 或 xmu.dat 的数值序列） + 元素类型 embedding
- **结构**: 1D CNN 或 1D Transformer（具体由 Step 2.1 决定）
- **元素条件化**: 将原子序数或元素 one-hot 编码投影为向量，与谱特征 concat 或相加
- **输出**: 固定维度的局部结构 embedding（建议 256 或 512 维）

```python
# 伪代码示意
element_emb = element_embedding_layer(atomic_number)   # [d_elem]
spectrum_feat = spectrum_encoder_1d(spectrum_signal)    # [d_spec]
site_embedding = MLP(concat(spectrum_feat, element_emb))  # [d_site]
```

### 3.3 离子标记
```python
ionic_flag_emb = ionic_embedding_layer(is_ionic)  # 二值 embedding [d_ionic]
site_embedding = site_embedding + ionic_flag_emb   # 加和融合
```

### 3.4 排列不变聚合器（Set Aggregator）
- **输入**: N 个 site_embedding 的集合（N 可变，不同化合物位点数不同）
- **结构**: Set Transformer（参考论文 *Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks*）
- **核心操作**: Induced Set Attention Block (ISAB) + Pooling by Multihead Attention (PMA)
- **输出**: 固定维度的结构级 embedding（与 N 无关）

```python
# 伪代码示意
site_embeddings = [enc(spec_i, elem_i, ionic_i) for i in sites]  # [N, d_site]
site_embeddings_tensor = stack(site_embeddings)  # [N, d_site]
structure_embedding = set_transformer(site_embeddings_tensor)  # [d_struct]
```

---

## 4. 谱文件选择策略

每个位点文件夹有 `chi.dat`（k空间）和 `xmu.dat`（能量空间）两个谱文件。

| 谱类型 | 文件 | 主要信息 | 建议 |
|--------|------|----------|------|
| EXAFS | chi.dat | 精确键长、配位数、径向分布 | 对结构重建最直接 |
| XANES | xmu.dat | 氧化态、局部对称性、电子结构 | 补充化学环境信息 |

**初步策略**：优先使用 `chi.dat`（EXAFS），因其与几何结构关联最直接。  
**后续可探索**：双路输入（chi + xmu 分别编码后 concat），或仅用 xmu 的 XANES 部分。

---

## 5. 变长输入处理

不同化合物的位点数量 N 不同（1 到几十不等），需要处理变长输入：

```python
# DataLoader 中使用 padding + mask 策略
# 或使用 pack_sequence / 动态批处理
# Set Transformer 天然支持变长输入，推荐使用
```

批处理时，同一 batch 内的样本需要 padding 到最大 N，并使用 attention mask 屏蔽 padding 位置。

---

## 6. 训练标签

- **目标**：预测 POSCAR_supercell_fixed 中的晶体结构
- **注意**：POSCAR 是超胞，可能需要转换为原胞（primitive cell）或统一处理方式（由 Step 1.2 确认）
- **与 DiffCSP 对接**：DiffCSP 原始输入格式参考 `diffcsp/pl_data/dataset.py`，Step 3 改造时需要匹配

---

## 7. 后续可扩展方向（暂不实现，供参考）

- **谱的不确定性建模**：对 ionic 谱赋予可学习的不确定性权重（Evidential Deep Learning）
- **对比学习预训练**：同一化合物的所有谱作为正样本对，不同化合物的谱作为负样本
- **物理先验约束**：在损失函数中加入键长约束项（参考 `all_center_neighbors_summary.csv`）
