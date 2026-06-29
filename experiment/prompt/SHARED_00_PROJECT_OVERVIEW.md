# SHARED_00_PROJECT_OVERVIEW.md
# 项目总背景文档 — 所有 Sub-Agent 必读

> **本文档版本**: v1.0  
> **维护者**: Main Agent  
> **用途**: 所有 Step Agent 在开始任何工作前必须完整阅读本文件及其他 SHARED_XX 文件

---

## 1. 项目目标

本项目目标是构建一个**多位点 XAS 谱 → 晶体结构**的逆问题求解 pipeline。

具体来说：给定一个化合物中所有吸收位点的 XAS 谱（chi.dat / xmu.dat），预测还原出该化合物的完整晶体结构（晶格参数 + 原子坐标，即 POSCAR 格式）。

骨干模型基于 **DiffCSP**（扩散模型做晶体结构预测），对其进行条件化改造，以多位点 XAS 谱作为条件输入。

---

## 2. 科学背景：多位点谱的处理哲学

### 2.1 核心问题
一个化合物中可能存在多种元素（Fe、Sc、Li、Sr 等），每种元素的每个不等价位点都对应一条独立的 XAS 谱。这些谱**不能平均**，因为：
- Fe 的谱描述 Fe 周围的局部几何环境
- Li 的谱描述 Li 周围的局部几何环境
- 两者提供的是结构的不同侧面，强行平均会丢失各自的局部信息

### 2.2 "盲人摸象"聚合策略（已确认设计方向）
> 每个人以自己的角度摸到的东西描述出来，进行汇总，才有可能得到整体的结构。

对应的 ML 实现：

| 层级 | 操作 | 说明 |
|------|------|------|
| 单谱编码 | Element-conditioned Spectrum Encoder | 每条谱 + 元素类型 embedding → 局部结构 embedding；Fe 和 Li 的编码器共享权重但通过 embedding 区分 |
| 离子区分 | `is_ionic` flag embedding | 对离子键元素（Sr、Li、Na 等，来自 ionic 数据集）加额外标记，让模型自适应其置信度 |
| 多谱聚合 | Set Transformer / Cross-Attention | 所有 site 的 embedding 组成一个集合，通过排列不变（permutation-invariant）的注意力机制聚合为结构级 embedding |
| 条件扩散 | DiffCSP conditioning | 结构级 embedding 作为条件信号注入 DiffCSP 扩散过程 |

**关键约束**：聚合操作必须是排列不变的（site_01 和 site_02 的顺序不影响结果）。

---

## 3. 整体 Pipeline 结构（Step 概览）

```
Step 1: 数据预处理与清洗
  Step 1.1 — 扫描两个数据文件夹，去重，建立数据清单
  Step 1.2 — 解析谱文件（chi.dat, xmu.dat）和 POSCAR 文件
  Step 1.3 — 从 feff_features CSV 提取特征，验证物理约束
  Step 1.4 — 划分训练/验证/测试集（含 1000 结构保留集策略）

Step 2: 谱编码模块
  Step 2.1 — 单谱编码器（1D CNN or Transformer）实现
  Step 2.2 — 多位点排列不变聚合模块实现

Step 3: DiffCSP 条件化改造
  Step 3.1 — 修改 DiffCSP 接受谱条件输入
  Step 3.2 — 训练配置与超参数设定

Step 4: 训练与验证
  Step 4.1 — 训练脚本与监控
  Step 4.2 — 验证集评估指标计算

Step 5: 保留集最终评估
  Step 5.1 — 1000 结构保留集盲测评估
```

---

## 4. 重要约束与禁止事项

### ⚠️ 严禁数据泄露
- **严禁**从文件夹名称中读取化学式（如 `Fe2N`、`Fe4N`）作为模型输入
- 文件夹名格式为 `mp_{id}_{formula}__feff_{element}_site_{nn}`，其中 `{formula}` 部分**绝对不能**被任何脚本作为特征输入模型
- 模型能用的信息**只有**谱文件（chi.dat, xmu.dat）和 POSCAR（作为训练标签）

### ⚠️ 元素种类信息保护
- 谱的元素种类（如 Fe、Sc）只能作为**内部 embedding 标识**使用，不能作为化学组成信息直接输入
- 如果脚本需要知道"这是什么元素的谱"，应从文件夹名中提取 `{element}` 字段（不是 formula），这是允许的，因为做 XAS 实验时知道用了哪种元素边

---

## 5. DiffCSP 代码库位置

```
C:\Users\T-Cat\Desktop\DiffCSP-main\
```

参考其原始目录结构（见 repo_structure.txt），本项目的所有新增脚本**不要修改原始 diffcsp/ 目录内的文件**，所有新增内容存放在 experiment/ 子目录下。
