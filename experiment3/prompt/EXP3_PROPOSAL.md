# EXP3_PROPOSAL.md
# Experiment 3 Proposal：独立原子类型分类 Head
# ★ 定稿版 ★

> **状态**：LOCKED
> **日期**：2026-04-09
> **基于**：Experiment 2 最终结果（RMSD=1.47Å，TypeAcc=0.249）

---

## 0. 一句话任务

在 Experiment 2 的基础上，加一个独立的原子类型分类 Head，使 Type Accuracy 从 25% 提升到 40%+，同时保持 RMSD ≤ 1.6 Å。

---

## 1. Experiment 2 遗留问题与本次改进逻辑

### 1.1 为什么 Exp2 的 Type Accuracy 卡在 25%

Exp2 中原子类型预测是扩散模型的附带任务，和坐标预测共享同一个 decoder。训练时 `type_loss` 在 epoch ~100 就停止下降，继续训练只改善坐标，不改善类型。

根本原因有两个：

第一，**优化目标竞争**。Decoder 同时优化坐标和类型，梯度信号互相干扰。坐标预测在后期需要精细的空间调整，类型预测需要的特征则更偏全局（整体化学环境），两者对 latent 向量的需求不同，共用一套参数导致两者都不能充分优化。

第二，**类型信号在当前路径上太弱**。SpectrumEncoder 的输出主要被 condition 注入 CSPNet，用于引导每一步的坐标去噪。类型信息在这条路径上经过多次变换后已经稀释，到达类型预测头时剩余信号不足。

### 1.2 本次改进逻辑

直接从 SpectrumEncoder 的 latent 向量拉出一条独立分支，用一个专用 MLP 做元素分类，不经过扩散 decoder，避免梯度竞争和信号稀释。这条分支用交叉熵损失单独优化，同时保留原来的坐标扩散路径不变。

---

## 2. 不变的部分（全部继承 Exp2）

以下所有内容与 Experiment 2 完全相同，不得修改：

```
Dataset          xas_local_dataset.py（Step4d 版，L=6，min-image 折叠）
虚拟晶格          L=6.0 Å，diag(6,6,6)
坐标系            [-0.5, 0.5]，forward() 无 % 1.
N_NEIGHBORS      20
训练超参          batch=16，lr=1e-4，bf16，num_workers=0
路径常量          全部继承（见第6节）
化学式禁止入模     model 输入中不得出现 formula / mp_id
Holdout 封存      holdout_1000_ids.txt 全程禁止接触
```

---

## 3. 新增内容：TypeClassifier Head

### 3.1 架构

```
SpectrumEncoder(xmu, chi1, feff_feats) → latent (B, 256)
                        │
                        ├──→ [原有路径] condition = cat(time_emb, latent) → CSPNet → 坐标扩散
                        │
                        └──→ [新增路径] TypeClassifier → (B, 20, N_elem)
```

TypeClassifier 是一个三层 MLP：

```
输入：latent (B, 256)
Layer 1：Linear(256, 256) + LayerNorm + SiLU
Layer 2：Linear(256, 256) + LayerNorm + SiLU
输出层：Linear(256, 20 × N_elem) → reshape → (B, 20, N_elem)
```

其中：
- `20` 是固定邻居数
- `N_elem` 是训练集中出现过的元素种类数（Step3b Agent 统计，预计 40-60 种）
- 输出经过 softmax 得到每个位置上各元素的概率分布

### 3.2 元素词表

训练前需要统计训练集所有样本的邻居原子序数，建立一个元素词表（element vocabulary），格式为 `{atomic_number: class_index}`。词表在 Step3b 生成，保存为 `experiment3/step3b/elem_vocab.json`，训练和推断时共用。

### 3.3 损失函数

```python
total_loss = diffusion_loss + lambda_type * type_ce_loss

# diffusion_loss：原版 DiffCSP 的坐标+晶格扩散损失（不变）
# type_ce_loss：对 20 个位置各自做交叉熵，取平均
# lambda_type：超参，初始值 0.5，如果 RMSD 比 Exp2 变差超过 0.2Å 则降低到 0.2
```

**注意**：TypeClassifier 的输入是 latent，不依赖于扩散步骤 t，所以它在每个 batch 只计算一次，不随 t 变化。

### 3.4 评估时的类型预测

推断（采样）时，TypeClassifier 直接从 SpectrumEncoder 的 latent 取最大概率的元素，作为每个位置的类型预测。不使用扩散 decoder 的类型预测输出。

---

## 4. 训练策略

```
max_epochs：500
early_stop patience：30（监控 val_total_loss）
scheduler：CosineAnnealingLR(T_max=500)（不变）
lambda_type 初始值：0.5
```

**开训前强制检查**（必须全部通过）：
1. 取 5 个样本，确认 frac_coords ∈ [-0.5, 0.5]（继承 Exp2 的 Dataset 验证）
2. 打印 `N_elem`（词表大小），确认在 [30, 80] 范围内
3. 跑一次 forward()，确认 `type_ce_loss` 非 NaN，`diffusion_loss` 在合理范围
4. 检查 TypeClassifier 的参数量（应在 200K-500K 之间，不应过大）

**早期监控（epoch 50）**：
- 若 `val_type_acc` < 0.05（低于随机猜），说明词表或 label 对齐有误，立即停训汇报
- 若 `val_rmsd` > 2.0 Å（在 val 上，采样检验），说明 TypeClassifier 梯度干扰坐标，把 lambda_type 降到 0.1

---

## 5. 评估指标（新增）

在 Exp2 的基础上，新增：

```
val_type_acc_topk：Top-1 和 Top-3 准确率分别统计
  Top-1：预测概率最高的元素 = 真实元素
  Top-3：真实元素在概率前三名内

按壳层分组统计：
  第一壳层（≤ 2.5Å 内邻居）：类型预测通常更准（主要是 O）
  第二壳层（2.5-3.5Å）
  第三壳层（3.5-4.0Å）：类型预测通常更难
```

---

## 6. Pipeline 步骤

```
Step 3b：统计元素词表，保存 elem_vocab.json（新增）
Step 3c：修改 diffusion_w_type_xas.py，加入 TypeClassifier（新增）
Step 4e：重新训练（继承 Step4d 的 Dataset，加 TypeClassifier loss）
Step 5b：Holdout 检验（重新评估，与 Exp2 结果对比）
```

---

## 7. 文件存储规范

```
根目录：experiment3/
子目录：step3b/ step3c/ step4e/ step5b/
脚本命名：step{N}{letter}_{描述}.py
```

---

## 8. 验收标准

| 指标 | 目标 | 说明 |
|------|------|------|
| RMSD（val） | ≤ 1.6 Å | 不能显著差于 Exp2（1.47Å） |
| Type Accuracy Top-1（val） | ≥ 0.40 | 主要目标，比 Exp2（0.249）提升 60% |
| Type Accuracy Top-3（val） | ≥ 0.65 | 辅助参考 |
| pred_in_cutoff | ≥ 15/20 | 维持 Exp2 水平 |

---

## 9. 关键路径常量（继承）

```python
DATA_ROOT     = r"C:\Users\T-Cat\Desktop\DiffCSP-main\site_dataset_Fe_only_oxide_one_site"
EXP2_ROOT     = r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment2"
EXP3_ROOT     = r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment3"
FEFF_FEAT_CSV = r"C:\Users\T-Cat\Desktop\DiffCSP-main\tesst_feff_features_all_full_v4.csv"
STEP1_DIR     = EXP2_ROOT + r"\step1"   # 继续用 Exp2 的 train/val/test/holdout 划分
L             = 6.0
N_NEIGHBORS   = 20
```

---

*Main Agent 2 定稿，2026-04-09*
