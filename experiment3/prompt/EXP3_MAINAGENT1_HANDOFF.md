# EXP3_MAINAGENT1_HANDOFF.md
# Experiment 3 Main Agent 1 交接文档与指令

> **写给 Experiment 3 Main Agent 1**
> **由 Experiment 2 Main Agent 2 撰写**
> **日期**：2026-04-09

---

## 你是谁，你要做什么

你是本项目 Experiment 3 的 Main Agent 1，接替 Experiment 2 继续工作。

你的核心任务是：**在 Experiment 2 训练好的模型基础上，加一个独立的原子类型分类 Head，把 Type Accuracy 从 25% 提升到 40% 以上**。坐标预测（RMSD=1.47Å）已经达标，不需要重新设计，只需维持不变。

你**不写代码**，你指挥 Sub-Agent 写代码。每个 Sub-Agent 完成一个小步骤后汇报，你确认无误再继续。

---

## 你需要理解的项目背景（必读）

### 项目是什么

给定一条 Fe K-edge XAS 谱（X 射线吸收谱），预测以 Fe 为中心的局部原子结构：最近 20 个邻居原子的元素类型和坐标。

### Experiment 2 做了什么，结果如何

用 DiffCSP（扩散模型）作为框架，加了一个 SpectrumEncoder 把谱图编码成条件向量，引导扩散过程生成原子结构。

最终结果（Holdout 787 个化合物盲测）：
- RMSD = 1.47 Å（坐标预测，随机基线 2.32Å，达标）
- Type Accuracy = 0.241（元素类型预测，随机基线 ~1%，但未达到目标 0.40）

Type Accuracy 卡在 25% 的原因：原子类型预测与坐标预测共享同一个 decoder，训练时两者的梯度互相竞争，type_loss 在 epoch ~100 就停止下降，继续训练对类型预测无效。

### Experiment 3 要做什么

加一个独立的 `TypeClassifier` MLP，直接从 SpectrumEncoder 的 latent 向量分支出来，专门预测每个位置的元素类型，不经过扩散 decoder。详细架构见 `EXP3_PROPOSAL.md`。

---

## 你需要持有的文件清单

用户应当在开始工作时给你以下文件：

| 文件 | 用途 | 来源 |
|------|------|------|
| `SHARED_00_v2.md` | 项目背景与约束 | outputs/ |
| `SHARED_01_DATA_MANIFEST.md` | 数据格式说明 | outputs/ |
| `EXP3_PROPOSAL.md` | 本次实验方案 | outputs/ |
| `experiment2/step3/xas_local_dataset.py`（Step4d 版） | 直接复用，不改 | 本地磁盘 |
| `experiment2/step3/spectrum_encoder.py` | SpectrumEncoder 定义 | 本地磁盘 |
| `experiment2/step3/diffusion_w_type_xas.py`（Step4c 版，无 % 1.） | 需要修改加 TypeClassifier | 本地磁盘 |
| `experiment2/step4/step4_2_train.py` | 参考，可能需要小改 | 本地磁盘 |
| `experiment2/step4/step4_3_sample.py` | 参考，可能需要小改 | 本地磁盘 |
| `experiment2/step4b/step4b_4_compute_metrics.py` | 评估脚本，需要扩展类型评估 | 本地磁盘 |
| `experiment2/step1/train_ids.txt` | 训练集 ID | 本地磁盘 |
| `experiment2/step1/val_ids.txt` | 验证集 ID | 本地磁盘 |
| `experiment2/step1/test_ids.txt` | 测试集 ID | 本地磁盘 |
| `experiment2/step1/holdout_1000_ids.txt` | Holdout（封存，Step5b 才用） | 本地磁盘 |

**注意**：`SHARED_02_SPECTRAL_AND_MODEL.md` 有已知错误（xmu 列索引有误），以 `SHARED_00_v2.md` 中的实测为准（data[:,0] 是能量，data[:,3] 是 μ(E)），不需要给 Sub-Agent。

---

## 你的工作流程

### Step 3b：统计元素词表

**任务**：扫描训练集所有样本，统计 20 个邻居原子的元素种类，建立词表。

**Sub-Agent 做**：写一个脚本，遍历 train_ids.txt 中的所有样本，用 `xas_local_dataset.py` 加载，收集所有出现过的原子序数，按频率排序，生成 `elem_vocab.json`（格式：`{"8": 0, "26": 1, ...}`，key 是原子序数字符串，value 是 class index）。

**你需要确认**：
- 词表大小 N_elem 在 [30, 80] 范围内
- 原子序数 8（O）和 26（Fe）必须在词表中，且排名靠前
- 保存路径：`experiment3/step3b/elem_vocab.json`

**完成后**：把 N_elem 告诉你，你记录下来，写入后续交接文档。

---

### Step 3c：修改 diffusion_w_type_xas.py，加入 TypeClassifier

**任务**：在原有文件基础上，加入 TypeClassifier 类并接入训练流程。

**Sub-Agent 做**：
1. 在文件中新增 `TypeClassifier` 类（架构见 EXP3_PROPOSAL.md 第 3.1 节）
2. 在 `CrystDiffPLModule.__init__()` 中实例化 TypeClassifier
3. 在 `forward()` 中，在 SpectrumEncoder 计算完 latent 之后，调用 TypeClassifier，计算 `type_ce_loss`，合并到 `total_loss = diffusion_loss + 0.5 * type_ce_loss`
4. 在 `sample()` 中，调用 TypeClassifier 输出的 argmax 作为最终类型预测（不用 decoder 的类型输出）

**你需要确认**：
- forward() 跑通，三个 loss 数值（diffusion_loss、type_ce_loss、total_loss）均非 NaN
- TypeClassifier 参数量在 200K-500K 之间（打印 `sum(p.numel() for p in model.type_classifier.parameters())`）
- 保存为：`experiment3/step3c/diffusion_w_type_xas_exp3.py`

---

### Step 4e：训练

**任务**：用 Step3c 修改后的模型从头训练。

**超参完全继承 Exp2**（见 EXP3_PROPOSAL.md 第 4 节）。

**Sub-Agent 做**：
- 把 Step3c 的模型文件和 Step4d 的 dataset 文件组合，写训练脚本
- 输出目录：`experiment3/step4e/checkpoints/`
- 每 10 个 epoch 打印：`total_loss`、`diffusion_loss`、`type_ce_loss`、`val_type_acc`

**你需要在 epoch 50 时收到一次中间汇报**，确认：
- `val_type_acc` > 0.05（否则停训，词表对齐有误）
- `val_rmsd`（采样 100 个样本快速估计）< 2.5 Å（否则把 lambda_type 降到 0.1 再继续）

**完成后**：记录 best val_loss、best epoch、val_type_acc、val_rmsd。

---

### Step 5b：评估与 Holdout 检验

**任务**：对 val/test/holdout 全部评估，输出完整指标报告。

**新增指标**（在原有基础上扩展评估脚本）：
- Type Accuracy Top-1 和 Top-3
- 按壳层分组（第一壳层≤2.5Å，第二2.5-3.5Å，第三3.5-4Å）分别统计 RMSD 和 Type Accuracy

**Holdout 要求**：只有在 val/test 结果达到验收标准后，才能运行 Holdout 评估。

---

## 工作原则

1. **不写代码，只出交接文档**。代码由 Sub-Agent 实现。
2. **每个步骤完成后确认再继续**，不跳步。
3. **lambda_type=0.5 是初始值，不是固定值**。如果 RMSD 比 Exp2 变差超过 0.2Å，授权 Sub-Agent 把它降到 0.2，不需要请示。如果 type_acc 没有提升，把它调到 1.0 再试一次。
4. **化学式/mp_id 禁止入模型**，始终坚守。
5. **Holdout 全程封存**，Step5b 之前不得接触。
6. **记录所有数字**：每个步骤的关键数字（loss、acc、epoch）都要记录在你的工作记录里，方便下一任接替。

---

## 验收标准（再次确认）

| 指标 | 目标 | 说明 |
|------|------|------|
| RMSD（val） | ≤ 1.6 Å | 不能显著差于 Exp2 |
| Type Accuracy Top-1（val） | ≥ 0.40 | 主要目标 |
| Type Accuracy Top-3（val） | ≥ 0.65 | 辅助参考 |
| RMSD（holdout） | ≤ 1.8 Å | 泛化检验 |
| Type Accuracy（holdout） | ≥ 0.35 | 泛化检验 |

---

*Experiment 2 Main Agent 2 撰写，2026-04-09*
