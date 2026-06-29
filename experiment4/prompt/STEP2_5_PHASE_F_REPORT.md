# STEP 2.5 PHASE F 摘要 — Option D 剔除诊断

**报告对象**：DiffCSP-Exp4-Main-Agent 2
**Sub-Agent**：DiffCSP-Exp4-Step2.5-SubAgent
**目的**：MA 决定是否走 Option D（剔除 incompat + 999 个 multiset-mismatch）的依据数据
**结论**：**数字支持 Option D 走得通**，影响在可接受范围

---

## 1. 三个核心数字（MA 关心的）

| 维度 | 数值 | 评价 |
|---|---|---|
| **保留样本** | 75,637 / 128,382 = 58.92% | 与 MA 预估完美吻合 |
| **元素覆盖** | **88 / 88 元素全保留** ✓ | 零损失，最关键的指标 |
| **Split 比例** | 80.00 / 10.08 / 5.92 / 4.00 % | 与目标 80/10/6/4 几乎对齐，**无需重做 split** |
| mp_id 完全消失 | 5,986 / 41,431 = 14.45% | 中等损失，可接受 |
| **每 split 元素覆盖** | train=88, val/test/holdout=84 | **与原数据一致**，没有元素从 split 中消失 |

---

## 2. 红旗 / 黄旗 / 绿旗清单

### 🟢 绿旗（4 项）
1. **88 元素全部保留**：包括稀有气体 Ne/Ar/He/Kr (drop 0-37.5%)，锕系 Ac/Pm (drop 0%)。Exp4 全元素覆盖目标完全保住。
2. **Split 比例自动保持**：keep_pct ≈ orig_pct，不用碰 Step 1 的 split 文件。
3. **每 split 元素覆盖一致**：原 val/test/holdout 是 84 元素，剔除后还是 84 元素。Test/holdout 的评估覆盖完整保留。
4. **75,637 样本对全元素学习足够**：每元素平均 ~860 样本，主流元素（O/Li/P/F/Fe 等）至少 1,200+ 样本。

### 🟡 黄旗（2 项）
1. **O 元素掉到 5,311（drop 76%）**：从 22,441 暴跌。但 5,311 仍是 dataset 中最大元素，超过其他任何元素。仍可学。
2. **F 元素掉到 1,291（drop 66%）**：1,291 仍 OK，对单元素训练而言。

### 🔴 红旗（仅 1 项，且影响有限）
1. **11 元素 < 200 样本**，其中：
   - 7 个**原本就少**（不是过滤造成的）：Ne(1), Ar(2), He(3), Kr(5), Xe(45), Ac(47), Pm(71)
   - 4 个**过滤后才掉到 200 以下**：Pa(52), Np(102), Tc(108), Pu(119)
   - 这 4 个都是锕系/超铀元素 + 锝（Tc 是 7d 过渡金属，常 radioactive）
   - 这些元素本身在原数据集就属于稀有，对模型整体影响小
   - **不阻塞 Option D**

---

## 3. mp_id Damage State 分布（关键观察）

| 状态 | 数量 | 占比 | 含义 |
|---|---:|---:|---|
| Untouched | 12,312 | 29.72% | 该 mp_id 的所有谱保留 |
| Partially dropped | 23,133 | 55.84% | 失去某些元素谱但 ≥1 个保留 |
| Fully dropped | 5,986 | 14.45% | 该 mp_id 整个不见了 |

**对 Step 3 训练**：partially_dropped 对单 sample 训练**完全无影响**——每个 sample 独立喂模型，不依赖同 mp_id 的其他元素谱完整性。所以 55.84% partially_dropped **不是问题**。只有 14.45% fully_dropped 是真正的损失。

---

## 4. Option D vs Option B 对比（最终决策维度）

| 维度 | Option D（剔除） | Option B（分层 + 随机采样） |
|---|---|---|
| 数据量 | 75,637（-41%） | 128,382（保留） |
| Label 严格对齐 | ✓ 100% | ✗ 40% incompat 走随机采样平均 |
| 元素覆盖 | 88/88 ✓ | 88/88 ✓ |
| 实现复杂度 | **零** — Dataset 加 filter | 高 — Dataset 分支 + 预算 ~2 GB shell pickle |
| 训练成本 | 标准 | incompat 走 brute-force on-the-fly |
| 评估清洁性 | ✓ test/holdout 严格对齐 | ✗ test/holdout 也含 40% incompat 噪声 |
| 调试成本 | 低 | 中 — 多分支逻辑 + cache invariants |
| 物理上正确性 | 严格 | 隐式 site-averaging（多 epoch 平均） |
| 失败时回退路径 | 容易 — 后续可加回 incompat | 已最大化数据，无回退空间 |

---

## 5. Sub-Agent 推荐 — **Option D**

理由（按重要性排序）：

1. **75K 样本对深度学习足够**：CIFAR-10 每类 5K 训练，Resnet 性能很好；NLP 数据集典型 train 也 50K 量级。Exp4 75K 不是数据瓶颈。

2. **88 元素全保留 + 主流元素都 ≥ 1.2K**：Exp4 核心目标"全元素架构泛化"完全保住。

3. **简单 pipeline = 更少 bug 面**：Option B 的随机采样 + 多 site cache 引入 3-5 个新 bug 入口。Step 3-5 已经够复杂，不必要再叠加。

4. **评估干净**：test/holdout 是严格对齐数据。如果 Step 5 看到 RMSD 偏高，归因清晰——不是 label noise，是模型问题。Option B 下评估自身带 40% noise，归因混乱。

5. **可逆性**：D 跑通后如果发现 75K 不够，**还能后续加回 incompat 走 B 的随机采样**（再花 1-2 周）。反过来从 B 退到 D 反而难（已经训了带噪 model）。

6. **MA 倾向 D 是正确的直觉**：simpler is better when "complex alternative does not buy you proportional benefit"。Option B 的 41% 数据增益 vs 它带来的 label noise，ROI 不见得正。

---

## 6. 实现影响（如果 MA 确认 D）

非常轻量。Step 3 Dataset 只需：

```python
# 在 Dataset 初始化时
tag_df = pd.read_csv("site_equivalence_tag.csv")
drop_mask = (tag_df["tag"] == "incompat") | \
            ((tag_df["tag"] == "near_equivalent") & 
             (tag_df["n_unique_shell1_multisets"] > 1))
valid_samples = set(tag_df[~drop_mask]["sample_name"])

# 然后过滤现有的 train/val/test/holdout id 文件
self.sample_names = [s for s in self.sample_names if s in valid_samples]
```

**不需要**重做 split、不需要重写 shell_boundaries.pkl、不需要 Phase E（site averaging）。

Step 2.5 实际上**今天就可以收工**，进入 Step 3 交接文档。

---

## 7. 请 MA 最终拍板

**决策**：是否确认走 Option D？

- [a] **确认 D，进入 Step 3 交接**（Sub-Agent 推荐）
- [b] 改回 Option B（数据更全但实现复杂）
- [c] 其他考虑

如果 [a]：Sub-Agent 立即开始写 Step 3 交接文档，包含：
1. 服务器上传清单（4 个 step1 产物 + shell_boundaries.pkl + site_equivalence_tag.csv + brute-force neighbor function 作为 utility module）
2. Exp2 脚本审计 + 改动点（xas_local_dataset.py 重写 + spectrum_encoder.py 路径改 + diffusion_w_type_xas.py 配置更新）
3. Dataset 类伪代码（含 incompat filter）
4. 前向测试协议（小批量 loss 检查）

预计这份交接文档篇幅是目前所有报告里最长的——Step 3 是这次实验真正的"工程大头"。

---

**Sub-Agent 待命。等 MA [a/b/c] 决定。**
