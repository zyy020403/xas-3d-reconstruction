# Experiment 7 Proposal: Conditional WGAN-GP with Distance-Matrix Discriminator + Curriculum Pairwise Constraint
# XAS → 局部原子结构预测(架构第三条路径)

> **状态**: DRAFT v6
> **日期**: 2026-05-10
> **作者**: Exp7-MA (Main Agent 7,GAN 方向主负责)
> **v6 变更** (incorporating 用户 round 6 feedback: 数据处理 verify 框架 — Day 0 SA1 决策):
> - **关键发现**: Exp5' (diffusion) 和 Exp6 (transformer with Hungarian) 的 dataset 不依赖 "atom_types 按距离 sorted",但 Exp7 GAN one-shot generation 的 type CE 严格依赖 GT 顺序与 generator 输出对齐。**v1-v5 漏判这点**,直接默认沿用 Exp5'/Exp6 dataset
> - **新增 §5.5 数据处理 verify 框架**: 5 个 verify 点 + 决策框架 (SA1 Day 0 必做)。**Distance matrix 计算位置** (Option A on-the-fly vs Option B 改 dataset) 由 SA1 verify 完 Exp5'/Exp6 dataset 后决定
> - **新增 §5.6 Atom types 排序方案**: 不锁定 dataset 是否 sorted,但锁定 SA1 verify 后的决策矩阵 (sorted 直接用 / unsorted 在 collate_fn 重排 / 模糊 raise)
> - **附录 B 加第 18 条**: SA1 Day 0 数据处理 verify 流程,verify 完成前禁止开始 implementation
> **v5 变更**: 路径全部锁死实测路径 + 利用 Exp6 已完成 step1 产出
> **定位**: Exp5 系列 wrap-up + Exp6 并行进行期间,用户拍板开 Exp7 GAN 方向。本实验与 Exp6 共享物理评估体系 (step5_3 7 项 + gate + collapse) 但走完全独立的架构路径
> **关键基础文档**:
>   - `EXPERIMENT5_SERIES_FINAL_REPORT.md` (Exp5'-MA 撰写 2026-05-10) — Exp5 系列三阶段全部教训,**v1 直接继承所有 14 条 lesson + 5 份 errata 教训**
>   - `EXP6_PROPOSAL_v8.md` (Exp6-MA 撰写 2026-05-01) — Exp6 Transformer 路径的设计;Exp7 v1 沿用其大部分 SOP / vocab / dataset 处理框架,只换架构
> **核心 GPU 配置**: 用户两块 RTX 4090,Exp6 占用其一并行训练中,Exp7 用另一块单卡训练
> **设计原则**: 与 Exp6 v8 一致 — 工作量最小化,优先复用 Exp5' 已验证组件 (dataset L=20 + pairwise loss + shell_boundaries.pkl + step5_3 评估)

---

## 0. 一句话目标

用 **conditional WGAN-GP + distance-matrix discriminator** 替换 Exp5 系列的扩散 decoder 和 Exp6 的 transformer decoder,验证 GAN 路径在 XAS → 局部结构任务上的可行性,特别是 discriminator 能否**隐式学到 shell 结构**(绕开 Exp5' 鸡蛋问题 + Exp5'' 几何冲突问题)。

---

## 1. 为什么开 Exp7 GAN

### 1.1 直接动机 — Exp5 系列 L14 lesson

`EXPERIMENT5_SERIES_FINAL_REPORT.md §7.6 L14` 明确:

> "Loss-level fixes 在固定架构上有上限。Exp5 三阶段(v2 + Exp5' + Exp5'')都 post-hoc 修 loss,model 架构未变。突破必须从架构层注入 inductive bias(等变 / 图卷积 with shell-aware edge / Transformer attention / GAN discriminator)"

**Exp6 Transformer** 探索的是 "attention 隐式学 shell" 假设。**Exp7 GAN** 探索的是 "discriminator 隐式学整体结构合理性" 假设。两者正交,同时跑互不干扰。

### 1.2 GAN 路径的特殊优势(相对 Exp5 / Exp6)

| 维度 | Exp5 系列 (diffusion) | Exp6 (transformer) | **Exp7 (GAN)** |
|---|---|---|---|
| 监督信号 | 显式 loss (pairwise + shell + cls + pos) | 同 (DETR-style + 三件套) | **隐式 (discriminator 学 real distribution)** + 显式 (pairwise) |
| Shell 学习方式 | Explicit shell loss → 鸡蛋问题 + 几何冲突 | Attention 隐式 (假设) | **Discriminator 隐式 (假设)** |
| 多样性 | 扩散 sampling 天然多样 | 一对一回归,需 query | **Generator z noise 天然多样** |
| 训练稳定性 | 高 (EarlyStop 用满) | 中 (DETR 经典慢收敛) | **低 (GAN 经典)** ⚠️ — 主要 risk |
| 训练时长 | ~ 71h (Exp5 系列) | ~ 1.5-2 天 (Exp6 估) | **~ 20-40h 估** |
| Sample 时长 | 4h (1000 step) | < 30 min (one-shot) | **< 30 min (one-shot)** |

### 1.3 不取代 Exp6,是并行验证

Exp7 v1 **不会**与 Exp6 互相干扰:
- Exp6 在 4090-0 训练 transformer (~ 1.5-2 天)
- Exp7 在 4090-1 训练 GAN (~ 20-40h)
- 两者共享 dataset (L=20 cache) + 评估脚本 (step5_3) + shell_boundaries.pkl,代码层 0 冲突
- Exp6 verdict 出来后,Exp7 可立即对比 (但 Exp7 已开,改动有限)

### 1.4 任务本质 — Inverse Design + One-to-Many (v3 重要补充)

**用户 round 2 提醒**: 这个任务的本质不是普通 "序列 → 3D 结构" 1-to-1 回归,而是 **inverse design + one-to-many 反问题**:

- **多个不同的局部结构可以产生几乎相同的 XAS 谱图**(物理上的不可分辨性)
- 这不是 model "学不好"导致的,**而是 spectrum → structure 这个映射本身在物理上多对一**
- CNO (C/N/O) 在元素维度几乎完全无法区分(Exp4/5 已实测 shell-1 elem score = 0.005)
- Exp5'/Exp4 评估端把 CNO 合并视为同一虚拟类,就是承认这一物理边界

**这对 GAN 设计的根本影响**:

1. **z noise 不是 "GAN 的 risk",而是任务本质需要**:
   - 对每个 spectrum,Generator 应该有能力**输出多个合理的不同结构**
   - 这是 conditional GAN 的天然优势 (cGAN 论文 Mirza 2014 已显式提到 multi-modal output 是 cGAN 的设计目标)
   - 类比文献: PMC 论文 "Inverse design of structural color: finding multiple solutions via cGAN" 直接对应任务范式,每个 target 平均生成 3.58 个 valid solution

2. **Mode collapse 是双重灾难** (Exp7-specific):
   - 一般 GAN: mode collapse 丢失多样性,quality 仍可
   - **Exp7: mode collapse 不仅丢多样性,还会把"谱图相似的不同真实结构"全 collapse 到同一个预测**,直接破坏评估
   - 因此 mode_diversity_per_spectrum 不是 GAN 训练稳定性的副指标,**是 Exp7 验收的必须项**(见 §11.1)

3. **Discriminator 的设计意义被强化**:
   - D 输入 distance matrix,等价于学到 "所有 valid 结构都在某个 distribution 上"
   - 同一 spectrum 条件下,D 接受 multiple modes (不强求唯一答案)
   - 这与 1-to-1 回归 (DETR matched loss、扩散 noise prediction) 的本质区别就在这里

4. **CNO 合并的物理意义** (sote 与 Exp4/5/6 完全一致):
   - 训练端: type CE 保持 88-way 分类 (保 architecture 与 Exp4/5/6 对照可比)
   - 评估端: step5_3 评估 CNO 合并 (Exp5' SA-METRICS-V3 已实现)
   - 这不是设计缺陷,是 inverse problem 物理边界的承认

**v3 因此调整设计语言**: "Exp7 验证 GAN 路径在 XAS → 结构 inverse design 任务上的可行性",而非 "GAN 路径在 XAS → 结构生成上的可行性"。**inverse design** 比 generation 更精准描述任务范式。

---

## 2. Exp5 系列 4 条 must-do 完整继承(L_SERIES §13.3)

Exp7 v1 必须严格沿用 Exp5 系列收尾时的 4 条 must-do(`EXPERIMENT5_SERIES_FINAL_REPORT.md §13.3`):

| Must-do | Exp7 v1 落实位置 |
|---|---|
| **Sanity**: Cartesian sanity 100/100 (L=20 + CUTOFF_R=10 / errata 3) | §3.1 + §附录 B 第 3 条 |
| **Loss**: Pairwise min distance penalty (λ=1.0) 沿用 (Exp5' gate 64% 硬证) | §6.2.1 + §附录 B 第 5 条 |
| **Verdict**: step5_3 7 项 + gate + collapse 双指标 (errata 4 §5.3) | §10 + §附录 B 第 7 条 |
| **Dry-run**: 训练 active + 评估 active 双层 (L3 + L4) | §11.3 risk 1 + §附录 B 第 6 条 |

### 2.1 不沿用的项(显式声明)

| 项 | 不沿用理由 |
|---|---|
| `_shell_distance_loss` (Exp5'/'' 三件套) | Exp5' 鸡蛋问题未解决,Exp5'' 候选 A failure。Exp7 用 discriminator 隐式学 shell,**不再加 explicit shell loss** |
| `_shell_count_loss` (Exp5'/'' 三件套) | 同上 |
| `_density_loss` (Exp4 / Exp5 v2) | errata 2 §1 已证为 88 元素任务的塌缩剂,Exp7 不引入 |
| TypeClassifier head (Exp3) | Exp3 双重证伪 + Exp5 三重证伪,Exp7 沿用 Exp6 决议(`EXP6_PROPOSAL_v8.md §附录 B 第 7 条`)不引入 |
| Diffusion sampling (Exp5 系列) | GAN one-shot generation,无需迭代去噪 |

---

## 3. Exp5' / Exp6 复用清单 (v5 — 路径全部实测锁死, SOP 4 严格通过)

> **v5 关键更新**: 基于用户 2026-05-10 服务器 tree 实测,所有路径**绝对路径锁死**。Exp6 已跑完所有 step1 calibration,Exp7 SA1 跳过 step1.0a/1.1/1.2/1.3 主套 (实测节省 1 天)

### 3.1 完全复用 — 直接 cp / symlink (零改动)

#### 3.1.1 来自 Exp5' 的资产

| 资产 | **实测绝对路径** | 操作 | 备注 |
|---|---|---|---|
| Dataset class | `/home/tcat/diffcsp_exp5_prime/code/step3/xas_local_dataset_v2.py` | **cp** 到 `experiment7/shared/` | 注意文件名是 `_v2.py` 不是 `_v3.py` (v1-v4 写错)。Exp5'-STEP1-FIX-C 阶段就地改造支持 L=20,文件名未变 |
| Datamodule | `/home/tcat/diffcsp_exp5_prime/code/step3/xas_local_datamodule_v2.py` | **cp** 到 `experiment7/shared/` | 同上,文件名是 v2 |
| L=20 cache (train) | `/home/tcat/diffcsp_exp5_prime/data/train_structure_cache.pt` (44 MB, May 3 build) | **symlink** 到 `experiment7/data/` | 60501 sample |
| L=20 cache (val) | `/home/tcat/diffcsp_exp5_prime/data/val_structure_cache.pt` (5.6 MB) | **symlink** | 7621 sample |
| L=20 cache (test) | `/home/tcat/diffcsp_exp5_prime/data/test_structure_cache.pt` (3.3 MB) | **symlink** | 4481 sample |
| **L=20 cache (holdout)** | ⚠️ **不存在** | **必跑** `precompute_structure_cache_exp5_prime.py` 单独构建 | 见 §3.1.4 警告 |
| L=20 cache metadata | `/home/tcat/diffcsp_exp5_prime/data/cache_metadata.json` (`{"L_VIRTUAL": 20.0}`) | cp + 强 verify | |
| Spectrum 数据 | `/home/tcat/diffcsp_exp5_prime/data/spectra_{train,val,test,holdout}.pkl` | symlink | 不变 |
| FEFF features | `/home/tcat/diffcsp_exp5_prime/data/feff_features_imputed.pkl` + `feff_feature_scaler.pkl` | symlink | 不变 |
| Train/val/test split CSV | `/home/tcat/diffcsp_exp5_prime/data/{train,val,test,holdout}_samples_v2.csv` | symlink | 不变 |
| Spectrum encoder | `/home/tcat/diffcsp_exp5_prime/code/step2/spectrum_encoder.py` | **cp** + diff Exp6 版本 | MV-attention 4 heads + center embedding 95×16d。⚠️ Exp6 同等实现叫 `spectrum_tokenizer.py`,SA1 必须 diff 验证两者一致性,选其中之一作 Exp7 的 base |
| Step5_3 副套评估 | `/home/tcat/diffcsp_exp5_prime/code/step5/step5_3_composite_score.py` **或** `step5_3_composite_score_exp5_prime.py` | **diff 后选一,cp** | ⚠️ 两个文件同时存在,见 §3.1.5 警告 |
| `precompute_structure_cache_exp5_prime.py` | `/home/tcat/diffcsp_exp5_prime/code/step3/` | cp | 用于构建 holdout cache |

#### 3.1.2 来自 Exp6 的资产 (v5 新发现 — Exp6 已完成 step1,Exp7 直接拿来用)

| 资产 | **实测绝对路径** | 操作 | 备注 |
|---|---|---|---|
| **MIN_PDIST calibration** | `/home/tcat/experiment6_v7/shared/min_pdist_calibration.json` | **cp** 到 `experiment7/shared/` | **`MIN_PDIST = 1.5075718402862548 Å`**,frozen,Exp7 直接复用,SA1 不许重新 calibrate |
| **Shell integrity report** | `/home/tcat/experiment6_v7/shared/shell_integrity_report.json` | **cp** | Exp6 step1.0a 产出,shell_boundaries.pkl 完整性已 verify |
| **Element vocab** | `/home/tcat/experiment6_v7/shared/exp6_element_vocab.json` | **cp** 改名 `exp7_element_vocab.json` | 88 元素 center + neighbor 双 vocab,Exp7 沿用 |
| **CPS baseline 数字** | `/home/tcat/experiment6_v7/shared/baseline_cps.json` | **cp + 加 Exp7 字段** | Exp4/Exp5' CPS 已实测,SA1 step1.3 只需追加 Exp7 训完后的数字 |
| **CPS 主套实现** ⭐ | `/home/tcat/experiment6_v7/shared/composite_score.py` | **cp** 到 `experiment7/shared/eval_cps.py` | v4 让 SA1 重写 ~250 行错!Exp6 已实现,SA1 直接 cp + 改 import path |
| RDF 直方图 (供 sanity) | `/home/tcat/experiment6_v7/shared/min_pdist_rdf_hist.png` | cp | 人工 review 用 |
| Shell n_atoms 直方图 | `/home/tcat/experiment6_v7/shared/shell_n_atoms_hist.png` | cp | 人工 review 用 |

#### 3.1.3 来自 Exp4 的基础数据

| 资产 | **实测绝对路径** | md5 | 操作 |
|---|---|---|---|
| `shell_boundaries.pkl` | `/home/tcat/diffcsp_exp4/data/shell_boundaries.pkl` | `cf2050e4899160f5698ad2481377e94c` ✅ verify pass | symlink (387 MB,不 cp 节省磁盘) |
| Exp4 best ckpt (历史 baseline) | `/home/tcat/diffcsp_exp4/checkpoints/best-epoch366-val0.7300.ckpt` | (SA1 启动时 verify) | 只读引用,不操作 |

#### 3.1.4 ⚠️ 严重警告: Holdout cache 不存在

实测 `/home/tcat/diffcsp_exp5_prime/data/` 包含 train/val/test 三个 `_structure_cache.pt`,**但没有 `holdout_structure_cache.pt`**:

```
$ ls /home/tcat/diffcsp_exp5_prime/data/*structure_cache*.pt
train_structure_cache.pt  test_structure_cache.pt  val_structure_cache.pt
# ← 没有 holdout_structure_cache.pt
```

**含义**: Exp5' 训练 + val/test sample 阶段未构建 holdout cache (Exp5' 三 split sample 中 holdout 3025 用了别的途径计算)。

**Exp7 应对**:
- Phase 1 训练 sample val/test 时,**不需要** holdout cache
- **Phase 4 sample holdout 前**,SA1 必须用 `cp /home/tcat/diffcsp_exp5_prime/code/step3/precompute_structure_cache_exp5_prime.py` 进 `experiment7/step1/` 然后跑产出 `holdout_structure_cache.pt`,估计 5-10 分钟
- **禁止**: SA1 在 step1 阶段就 build holdout cache (有 user round 决议 "holdout 永久封存",但 build cache 不算违反 "封存",sample 才算)。**v5 SOP**: holdout cache build 必须发生在 Phase 4 开始前,不许提前

#### 3.1.5 ⚠️ 严重警告: step5_3 两个版本并存

实测 Exp5' 目录有两个几乎同名的 step5_3 实现:

```
/home/tcat/diffcsp_exp5_prime/code/step5/step5_3_composite_score.py            ← 原版 (创建时间需 verify)
/home/tcat/diffcsp_exp5_prime/code/step5/step5_3_composite_score_exp5_prime.py ← 修订版 (含 "_exp5_prime" 后缀,推测是 Exp5' MA 修订)
```

**SA1 Day 0 必做**: diff 两个文件,确认:
1. 哪个是 Exp5'-MA final report §3.2 verdict `composite_val 0.0801` 实际使用的版本
2. 两者差异点 (如果有的话)
3. 看 Exp5'-MA hand-back 文档 `EXP5_PRIME_STEP3_SAMPLE_HANDBACK_v2.md` 确认

**默认假设** (待 SA1 verify): `_exp5_prime` 后缀版是修订版,与 Exp5' final report 数字一致,Exp7 应该用这个。如果 SA1 verify 后发现是原版,**raise 给用户决议**,不许擅自选。

#### 3.1.6 ⚠️ 警告: Spectrum encoder 命名分歧 (Exp5' vs Exp6)

```
/home/tcat/diffcsp_exp5_prime/code/step2/spectrum_encoder.py   ← Exp5' 命名
/home/tcat/experiment6_v7/shared/spectrum_tokenizer.py          ← Exp6 命名
```

**两个文件应该是同一实现** (MV-attention + center embedding 95×16d,Exp5'/Exp6 都沿用 Exp5 v2 设计),但**文件名不同会迷惑 SA1**。

**SA1 Day 0 必做**:
1. `diff spectrum_encoder.py spectrum_tokenizer.py` 验证是否同一实现
2. 如果有差异,看 Exp6 v8 §3.2 说明哪些是 Exp6 改动 (附录 B 第 5 条)
3. 选其中之一 cp 到 `experiment7/shared/`,**命名**: Exp7 使用 `spectrum_encoder.py` (沿用 Exp5' 命名,因为 Exp7 thesis 上 spectrum 作 Generator condition 是 "encoder" 语义,不是 "tokenizer")

### 3.2 部分复用 (改造)

| 资产 | **实测路径** | Exp7 改造 |
|---|---|---|
| Pairwise min distance penalty | Exp5' 在 `diffusion_w_type_xas.py` 内部实现 (路径: `/home/tcat/diffcsp_exp5_prime/code/step3/diffusion_w_type_xas.py`,搜索 `_pairwise_min_distance_penalty`) | **cp 公式 + 改造**: 输入 shape 从 `(n_atoms_total, 3)` diffusion-flat 改 `(B, 20, 3)` GAN-batched + 加 v2 curriculum schedule (§6.2.4) |

### 3.3 完全废弃 (不沿用)

- ❌ CSPNet decoder (`diffusion_w_type_xas.py` 主体扩散逻辑)
- ❌ DETR transformer encoder-decoder (`/home/tcat/experiment6_v7/shared/detr_xas.py`)
- ❌ Diffusion forward/reverse (整个 `diffusion_w_type_xas.py` 中的 noise schedule + reverse 部分)
- ❌ Hungarian matcher (`/home/tcat/experiment6_v7/shared/matcher.py`) — GAN one-shot 生成不需要
- ❌ Shell distance / count loss (Exp5'/Exp5'' 三件套之 2/3) — Exp7 v3 §2.1 已说明
- ❌ Exp5 `step5_2_compute_metrics.py` — errata 3 R_max=5.5 Å fallback bug 锚点

---


## 4. GitHub 起点 (v3 clone-first 策略, sote 与 Exp6 v8 一致)

**设计原则**(Exp6 v8 § 3 同款): 优先 clone 已验证 PyTorch 实现,**禁止从 TF 仓库手写 PyTorch 版**。SA1 工作流应是"clone + 改造",不是"参考 + 手写"。

### 4.1 主 clone — `eriklindernoren/PyTorch-GAN`

**地址**: https://github.com/eriklindernoren/PyTorch-GAN
**Stars**: 16k+ (主流 PyTorch GAN 实现集合)
**License**: MIT (无商业限制)
**关键文件 (SA1 直接 clone 后从中拷贝两个)**:

| 源文件 | 来源 | 文件行数 | SA1 用途 |
|---|---|---|---|
| `implementations/cgan/cgan.py` | Mirza & Osindero 2014 cGAN | ~ 200 行 | Conditional GAN 基础 (Generator condition concat + Discriminator condition concat) |
| `implementations/wgan_gp/wgan_gp.py` | Gulrajani et al. 2017 WGAN-GP | ~ 200 行 | Wasserstein loss + gradient penalty + n_critic=5 训练循环 |

**SA1 任务**: 把这两个文件合并为 `cond_wgan_gp.py` (cGAN 提供条件 architecture,WGAN-GP 提供训练范式),再做 XAS 适配。

### 4.2 合并参考 — `gcucurull/cond-wgan-gp`

**地址**: https://github.com/gcucurull/cond-wgan-gp
**用途**: cGAN + WGAN-GP **已合并的 PyTorch 实现** — SA1 在合并 §4.1 两个文件时 cross-check 这个仓库的合并方式
**警告**: 这个仓库是 image-based (MNIST),condition 是 class label。Exp7 的 condition 是 256d continuous spectrum vector,不是 label。**仅参考"如何合并 cGAN + WGAN-GP",不照搬 model architecture**

### 4.3 Spectral Normalization 模块

**仓库**: https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
**关键文件**: `spectral_normalization.py` (~ 50 行,SA1 直接 cp 进 `experiment7/shared/`)
**用途**: Discriminator 每层加 `SpectralNorm(nn.Linear(...))` wrap,稳定 D 训练
**论文来源**: Miyato et al. 2018 ICLR (arxiv 1802.05957)

### 4.4 GAN 关键算法论文(SA1 必读)

只读不改 implementation:

| 概念 | 来源论文 |
|---|---|
| **WGAN-GP** (主算法) | Gulrajani et al. 2017 (NeurIPS) "Improved Training of Wasserstein GANs",arxiv 1704.00028 |
| **Spectral normalization** | Miyato et al. 2018 (ICLR),arxiv 1802.05957 |
| **TTUR** (D 快 G 慢) | Heusel et al. 2017 (NeurIPS),arxiv 1706.08500 |
| **Conditional GAN** | Mirza & Osindero 2014,arxiv 1411.1784 |
| **Inverse design 类比** (one-to-many) | "Inverse design of structural color: finding multiple solutions via cGAN", PMC11501759 — 不是 implementation,是任务范式参考 |

### 4.5 真实工作量重新评估 (v3)

| 部件 | 来源 | 行数 | 标注 |
|---|---|---|---|
| `cgan.py` | clone from `eriklindernoren/PyTorch-GAN` | 200 行 | **直接拷,零改动 (sote)** |
| `wgan_gp.py` | clone from `eriklindernoren/PyTorch-GAN` | 200 行 | **直接拷,零改动 (sote)** |
| `spectral_normalization.py` | clone from `christiancosgrove/pytorch-spectral-normalization-gan` | 50 行 | **直接拷,零改动** |
| `cond_wgan_gp.py` (SA1 合并两个) | SA1 写,cross-check `gcucurull/cond-wgan-gp` | ~ 150 行 | **modify**: 主要是合并 cGAN condition 进 WGAN-GP 训练循环 |
| Generator architecture | SA1 改造 cgan.py 的 image generator | ~ 80 行 modify | **modify**: 替换 ConvTranspose2d decoder 为 MLP for (20, 3+K+1) |
| Discriminator architecture | SA1 改造 wgan_gp.py 的 image discriminator | ~ 80 行 modify | **modify**: 替换 Conv2d image input 为 distance matrix + atom types 1D CNN |
| SpectrumEncoder integration | 从 Exp5'/Exp6 sote cp | ~ 100 行 | **partial reuse**: cp Exp5' encoder, generator 接入 |
| Pairwise hinge loss (curriculum-aware) | 从 Exp5' sote cp + curriculum adapt | ~ 50 行 | **modify**: 加 v2 §6.2.4 curriculum schedule |
| Type CE loss | 与 Exp6 v8 同款 | ~ 30 行 | **partial reuse** |
| Curriculum callbacks (ckpt + EarlyStop filter) | SA1 新写,基于 PyTorch Lightning | ~ 100 行 | **new** |
| 评估 glue (调用 step5_3) | Exp5' sote 直接 import | ~ 50 行 | **partial reuse** |

**总工作量** (v3 vs v2 对比):
- v2 估计: ~ 700 行**新写**
- **v3 估计**: ~ 500 行 clone (零改) + ~ 250 行 modify + ~ 100 行 new = ~ 850 行 total,但**真正手写只 ~ 100 行**
- 对照 Exp6 v8: ~ 4 文件 clone DETR + ~ 80 行 modify + ~ 200 行 new — 工作量级别匹配

**v3 关键的设计差别**: SA1 不再"从论文 + TF 反推手写",而是"clone + 改造 + cross-check"。Bug surface 减少 90% (因为 WGAN-GP 的 gradient penalty 实现、interpolation alpha broadcast、TTUR optimizer 配置这些细节都在 cloned PyTorch 代码里已经被千万人验证过)。


---

## 5. Exp7 整体架构

```
                    ┌─────────────────────────────────────────────┐
                    │  Spectrum (xmu 150 + chi1 200 + feff 73)   │
                    │            +                                │
                    │  center_element_Z (一标量)                  │
                    │            +                                │
                    │  z ~ N(0, I) (noise, 128d)                  │  ← GAN 特有, generator 多样性源头
                    └────────────────────┬────────────────────────┘
                                         │
                              Generator G
                              ────────────
                    ┌─────────────────────┴───────────────────────┐
                    │  SpectrumEncoder (MV-attn 4 heads)          │  ← Exp5'/Exp6 沿用
                    │    + Center embedding (95×16d)              │
                    │    → 256d spectrum_cond                     │
                    │                                              │
                    │  spectrum_cond + z → MLP decoder            │  ← Exp7 新设计
                    │    输出: (20, 3) frac_coords                 │
                    │           (20, K+1) type_logits             │
                    └─────────────────────┬───────────────────────┘
                                         │ generated structure
                                         │
                                         ▼
              ┌──────────────────────────────────────────────────────┐
              │                                                       │
              │  Discriminator D ⭐ (Exp7 核心创新)                   │
              │  ───────────────────                                  │
              │                                                       │
              │  Input: (spectrum_cond, structure)                   │
              │    where structure = (20×20 pairwise dist matrix,    │
              │                       20 atom_types softmax,         │
              │                       20 dist-to-center)             │
              │                                                       │
              │  ⚠️ 不直接用 raw frac_coords (排列不敏感性 + GAN 旋转  │
              │  不变性问题)。改用 pairwise distance matrix:         │
              │  D 输入是结构的 invariant 表示,自动学到 shell        │
              │                                                       │
              │  Architecture:                                       │
              │    1. distance matrix → 1D CNN (kernel 3) → 128d    │
              │    2. atom_types → small MLP → 64d                  │
              │    3. spectrum_cond → projection → 256d             │
              │    4. concat + MLP + spectral norm → scalar critic  │
              │                                                       │
              │  Output: Wasserstein critic score (scalar)           │
              │                                                       │
              │  v1 关键: D 输入只看 distance matrix,**自动学到**     │
              │  "正确的 distance 分布",包含 shell 边界。             │
              │  这是绕开 Exp5' 鸡蛋问题 + Exp5'' 几何冲突的核心。    │
              └──────────────────────────────────────────────────────┘
                                         │
                                         ▼
                            ┌────────────────────────┐
                            │  WGAN-GP Loss          │
                            │   + Gradient Penalty   │
                            │   + Pairwise hinge     │  ← Exp5' 沿用,gate 64% 硬证
                            │   + Type CE            │  ← element 监督
                            └────────────────────────┘
```

### 5.1 关键架构决策辨析

**(a) Discriminator input 选 distance matrix 而非 raw coords**

理由(基于 Exp5 系列教训):
1. **Rotation invariance** — Raw frac coords 受旋转影响,GAN 训练时 generator 学到某个特定取向但实际物理上等价旋转都该被接受。Distance matrix 是 SE(3) 不变的
2. **Permutation handling** — 20 个邻居没有自然顺序,distance matrix sorted (e.g. distances from center sorted ascending) 提供 canonical 排序
3. **Shell 自然涌现** — 真实结构的 distance matrix 会有 shell 间的 gap (Exp4 step 2.5 gap=0.1563 已 verify),discriminator 学习真实 distance 分布等价于隐式学 shell 概念
4. **避开 Exp5'' 几何冲突** — Distance matrix-based D 不依赖任何 explicit shell label,因此不会与 pairwise constraint 几何冲突

**(b) Generator 输出 frac_coords 不是 distance — 因为评估端用 frac_coords**

- Generator 内部输出 frac_coords (与 Exp4/Exp5/Exp6 一致)
- 但在送入 D 之前,先 compute distance matrix
- 这样 step5_3 评估端能直接复用 (评估输入是 frac_coords + types)

**(c) Spectral normalization 在 D 不在 G**

- SN 限制 Lipschitz 常数,稳定 D 训练
- G 不加 SN,允许 expressive 输出

**(d) 单 noise z 不是 per-atom noise**

- Per-sample 一个 128d z,所有 20 个邻居从同一 z 解码 (MLP 一次输出 20×3)
- Per-atom z 会让 generator 学到 "原子间独立分布",反 shell 物理

### 5.5 数据处理 verify 框架 (v6 新增, SA1 Day 0 必做)

**设计动机**: v1-v5 默认沿用 Exp5'/Exp6 dataset 但**没有验证 GAN 特有需求是否被覆盖**。Exp5' (diffusion) 和 Exp6 (transformer with Hungarian) 都不依赖某些数据 contract,但 Exp7 GAN one-shot generation 依赖。

**v6 决定**: 不在 proposal 阶段锁定 dataset 改动方式,而是**给 SA1 一个结构化 verify + 决策框架**,让 SA1 Day 0 读 Exp5' dataset 代码后基于 verify 结果决策。

#### 5.5.1 SA1 Day 0 必 verify 的 5 个数据 contract

SA1 必须读 `/home/tcat/diffcsp_exp5_prime/code/step3/xas_local_dataset_v2.py` 代码 + 实际跑 5 个 sample 看 `__getitem__` 输出,verify 以下 5 项,产出 `experiment7/data/dataset_contract_audit.json`:

| # | Verify 项 | 期望行为 | 影响 |
|---|---|---|---|
| **V1** | `atom_types` 是否按距中心 cart distance **升序排序** | YES 或 NO (二元判定) | 决定 §5.6 排序方案 |
| **V2** | n_neighbors < 20 sample 的 padding 方式 | one of: (a) 填 NO_OBJECT_IDX, (b) attention mask 标记, (c) 填中心位置 0,0,0, (d) 其他 | 决定 type CE `ignore_index` 怎么设 |
| **V3** | `frac_coords` 范围 | [-0.5, 0.5] 或 [0, 1] | 决定 distance matrix 计算 min-image fold 公式 |
| **V4** | `center_element_Z` 是否单独输出 | YES (作为 condition) / NO (需 collate 时从 sample_name lookup) | 决定 generator condition 取值方式 |
| **V5** | `lengths` 是 per-sample tensor (3,) 还是默认 L_VIRTUAL=20.0 全局 | per-sample / global | 决定 distance matrix 算 cart 时怎么 broadcast |

**Verify 方法 (SA1 必做)**:

```python
# 读 dataset,跑 5 个 sample
from xas_local_dataset_v2 import XASLocalDataset
ds = XASLocalDataset(split='train', L_VIRTUAL=20.0, ...)
sample = ds[0]
print("Output keys:", sample.keys())
print("frac_coords:", sample['frac_coords'].shape, "min/max:",
      sample['frac_coords'].min(), sample['frac_coords'].max())
print("atom_types:", sample.get('atom_types'))

# V1: check sorted
import torch
fc = sample['frac_coords']
lengths = torch.tensor([20.0, 20.0, 20.0])
dists = (fc * lengths).norm(dim=-1)  # cart Å
print("dists:", dists)
print("dists sorted?", torch.all(dists[:-1] <= dists[1:]).item())

# V2: padding (看连续 NO_OBJECT_IDX / 0 出现的位置)
# 跑 10 个 sample,看 n_atoms_real vs padding

# V3: 范围
# V4: center_element_Z 是不是单独 key
# V5: lengths 是不是 sample-level
```

**Verify 失败 (任一项)**: SA1 raise 给用户决议,**禁止**擅自适配 (改 dataset 是 SOP 12 红线项)。

#### 5.5.2 Distance matrix 计算位置 — 基于 V1-V5 verify 后的决策矩阵

SA1 基于 V1-V5 verify 结果,**严格按下表决策**:

| Verify 结果 | Distance matrix 计算位置 | 实施方式 |
|---|---|---|
| V3=[-0.5,0.5] **且** V5=per-sample lengths **且** V1=sorted | **Option A** (training loop on-the-fly) | training_step 内调用 `compute_distance_matrix(frac, lengths)`,见 §5.5.3 公式锁定 |
| V3=[0,1] **或** V5=global | **Option A**,但 compute 函数需相应 broadcast 调整 | 同上 |
| V1=unsorted | **Option A** + **collate_fn 重排** (atom_types 按 dist sort) | §5.6 collate_fn 公式锁定 |
| 任一 V verify 出乎预期 (如 dataset 给的 frac_coords 范围既不是 [-0.5,0.5] 也不是 [0,1]) | **RAISE 给用户决议** | 不许 SA1 自己 patch |

**SA1 不许选 Option B (改 dataset)** — 这违反 §3.1 "Exp5'/Exp6 dataset 1:1 复用" 原则。Option B 仅在 Option A 完全不可行时 (e.g. Exp5' dataset 内有某种隐藏 stateful 行为) 才考虑,且必须先 raise 用户。

#### 5.5.3 Distance matrix 计算公式锁定 (SOP 1)

```python
def compute_distance_matrix(frac_coords, lengths):
    """
    从 (B, 20, 3) frac coordinates 算 (B, 20, 20) min-image cart pairwise distance.
    与 Exp5' _pairwise_min_distance_penalty 用同一 min-image fold 算法。
    
    Args:
        frac_coords: (B, 20, 3) tensor, frac ∈ [-0.5, 0.5]
                     (若 dataset V3 verify 出 [0, 1] 范围, SA1 在调用前先 - 0.5 normalize)
        lengths: (B, 3) per-sample lengths OR (3,) global lengths
                 (若 V5 verify 出 global, SA1 broadcast 到 (B, 3))
    
    Returns: (B, 20, 20) cart distance Å
    """
    # min-image fold
    diff = frac_coords[:, :, None] - frac_coords[:, None, :]  # (B, 20, 20, 3) frac
    diff = diff - torch.round(diff)                           # min-image
    
    # cart
    if lengths.dim() == 1:  # global (3,)
        lengths = lengths.view(1, 1, 1, 3).expand(diff.shape[0], 20, 20, 3)
    elif lengths.dim() == 2:  # per-sample (B, 3)
        lengths = lengths.view(-1, 1, 1, 3).expand(-1, 20, 20, 3)
    
    cart = diff * lengths  # (B, 20, 20, 3) Å
    dist = torch.norm(cart, dim=-1)  # (B, 20, 20)
    
    return dist
```

**SA1 不许改**: min-image fold 算法 (与 Exp5' pairwise loss 同算法) / cart 转换方式 / dim handling。这是 SOP 1 锁定。

#### 5.5.4 决策结果记录 (强制流程闸门)

SA1 完成 verify 后必须写 `experiment7/data/dataset_contract_audit.json`:

```json
{
  "audit_date": "2026-05-XX",
  "dataset_source": "/home/tcat/diffcsp_exp5_prime/code/step3/xas_local_dataset_v2.py",
  "dataset_md5": "...",  // SA1 cp 后算
  "V1_atom_types_sorted": true/false,
  "V2_padding_method": "NO_OBJECT_IDX" / "mask" / "zero" / "other",
  "V3_frac_range": "[-0.5, 0.5]" / "[0, 1]" / "other",
  "V4_center_element_separate": true/false,
  "V5_lengths_scope": "per_sample" / "global",
  "decision": {
    "distance_matrix_compute_location": "training_loop_on_the_fly",  // 只允许此值,除非用户决议
    "atom_types_resort_needed": true/false,  // 取决于 V1
    "resort_implementation": "collate_fn" / "training_step" / "not_needed",
  }
}
```

**没产出此 json 不许进 Phase 2 训练** (流程闸门)。

### 5.6 Atom types 排序方案 (v6 新增,公式锁定,SA1 不许改)

#### 5.6.1 设计原理

Exp7 GAN 用 type CE 监督 generator 输出的 element types,而 type CE **必须有一个明确的 GT-prediction 1-to-1 对应**(不像 DETR Hungarian 自动匹配)。

要让 1-to-1 对应物理上合理,选 "按到中心距离升序" 作为 canonical ordering:
- Generator 输出第 i 位 → GT 第 i 个最近的邻居
- 这强制 generator 学到 "shell 内层 atoms 先输出"
- 与 distance matrix discriminator input 的 ordering 自然一致 (D 也用 sorted)

#### 5.6.2 实施 — 按 V1 verify 结果分流

```python
# 情况 A: V1=sorted (Exp5'/Exp6 dataset 已 sort)
# → 直接用,collate_fn 不动

# 情况 B: V1=unsorted (Exp5'/Exp6 dataset 给 arbitrary order)
# → SA1 写 collate_fn,batch 内重排
def collate_fn_resort(batch):
    """Custom collate that re-sorts atoms by distance to center."""
    out = default_collate(batch)
    frac = out['frac_coords']     # (B, 20, 3)
    types = out['atom_types']     # (B, 20)
    lengths = out['lengths']
    
    # Per-sample sort
    dists = (frac * lengths.view(-1, 1, 3)).norm(dim=-1)  # (B, 20) cart
    sorted_idx = dists.argsort(dim=-1)                     # (B, 20)
    
    # Gather
    sorted_frac = torch.gather(frac, 1, sorted_idx.unsqueeze(-1).expand(-1, -1, 3))
    sorted_types = torch.gather(types, 1, sorted_idx)
    
    out['frac_coords'] = sorted_frac
    out['atom_types'] = sorted_types
    return out
```

#### 5.6.3 验证 — collate_fn 重排正确性

SA1 必须在 step1.4 smoke test 中加 sanity check:

```python
# 用 collate_fn 取 5 个 sample,verify:
# 1. 排序后 atom 数量 / 元素总数 不变
# 2. 排序后 dists 单调递增
# 3. 排序后 frac_coords 与 types 1-to-1 对应 (不能错位)
for batch in [sample_batch for _ in range(5)]:
    fc = batch['frac_coords']
    ts = batch['atom_types']
    lengths = batch['lengths']
    dists = (fc * lengths.view(-1, 1, 3)).norm(dim=-1)
    # 单调
    assert torch.all(dists[:, :-1] <= dists[:, 1:]), "Sort failed!"
    # 元素总数不变
    # (比较 sorted batch 与 unsorted batch 各元素 Counter)
```

#### 5.6.4 Padding 与 sort 的交互

V2 padding 方式必须与 sort 方式兼容:
- 如果 padding 用 `NO_OBJECT_IDX` (e.g. dataset 给 (atom_types[:n_real], [NO_OBJECT_IDX] * (20-n_real)),那 padding 原子的 dist 必须**视为无穷大** (排到最后),sort 才合理
- SA1 在 sort 前必须把 padding 位置的 dist 设为 `float('inf')` 让它们排到末尾,避免 padding 原子和真实原子在 sort 后混淆

```python
# 处理 padding (在 sort 之前):
n_real = (types != NO_OBJECT_IDX).sum(dim=-1, keepdim=True)  # (B, 1)
position_idx = torch.arange(20, device=types.device).unsqueeze(0).expand_as(types)
is_padding = (position_idx >= n_real)  # (B, 20) bool
dists = dists.masked_fill(is_padding, float('inf'))  # padding 排末尾
```

**SA1 不许改 sort 算法,但可以根据 V2 verify 结果调整 padding 检测方式**。

---

## 6. Loss 设计

### 6.1 总 loss 公式

```python
# Generator G loss (per training step)
loss_G = -E[D(G(z, spectrum, center))]                # WGAN adversarial: maximize critic
       + lambda_pmin * loss_pairwise_min_G            # 沿用 Exp5' 三件套之 1
       + lambda_type * loss_type_ce_G                 # element classification

# Discriminator D loss (per training step, x_real and x_fake)
loss_D = E[D(x_fake)] - E[D(x_real)]                  # WGAN critic loss
       + lambda_gp * loss_gradient_penalty             # WGAN-GP 1-Lipschitz constraint
```

### 6.2 三个 loss 项公式锁定 (SA1 不许改)

#### 6.2.1 `loss_pairwise_min_G` — Exp5' 沿用 + v2 curriculum 接口

直接从 Exp5' `_pairwise_min_distance_penalty` 拷贝,公式不变。**v2 改动: `min_pdist` 不再硬编码,改为从 curriculum schedule 取**。

```python
def compute_pairwise_min_penalty(pred_pos, lengths, min_pdist):
    """
    Exp5' 沿用,公式锁定。Hinge form: ReLU(min_pdist - d_pair)² mean
    
    v2 修订: min_pdist 不再 default = MIN_PDIST (calibration value),
    改为运行时由 LightningModule.on_train_epoch_start 通过 curriculum schedule 注入
    
    输入: pred_pos shape (B, 20, 3) frac coord
          min_pdist: float, 由 curriculum schedule 决定 (v2 §6.2.4)
    输出: scalar tensor
    """
    B, N, _ = pred_pos.shape
    diff = pred_pos[:, :, None] - pred_pos[:, None, :]    # (B, 20, 20, 3) frac
    diff = diff - torch.round(diff)                        # min-image fold
    cart = diff * lengths                                  # (B, 20, 20, 3) Å
    pdist = torch.norm(cart, dim=-1)                       # (B, 20, 20) Å
    
    eye = torch.eye(N, device=pdist.device, dtype=torch.bool).unsqueeze(0).expand(B, -1, -1)
    pdist_off = pdist.masked_fill(eye, float('inf'))
    
    violation = torch.clamp(min_pdist - pdist_off, min=0.0) ** 2
    return violation.mean()
```

**关键沿用 (v1)**:
- `MIN_PDIST` 最终值 (curriculum 末段) 从 Exp5'/Exp6 `min_pdist_calibration.json` load
- 若 Exp6 已 calibrate,Exp7 直接复用同一文件;若未 calibrate,Exp7 自跑 RDF analysis

**v2 关键新增**: 运行时 `min_pdist` 由 curriculum 决定,见 §6.2.4。

#### 6.2.2 `loss_type_ce_G` — element classification

```python
def compute_type_ce_loss(pred_type_logits, true_types, valid_mask):
    """
    pred_type_logits: (B, 20, K+1), K = N_NEIGHBOR_TYPES, +1 for "no_object"
    true_types: (B, 20), padding indices are NO_OBJECT_IDX
    valid_mask: (B, 20) bool, true for valid (non-padding) positions
    
    与 Exp6 v8 cls loss 实现一致,使用 cross entropy
    """
    # 用 ground truth 排序 (距中心 ascending) 后,与 pred 同序 1-to-1
    # GAN 不用 Hungarian (one-shot generation 直接索引对齐)
    return F.cross_entropy(
        pred_type_logits.permute(0, 2, 1),       # (B, K+1, 20)
        true_types,                                # (B, 20)
        ignore_index=NO_OBJECT_IDX,                # padding 不算
        reduction='mean'
    )
```

**关键设计**:
- GT 端 atom_types 必须**按到中心距离升序排序** (与 Exp5'/Exp4 dataset_v3 一致),pred 端也按生成顺序对齐
- 不用 Hungarian (Exp6 v8 用 Hungarian 因为 query 无序;Exp7 generator 输出按 distance 升序的隐式 ordering)
- **v1 待 verify**: SA1 在 smoke test 阶段必须 confirm dataset_v3 atom_types 是按 distance 升序的

#### 6.2.3 `loss_gradient_penalty` — WGAN-GP 核心 (SA1 直接 implement)

```python
def compute_gradient_penalty(D, real_struct, fake_struct, spectrum_cond, device):
    """
    WGAN-GP gradient penalty: enforces 1-Lipschitz on D.
    Gulrajani et al. 2017 (arxiv 1704.00028) §3.
    
    Args:
        D: Discriminator module
        real_struct: (B, distance_matrix + atom_types + dist_to_center) — real
        fake_struct: 同结构,from generator
        spectrum_cond: (B, 256) condition (用于 cGAN-GP)
        device: cuda
    
    Returns: scalar tensor
    """
    B = real_struct['dist_matrix'].shape[0]
    
    # Interpolation alpha
    alpha = torch.rand(B, 1, 1, device=device)
    
    # 注意: 由于 struct 是 dict,每个 component 都要 interpolate
    interpolated = {}
    for key in real_struct:
        a = alpha
        # broadcast alpha to component shape
        while a.dim() < real_struct[key].dim():
            a = a.unsqueeze(-1)
        interpolated[key] = a * real_struct[key] + (1 - a) * fake_struct[key]
        interpolated[key].requires_grad_(True)
    
    # D forward on interpolated
    d_interp = D(interpolated, spectrum_cond)
    
    # Gradient w.r.t. each input component
    grads = torch.autograd.grad(
        outputs=d_interp.sum(),
        inputs=list(interpolated.values()),
        create_graph=True,
        retain_graph=True,
    )
    
    # Concatenate gradients, compute 2-norm per sample, penalty (||grad|| - 1)^2
    grads_flat = torch.cat([g.flatten(start_dim=1) for g in grads], dim=1)  # (B, total_dim)
    grad_norm = grads_flat.norm(2, dim=1)
    penalty = ((grad_norm - 1.0) ** 2).mean()
    
    return penalty
```

**关键超参 (SA1 必查 Gulrajani 论文 §4 表 1)**:
- `lambda_gp = 10.0` (WGAN-GP 论文标准值)
- 不许擅自调

#### 6.2.4 Curriculum learning schedule (v2 新增,SA1 不许改)

**设计动机**(Exp5 系列教训反推):

Exp5 系列 EarlyStop 频繁触发 (Exp5'/Exp5'' 均在 patience 用满前触发 plateau)。根因分析:
- Random init 时 G 输出原子位置近随机 → pairwise loss 极大
- G 被迫先学 "推开原子",但推过头 → shell 结构散乱
- Shell-aware loss (Exp5'/'' 三件套) 又需要 shell 结构作 trigger → 鸡蛋问题 (errata 5)
- 双目标在 random init 下互斥 → model 在某个 plateau 卡住 → EarlyStop

**v2 解决方案**: Curriculum learning,**让 model 先在弱 pairwise 约束下学 shell 结构,再逐渐加约束**。最终 schedule 末段 (epoch ≥ 150) 等价于完整物理约束,与 v1 一致;但中间过渡让 D 有机会先学到 real distance distribution,Generator 有机会先学到合理 shell。

**公式锁定**:

```python
# 模块级常量 (SA1 在 criterion.py 顶部声明)

CURRICULUM_EPOCH_BOUNDARIES = [50, 100, 150]   # epoch 切换点 (hand-picked, training schedule)
CURRICULUM_FRACTIONS = [0.33, 0.53, 0.73, 1.00]  # min_pdist as fraction of calibration

# 注: fraction 数值是从用户 round 1 提案 (0.5, 0.8, 1.1, 1.5076) 反推得到
#     0.5 / 1.5076 ≈ 33%, 0.8 / 1.5076 ≈ 53%, 1.1 / 1.5076 ≈ 73%
#     v2 用 fraction-of-calibration 形式而非 hand-picked 数字,
#     使 schedule data-driven: calibration 出什么 MIN_PDIST 自动 scale

def get_curriculum_min_pdist(epoch, calibrated_min_pdist):
    """
    Curriculum schedule for min_pdist hinge constraint.
    
    Args:
        epoch: current epoch (int)
        calibrated_min_pdist: from step1.2 RDF calibration (float, typically 1.4-1.6 Å)
    
    Returns: float, the min_pdist to use in pairwise_min_penalty this epoch
    
    Schedule:
    - Phase 0 (epoch 0-49):   33% of calibrated  (~0.5 Å if cal=1.5076)
    - Phase 1 (epoch 50-99):  53% of calibrated  (~0.8 Å)
    - Phase 2 (epoch 100-149): 73% of calibrated (~1.1 Å)
    - Phase 3 (epoch 150+):  100% of calibrated  (full physical threshold)
    """
    M = calibrated_min_pdist
    if epoch < CURRICULUM_EPOCH_BOUNDARIES[0]:        # < 50
        return CURRICULUM_FRACTIONS[0] * M             # 0.33 M
    elif epoch < CURRICULUM_EPOCH_BOUNDARIES[1]:      # 50-99
        return CURRICULUM_FRACTIONS[1] * M             # 0.53 M
    elif epoch < CURRICULUM_EPOCH_BOUNDARIES[2]:      # 100-149
        return CURRICULUM_FRACTIONS[2] * M             # 0.73 M
    else:                                              # >= 150
        return CURRICULUM_FRACTIONS[3] * M             # 1.00 M (final)


# LightningModule 实现 (v2 SA1 必须严格按此 patch):

class WGANGPModule(pl.LightningModule):
    def on_train_epoch_start(self):
        """Curriculum: gradually increase min_pdist by epoch."""
        ep = self.current_epoch
        new_pdist = get_curriculum_min_pdist(ep, self.calibrated_min_pdist)
        self.criterion.min_pdist = new_pdist
        # 在每 phase 切换的第一个 epoch 打印 (epoch 0, 50, 100, 150)
        if ep in [0] + CURRICULUM_EPOCH_BOUNDARIES:
            phase_idx = sum(1 for b in CURRICULUM_EPOCH_BOUNDARIES if ep >= b)
            self.print(
                f"\n[Curriculum] epoch={ep} Phase {phase_idx}: "
                f"min_pdist = {new_pdist:.4f} Å "
                f"({CURRICULUM_FRACTIONS[phase_idx]:.2f} × calibrated {self.calibrated_min_pdist:.4f})"
            )
        # 也 log 进 train metrics 给训练监控
        self.log('train_curriculum_min_pdist', new_pdist, on_epoch=True)
```

**Curriculum 与 ckpt selection / EarlyStop 的互动 (v2 关键)**:

| 阶段 | min_pdist | gate pass rate 期望 | ckpt selection? | EarlyStop counting? |
|---|---|---|---|---|
| Phase 0 (0-49) | 0.33 × cal | 高 (约束弱) | ❌ **disabled** | ❌ **disabled** |
| Phase 1 (50-99) | 0.53 × cal | 中 | ❌ disabled | ❌ disabled |
| Phase 2 (100-149) | 0.73 × cal | 中-低 | ❌ disabled | ❌ disabled |
| **Phase 3 (150+)** | 1.00 × cal (full) | 真实物理值 | ✅ **enabled** | ✅ **enabled** |

**为什么 Phase 0-2 必须 disable ckpt selection**:
- 在弱约束下 gate_pass_rate 自然高 (e.g. epoch 30 min_pdist=0.5 时 gate 可能 95%,但这是假的)
- 这些 epoch 的 best ckpt 是**假阳性**,不能选
- Lightning 实现: `ModelCheckpoint` 配 callback `every_n_epochs=1` + 自定义 `epoch >= 150` 条件,或包 wrap

**为什么 Phase 0-2 必须 disable EarlyStop**:
- EarlyStop 看 monitor metric plateau,但 curriculum 切换会让 metric 在 phase 边界**人为跳变** (e.g. epoch 50 min_pdist 从 0.5 → 0.8,pairwise violation 从 5% → 25%)
- 这种跳变不是 model 退步,是 curriculum 设计
- 必须 Phase 3 (epoch ≥ 150) 开始 EarlyStop counting,避免误触发

**SA1 实施细节** (Lightning 2.x):

```python
# step2/step2.1_train_wgangp.py

# Best ckpt only after curriculum ends
ckpt_callback = ModelCheckpoint(
    monitor='val_composite_ckpt_score',
    mode='max',
    save_top_k=3,
    every_n_epochs=1,
    # v2 关键: 自定义 condition 只在 epoch >= 150 save
    save_on_train_epoch_end=False,
)

class CurriculumCkptFilter(Callback):
    """只在 curriculum 结束后 (epoch ≥ 150) 才让 best ckpt 触发 save."""
    def on_validation_epoch_end(self, trainer, pl_module):
        if pl_module.current_epoch < CURRICULUM_EPOCH_BOUNDARIES[-1]:  # < 150
            # 临时屏蔽: 把当前 val metric 设为 -inf 让 ckpt callback 不选
            # 实现方式: hook ckpt_callback 内部 best_model_score / best_k_models
            pass  # 实际实现: 见 SA1 hand-back

# EarlyStop only after curriculum ends
es_callback = EarlyStopping(
    monitor='val_composite_ckpt_score',
    mode='max',
    patience=30,
    strict=False,  # L7 lesson
    verbose=True,
    # v2 关键: divergence threshold 或自定义 callback 让 epoch < 150 不计 patience
)

class CurriculumEarlyStopFilter(Callback):
    """只在 curriculum 结束后才让 EarlyStopping 开始 count patience."""
    def on_validation_epoch_end(self, trainer, pl_module):
        if pl_module.current_epoch < CURRICULUM_EPOCH_BOUNDARIES[-1]:
            # 重置 EarlyStop wait counter,patience 不消耗
            for cb in trainer.callbacks:
                if isinstance(cb, EarlyStopping):
                    cb.wait_count = 0  # 强制重置 patience counter
```

SA1 必须实现这两个 callback 并 verify epoch 0/30/50/100/120/150/151 的行为:
- epoch 0-149: ckpt_callback 不 save, es_callback wait_count = 0
- epoch 150+: ckpt_callback 正常 save,es_callback 开始 count

### 6.3 初始超参 (v1)

| Lambda / 参数 | v1 设值 | 来源 |
|---|---|---|
| `lambda_pmin` (Generator) | 1.0 | Exp5' 沿用 |
| `lambda_type` (Generator) | 1.0 | Exp6 v8 风格 |
| `lambda_gp` (Discriminator) | 10.0 | Gulrajani 2017 WGAN-GP §4 表 1 |
| `n_critic` (D:G 训练比例) | 5 | Gulrajani 2017 WGAN-GP §4 默认值 |
| `noise_dim` (z) | 128 | DCGAN 经典值 |
| Generator lr | 1e-4 | TTUR 设计,G 慢 |
| Discriminator lr | 4e-4 | TTUR 设计,D 快 (Heusel 2017 推荐) |
| Optimizer | Adam, β1=0.0, β2=0.9 | WGAN-GP §4 推荐 (注意 β1=0,不是 DCGAN 的 0.5) |

**⚠️ Lambda 量级 caveat (沿用 Exp6 v8 SOP)**:

Phase 2 sanity 必须先观察各项 loss 比值,目标后期 ratio:
```
loss_G_adversarial : loss_pmin : loss_type ≈ 1.0 : 0.3 : 1.0
loss_D_real - loss_D_fake : loss_gp ≈ 1.0 : 5.0 (gradient penalty 应主导)
```

若 epoch 5 sanity 比值偏离 > 3×,触发 hyperparam tuning re-run。

---

## 7. 训练配置

```yaml
硬件: 1× RTX 4090 24GB (Exp6 占用另一块)
batch_size: 32 (GAN 一般用更小 batch 防止 mode collapse)
n_critic: 5  # 每训 1 次 G,先训 5 次 D
optimizer:
  G: AdamW(lr=1e-4, betas=(0.0, 0.9), weight_decay=1e-4)
  D: AdamW(lr=4e-4, betas=(0.0, 0.9), weight_decay=1e-4)
gradient_clip: 1.0  # 防止 G 训练 explode
max_epochs: 500     # GAN 难判收敛,留充裕
mixed_precision: fp32  # GAN 训练精度敏感,bf16 易 NaN
spectral_norm:
  enabled: True
  modules: ['discriminator_only']  # 只 D 加 SN
ttur:
  enabled: True
  G_lr_per_step: 1e-4
  D_lr_per_step: 4e-4
```

### 7.1 开训前强制 sanity check (Exp5 系列 L1+L3+L4 lessons)

**L1 cartesian sanity (errata 3)**:
```python
# dataset_v3 已 verify,但 Exp7 启动前再跑一次
batch = next(iter(train_loader))
cart = batch['frac_coords'] * 20.0  # L=20
d_pairs = pairwise_distances(cart.view(-1, 3))
assert d_pairs[d_pairs > 0].min() >= 0.7, "Dataset has invalid bonds < 0.7 Å"
```

**L3 + L4 dry-run 双层 active (Exp5'' 教训)**:

```python
# 1) 训练 active:跑 5 sample 看 G/D/pairwise loss 都产生有限梯度
# 2) 评估 active:同 5 sample sample 一次,跑 step5_3 在 batch 上,验证:
assert (n_pred_shells_per_sample > 0).mean() >= 0.80, \
    "GAN generator 输出无 shell 结构,与 Exp5'' 候选 A 同坑"
```

L3 + L4 是 Exp5 系列**最痛**的两条 lesson,Exp7 v1 必须双层都过才进训练。

### 7.2 GAN-specific 训练监控 (新增,Exp5 系列没踩过的坑)

每 epoch 必 logging:

| Metric | 公式 | 健康阈值 |
|---|---|---|
| `G_loss_adversarial` | `-E[D(G(z))]` per epoch mean | 应缓慢下降但不到 0(0 = D 完全骗过,G 太强) |
| `D_loss_critic` | `E[D(x_fake)] - E[D(x_real)]` per epoch mean | 应缓慢上升(更负)但不到 -∞(D 太强) |
| `D_gp_loss` | gradient penalty per epoch mean | 健康区间 [0.01, 0.5] |
| `G_pmin_loss` | pairwise min loss | 应在 30 epoch 后 < 0.1 |
| `G_type_ce_loss` | type CE loss | 应缓慢下降 |
| **`mode_diversity`** | per batch: std(pred_pos) across batch | 健康区间 [0.05, 0.3] frac unit |
| **`mode_diversity_per_spectrum`** | 固定一个 spectrum 跑 10 个 z noise,看 10 个输出的 std | 应 > 0 (不是 mode collapse 到单一输出) |
| **`pairwise_violation_rate`** | per-sample 硬指示器 (`min_d < MIN_PDIST`).float().mean() | < 5% 后期 (沿用 Exp6 v8 公式) |
| **`val_n_pred_shells_zero_ratio`** | per-sample step5_3 切壳 = 0 的比例 | < 20% (沿用 Exp5'' L4 教训) |

`mode_diversity` 系列指标是 GAN-specific,Exp5/6 没有。**SA1 实现时必须 logging,GAN 训不出来就是看这个判断 mode collapse 还是其他问题**。

---

## 8. 评估配置 (v4 双套并报 + 公式完全锁定)

### 8.0 双套评估的设计动机

Exp7 验收**同时跑两套独立评估**:

| 套 | 名字 | 公式来源 | 主/副 | 用途 |
|---|---|---|---|---|
| **A** | **Composite Physical Score (CPS)** | EXP6_PROPOSAL_v8 §7.2 (公式已完全锁定) | **主验收** | Exp6 vs Exp7 架构对照 verdict;sote 最新 (round 3 后 shell-aware) |
| **B** | step5_3 7 项复合分 | Exp5' SA-METRICS-V3 (`step5_3_composite_score.py` 实现) | 副指标 | 与 Exp5'/Exp5'' 历史 baseline 直接可比;MA5 final report §13.3 must-do 第 3 条 |

**设计原则**:
- 两套**独立计算**,SA1 不允许共享中间状态 (避免 implementation 串扰)
- 任一套异常都触发诊断 (附录 B 第 19 条)
- 两套都过才 GREEN verdict (§11.1)
- 两套显著背离 (差 > 30% relative) 必须 raise,不允许"挑好的报"

---

### 8.1 套 A — Composite Physical Score (CPS,主验收)

**公式锁定完整搬自 EXP6_PROPOSAL_v8 §7.2.3,SA1 不许改任何常量 / 权重 / 公式形式**。

#### 8.1.0 数据源 — `shell_boundaries.pkl`

per-sample 从 `/home/tcat/diffcsp_exp4/data/shell_boundaries.pkl` (md5 `cf2050e4899160f5698ad2481377e94c`) 加载 sample-specific GT shell 边界,schema:

```python
{
    'threshold': 0.1563,           # gap_threshold (Å), p10 of train gap distribution
    'distances': np.float32 (n,),  # n 个邻居距离 (笛卡尔 Å),已 sorted ascending
    'species_Z': np.int8 (n,),     # 各邻居的 Z
    'shell_starts': np.float32 (S,),  # 每个 shell 的最小距离 (S 个 shell)
    'shell_ends': np.float32 (S,),    # 每个 shell 的最大距离
    'shell_n_atoms': np.int32 (S,),   # 每个 shell 的原子数
    'shell_of_atom': np.int32 (n,),   # 每个原子的 shell index (0..S-1)
    'eval_cutoff': float,             # 包含第 20 邻居的那个 shell 的外缘 (Å)
    'n_center_sites': int,
}
```

#### 8.1.1 设计 (从 Exp6 v8 §7.2.1 完整复刻)

- **Hard gate**: 物理可行性 (PV)。pred-pred 任意对距离 < MIN_PDIST → CPS = 0
- **6 加权子项**:
  - C1=0.25 (1st shell coordination number)
  - D1=0.20 (1st shell distance)
  - T1=0.17 (1st shell type, CNO 合并)
  - C2=0.15 (2nd shell coordination number)
  - D2=0.13 (2nd shell distance)
  - T2=0.10 (2nd shell type, CNO 合并)
- **Shell 划分**: per-sample 从 `shell_boundaries.pkl` 加载 GT `shell_starts[0:2]` 和 `shell_ends[0:2]`,**只评估前 2 个壳**
- **Pred 原子分壳**:
  - 若 `GT_shell_starts[s] - tol_shell_band ≤ d ≤ GT_shell_ends[s] + tol_shell_band` → 算入 shell s
  - `tol_shell_band = 0.1 Å`
- **Edge case**: 若 sample GT 只有 1 shell, 跳过 T2/D2/C2, weight 重新归一化到 [C1, D1, T1]
- **CNO 合并**: C(Z=6), N(Z=7), O(Z=8) 合并为虚拟类
- per-sample CPS ∈ [0, 1], dataset-level CPS = per-sample 平均

#### 8.1.2 实现来源 — 直接 cp from Exp6 (v5 关键修订)

**v5 不再让 SA1 重写 CPS 公式!Exp6-SA1 已落地全套实现**:

```bash
# SA1 step1 阶段必跑:
cp /home/tcat/experiment6_v7/shared/composite_score.py experiment7/shared/eval_cps.py
# 然后改 import path: 把 Exp6 内部 import 路径改为 experiment7 路径
# (具体 import path 调整由 SA1 在 step1.4 smoke test 中完成 + verify)
```

**为什么 v5 删除 v4 §8.1.2 的 160 行公式重写**:
- v4 写的 ~160 行 CPS Python 公式与 `/home/tcat/experiment6_v7/shared/composite_score.py` 实测内容 100% 重叠
- v4 让 SA1 "公式锁定不许改" 但又让 SA1 重写一遍 — **逻辑矛盾且违反 v3 clone-first 原则**
- v5 改为: Exp6 实现是事实的 ground truth,Exp7 直接 cp,SA1 不许重写

**SA1 实施步骤**:
1. `cp /home/tcat/experiment6_v7/shared/composite_score.py experiment7/shared/eval_cps.py`
2. 打开 `eval_cps.py`,在文件开头加 license note: `# Imported from /home/tcat/experiment6_v7/shared/composite_score.py @ 2026-05-10`
3. 调整 import path (e.g. 若原文件 `from .shared_const import ...`, 改 Exp7 对应路径)
4. **公式 / 权重 / 常量 (TOL_SHELL_BAND, WEIGHTS, MIN_PDIST loading 方式) 全部不许改**
5. 必须 verify: `eval_cps.py` 跑 5 个 sample 输出与 `/home/tcat/experiment6_v7/shared/composite_score.py` 直接调用结果完全一致 (sanity)

**禁止**:
- SA1 凭直觉调任何常量 (TOL_SHELL_BAND 0.1 / 各权重)
- SA1 改算法逻辑 (split_shells_from_gt / type_score_cno_merged 等函数)
- 任何想法说"Exp6 实现可能有 bug,我重写一个" — 这种争议必须 raise 给用户,不许擅自重写

**例外**: 若 Exp6-MA1 之后修订 `composite_score.py` (e.g. fix shell 边界 NaN 处理),Exp7 必须同步 update,SA1 需要看 Exp6 hand-back 文档判断是否需要重新 cp。


#### 8.1.3 CPS Logging 项 (val 每 epoch 必报)

| Metric | 用途 |
|---|---|
| `val_cps_mean` | 主验收指标 |
| `val_pv_pass_rate` | PV gate 通过率 |
| `val_C1_mean`, `val_D1_mean`, `val_T1_mean` | 1st shell 子项 (仅 PV=True sample) |
| `val_C2_mean`, `val_D2_mean`, `val_T2_mean` | 2nd shell 子项 |
| `val_pred_outside_shells_ratio` | 落 GT 前两壳之外的 valid pred 比例 |
| `val_pred_in_shell1_mean` / `val_pred_in_shell2_mean` | 预测原子的壳内分布 |

---

### 8.2 套 B — step5_3 7 项复合分 (副指标)

直接调用 Exp5' `step5_3_composite_score.py`,**SA1 不重新实现,只 import**。

#### 8.2.1 Step5_3 7 项内容 (公式见 Exp5' 实现,文档见 MA5 final report §5.1)

| # | 指标 | 类型 | 备注 |
|---|---|---|---|
| 1 | shell-1 dist score | 距离评估 | Exp5'/Exp5'' 历史 RED 项 |
| 2 | shell-1 coord_n score | 配位数评估 | |
| 3 | shell-1 elem score | 元素评估 | errata 2 §2 病态问题,CNO 合并 |
| 4 | shell-2 dist score | 距离评估 | |
| 5 | shell-2 coord_n score | 配位数评估 | Exp5'' 唯一改进项 |
| 6 | shell-2 elem score | 元素评估 | |
| 7 | overall geometric score | 整体几何 | Exp5'/'' 综合分 |

权重在 Exp5' `step5_3_composite_score.py` 已锁,SA1 verify md5 后**不许改任何系数**。

#### 8.2.2 step5_3 副指标 Logging

| Metric | 用途 |
|---|---|
| `val_step5_3_composite` | 副 verdict 指标,与 Exp5' 0.080 直接对比 |
| `val_step5_3_gate_pass_rate` | min_d ≥ MIN_PDIST 通过率 (与 CPS PV 同义但独立计算) |
| `val_step5_3_collapse_rate` | 沿用 Exp5 系列 collapse 定义 |
| `val_step5_3_min_d_mean` | 副指标 |
| `val_step5_3_n_pred_shells_zero_ratio` | L4 lesson 监控 |

---

### 8.3 双套对照表 + Exp7-specific 新增指标

#### 8.3.1 双套独立性 verify

每 epoch val 后 SA1 必须打印对照:

```
[Eval Dual] epoch=200
  CPS (主):     mean=0.156, PV_pass=87.3%
  step5_3 (副): composite=0.094, gate=72.1%, collapse=2.1%
  Δ_composite (CPS - step5_3) = +0.062
  Δ check: |Δ| / max = 39.7%  ← 若 > 30% 触发 SA1 raise (附录 B 19)
```

若两套差 > 30% relative,SA1 必须:
1. Dump 5 个差异最大的 sample 看具体哪项不一致
2. 检查是否 shell_boundaries.pkl 加载方式不一致 / MIN_PDIST 取值不同
3. raise 给用户决策,不允许继续训练

#### 8.3.2 Exp7-specific 新增评估指标 (双套之外)

| 指标 | 公式 | 用途 |
|---|---|---|
| **Mode coverage** | 同 spectrum 跑 N=50 z,KMeans on distance matrix K=5,数 unique cluster | 检查 mode collapse |
| **Mode diversity per spectrum** | std(pred_pos) across N=50 z,固定 spectrum | GAN-specific |
| **Diversity-quality tradeoff** | 横轴 mode_diversity,纵轴 CPS | GAN 经典权衡曲线 |

---

### 8.4 SA1 step1.3 必跑: Exp5' ckpt 双套 baseline

`step1.3_dual_baseline.py` 必须产出 `baseline_dual.json`:

```json
{
  "exp5_prime": {
    "ckpt": "composite_epoch169_score0.5881.ckpt",
    "md5": "127afa44a850d8f7e4fcdae17e2761a1",
    "CPS_val": 0.XX,        // SA1 step1.3 实测 (估计 30 min)
    "CPS_holdout": 0.XX,
    "step5_3_composite_val": 0.080,   // 已知 (MA5 final report §3.2)
    "step5_3_gate_val": 0.640,
    "step5_3_collapse_val": 0.0,
  },
  "exp6": {  // 等 Exp6 训完后回填
    ...
  },
  "exp7": null  // Exp7 训完后回填
}
```

**为什么必须重算 Exp5' CPS** (即使 step5_3 0.080 已知):
- CPS 用 shell_boundaries.pkl 的 GT 边界,step5_3 实现 SA1 需 verify 是否用同一边界
- 两套独立计算的 Exp5' baseline 才能严格作为 Exp7 verdict 对照
- 30 min 工作量换 verdict 严谨性,ROI 极高

---



## 9. 文件结构 (v5 — 路径全部锁死实测路径)

```
experiment7/
├── _vendor/                                  # cloned 第三方代码原文件,保留作 traceability
│   ├── eriklindernoren_PyTorch-GAN/          # git clone
│   ├── gcucurull_cond-wgan-gp/               # git clone, 只读参考
│   └── christiancosgrove_pytorch-spectral-normalization-gan/ # git clone
├── data/                                      # 链接到 Exp5'/Exp4 数据
│   ├── train_structure_cache.pt -> /home/tcat/diffcsp_exp5_prime/data/train_structure_cache.pt    # symlink, 44 MB
│   ├── val_structure_cache.pt   -> /home/tcat/diffcsp_exp5_prime/data/val_structure_cache.pt      # symlink, 5.6 MB
│   ├── test_structure_cache.pt  -> /home/tcat/diffcsp_exp5_prime/data/test_structure_cache.pt     # symlink, 3.3 MB
│   ├── holdout_structure_cache.pt                                                                  # SA1 Phase 4 前必跑 precompute_structure_cache 构建,不存在!
│   ├── cache_metadata.json     -> /home/tcat/diffcsp_exp5_prime/data/cache_metadata.json          # symlink, verify L_VIRTUAL=20.0
│   ├── shell_boundaries.pkl    -> /home/tcat/diffcsp_exp4/data/shell_boundaries.pkl               # symlink, md5 cf2050e4..., 387 MB
│   ├── spectra_*.pkl           -> /home/tcat/diffcsp_exp5_prime/data/spectra_*.pkl                # 4 个 symlink
│   ├── feff_features_imputed.pkl -> /home/tcat/diffcsp_exp5_prime/data/feff_features_imputed.pkl  # symlink
│   ├── feff_feature_scaler.pkl -> /home/tcat/diffcsp_exp5_prime/data/feff_feature_scaler.pkl      # symlink
│   ├── {train,val,test,holdout}_samples_v2.csv -> /home/tcat/diffcsp_exp5_prime/data/...           # 4 个 symlink
│   └── data_integrity.json                  # SA1 step1.0 写入,所有 md5 verify 记录
├── shared/
│   ├── xas_local_dataset_v2.py              # [CP from /home/tcat/diffcsp_exp5_prime/code/step3/xas_local_dataset_v2.py] 注意是 _v2 不是 _v3 (v1-v4 写错文件名)
│   ├── xas_local_datamodule_v2.py           # [CP from /home/tcat/diffcsp_exp5_prime/code/step3/xas_local_datamodule_v2.py]
│   ├── spectrum_encoder.py                  # [CP from /home/tcat/diffcsp_exp5_prime/code/step2/spectrum_encoder.py] (SA1 必先 diff Exp6 spectrum_tokenizer.py 验证一致性,见 §3.1.6)
│   ├── min_pdist_calibration.json           # [CP from /home/tcat/experiment6_v7/shared/min_pdist_calibration.json] MIN_PDIST = 1.5075718402862548 Å (frozen)
│   ├── shell_integrity_report.json          # [CP from /home/tcat/experiment6_v7/shared/shell_integrity_report.json]
│   ├── exp7_element_vocab.json              # [CP from /home/tcat/experiment6_v7/shared/exp6_element_vocab.json] 改名而已
│   ├── min_pdist_rdf_hist.png               # [CP from /home/tcat/experiment6_v7/shared/min_pdist_rdf_hist.png] 人工 review
│   ├── shell_n_atoms_hist.png               # [CP from /home/tcat/experiment6_v7/shared/shell_n_atoms_hist.png] 人工 review
│   ├── eval_cps.py                          # [CP from /home/tcat/experiment6_v7/shared/composite_score.py] ⭐ CPS 主套实现,SA1 改 import path 即可
│   ├── eval_step5_3.py                      # [CP from /home/tcat/diffcsp_exp5_prime/code/step5/step5_3_composite_score{_exp5_prime}.py] ⚠️ SA1 必 diff 两个版本选其一 (见 §3.1.5)
│   ├── eval_dual_runner.py                  # [NEW] 同时跑 CPS + step5_3,产出 dual report ~80 行
│   ├── spectral_normalization.py            # [CP from _vendor/christiancosgrove_.../spectral_normalization.py] zero modify, ~50 行
│   ├── cgan_base.py                         # [CP from _vendor/eriklindernoren_PyTorch-GAN/implementations/cgan/cgan.py] zero modify, ~200 行
│   ├── wgan_gp_base.py                      # [CP from _vendor/eriklindernoren_PyTorch-GAN/implementations/wgan_gp/wgan_gp.py] zero modify, ~200 行
│   ├── cond_wgan_gp.py                      # [SA1 合并] SA1 合并 cgan_base + wgan_gp_base,cross-check gcucurull_cond-wgan-gp,~150 行 modify
│   ├── generator.py                         # [SA1 改造 cgan_base.py Generator] 替换 image ConvTranspose decoder 为 MLP for (20, 3+K+1),~80 行 modify
│   ├── discriminator.py                     # [SA1 改造 wgan_gp_base.py Discriminator] 替换 image Conv2d input 为 distance matrix 1D CNN,~80 行 modify
│   ├── pairwise_min_loss.py                 # [SA1 改造] 从 /home/tcat/diffcsp_exp5_prime/code/step3/diffusion_w_type_xas.py 摘出 _pairwise_min_distance_penalty,改输入 shape 为 (B, 20, 3) + 加 curriculum schedule (§6.2.4)
│   ├── type_ce_loss.py                      # [NEW] ~30 行
│   └── curriculum_callbacks.py              # [NEW] CurriculumCkptFilter + CurriculumEarlyStopFilter,~100 行
├── step1/
│   ├── step1.0_cache_setup.py               # [SA1 写] symlink 上述所有数据文件 + md5 verify 写入 data_integrity.json
│   ├── (step1.0a_shell_integrity_check.py)  # ✘ 跳过, Exp6 已跑,直接 cp shell_integrity_report.json
│   ├── (step1.0_rdf_analysis.py)            # ✘ 跳过, Exp6 已 calibrate,直接 cp min_pdist_calibration.json
│   ├── (step1.1_build_vocab.py)             # ✘ 跳过, Exp6 已建,直接 cp 改名
│   ├── (step1.2_baseline_recompute.py)      # ✘ 跳过, baseline_cps.json (Exp4/Exp5') 已存在
│   ├── step1.3_dual_baseline_exp5_prime.py  # [SA1 必跑] 用 Exp5' ckpt 跑 CPS + step5_3 双套 baseline ~30 min,产出 baseline_dual.json (因 baseline_cps.json 是 Exp4 ckpt 的 baseline,Exp5' CPS 待 SA1 实测)
│   └── step1.4_smoke_test.py                # [SA1 必跑] 5 sample G/D forward + 4 loss + curriculum schedule verify + L3 + L4 双层 dry-run + 双套评估各跑一次
├── step2/
│   └── step2.1_train_wgangp.py              # [SA1 写] 训练主脚本,~250 行 (single GPU cuda:1, TTUR, n_critic=5, curriculum callbacks)
├── step3/
│   ├── step3.0_build_holdout_cache.py       # [SA1 Phase 4 前必跑] cp /home/tcat/diffcsp_exp5_prime/code/step3/precompute_structure_cache_exp5_prime.py + 跑 holdout 构建 holdout_structure_cache.pt (~5-10 min)
│   └── step3.1_sample_and_eval.py           # val/test/holdout sample + 双套评估
├── step4/
│   └── step4.1_final_report.md              # 训完后产出
└── EXP7_GAN_PROPOSAL_v5.md                  # 本文档
```

**v5 关键变化** (相对 v4):

| 变化 | 说明 |
|---|---|
| 文件名修正 | dataset 是 `_v2.py` 不是 `_v3.py` (v1-v4 错) |
| `eval_cps.py` 来源改 cp | v4 让 SA1 重写 ~250 行,v5 改为 cp Exp6 `composite_score.py` |
| `step1.0a/1.0/1.1/1.2` 全跳过 | Exp6 已跑完,Exp7 直接复用产出文件 |
| 加 `step3.0_build_holdout_cache.py` | holdout cache 不存在,Phase 4 前必跑 |
| 加 `data/` 目录的 symlink 清单 | 8+ 个 symlink 全部锁死实测路径 |
| 加 `data_integrity.json` 流程闸门 | SA1 step1.0 必跑 md5 verify |

---

## 10. 时间表 (v5 — 利用 Exp6 已完成 step1, Phase 1 减半)

| Phase | 内容 | 时间 (single 4090 cuda:1) | 输出 |
|---|---|---|---|
| Phase 0 | 读 3 篇论文 (Gulrajani 2017 / Miyato 2018 / Heusel 2017) + Exp5 系列 final report §10.4 + EXP6_PROPOSAL_v8 §附录 B + clone 3 个 vendor repos + diff step5_3 双版本 + diff spectrum_encoder vs spectrum_tokenizer + **§5.5 dataset contract V1-V5 verify (v6 新增)** | 1 天 | SA1 笔记 + `_vendor/` 准备 + step5_3 / spectrum_encoder 决议 + **`dataset_contract_audit.json` (v6 流程闸门)** |
| Phase 1 | step1.0 symlink + md5 verify (Exp6 已完成 step1.0a/1.0/1.1/1.2 直接 cp) + step1.3 Exp5' dual baseline (~30 min) + 合并 cond_wgan_gp.py + 改造 generator/discriminator + 加 curriculum + smoke_test | **1.5 天** (v5 从 v4 的 2.5 天减 1 天,因 Exp6 step1 已完成) | smoke_test 通过,baseline_dual.json 含 Exp5' 双套数字 |
| Phase 2 | 训练脚本,跑 10 epoch sanity check + lambda tuning + mode collapse 诊断 + curriculum schedule verify | 2 天 | 4 loss + curriculum + mode_diversity 全正常 |
| Phase 3 | 完整训练 (单 4090 估计 1-1.5 天 / 500 epoch) | 1.5-2 天 | val_cps ≥ baseline + 0.05 |
| Phase 3.5 | Phase 4 前: SA1 跑 `step3.0_build_holdout_cache.py` 构建 holdout cache | 0.5 天 (含 sanity verify) | holdout_structure_cache.pt |
| Phase 4 | val/test/holdout sample + 双套评估 (CPS + step5_3) + 三档 ckpt 对照 + 用户物理 sanity 必经 | 1.5 天 | EXP7_FINAL_REPORT.md |
| **总计** | | **~ 8 天** (v4 9 天 → v5 8 天,Exp6 step1 复用节省 1 天) | |

**v5 关键节省来源**:
- step1.0a / 1.0_rdf / 1.1_vocab / 1.2_baseline_cps 全跳过 (Exp6 已完成,直接 cp 5 个 json/png 文件) — 节省 ~1 天
- eval_cps.py 不写 (Exp6 composite_score.py 直接 cp) — 节省 ~0.5 天
- 部分抵消: 加 Phase 3.5 build holdout cache + 加 Day 0 diff step5_3 两版本/spectrum_encoder/tokenizer 验证 — 多 ~0.5 天


---

## 11. 验收标准

**v1 采用 baseline-relative 框架**(沿用 Exp6 v8 SOP 3):proposal 阶段不硬定阈值,SA1 step1.3 跑出 Exp5' 实测 baseline 后写入 `acceptance_thresholds.json` 并冻结。

### 11.1 通过 (Exp7 GREEN)

**v2 关键前置 (curriculum)**: 所有验收指标必须**在 epoch ≥ 150 (Phase 3, full min_pdist) 期间评估**。Curriculum 早期 (epoch < 150) 的 metric 数值不参与 verdict。

**v4 关键前置 (双套并报)**: 验收指标分**主套 (CPS)** 和**副套 (step5_3)**,**两套都过**才 GREEN。任一套失败降级 AMBER。

#### 11.1.1 主套 — CPS 验收阈值 (必过)

| 指标 | v4 阈值 | 依据 (SA1 step1.3 baseline_dual.json 实测后回填) |
|---|---|---|
| **val_cps_mean** | `≥ max(Exp4_CPS, Exp5'_CPS, Exp6_CPS) + 0.05` | SA1 step1.3 实测;Exp6_CPS 等 Exp6 训完回填 |
| **holdout_cps_mean** | `≥ max(Exp4_CPS, Exp5'_CPS, Exp6_CPS)` | 防过拟合 buffer |
| **val_pv_pass_rate** | `≥ max(Exp4_PV_pass, Exp5'_PV_pass) + 5pp` | SA1 step1.3 实测 |
| **val_pred_outside_shells_ratio** | `≤ Exp4_outside_shells_ratio - 0.1` | SA1 step1.3 实测 |
| **val_C1_mean, D1_mean, T1_mean** (1st shell 子项) | 三项均**显著优于 Exp5'** (具体阈值 SA1 实测 Exp5' 后回填) | shell-1 是 Exp5 系列 RED 重灾区 |

#### 11.1.2 副套 — step5_3 验收阈值 (必过)

| 指标 | v4 阈值 | 依据 |
|---|---|---|
| **val_step5_3_composite** | `≥ Exp5'_step5_3 + 0.02` (≥ 0.10 absolute) | MA5 final report §3.2 给 Exp5' val composite 0.080 |
| **holdout_step5_3_composite** | `≥ Exp5'_step5_3` (≥ 0.080) | 防过拟合 |
| **val_step5_3_gate_pass_rate** | `≥ 0.69` (Exp5' 0.640 + 5pp) | MA5 final report §3.2 |
| **val_step5_3_collapse_rate** | `≤ 0.05` (Exp5' 0% baseline,Exp7 GAN 风险更高) | Exp5'' fail 30% collapse 历史警示 |
| **val_step5_3_min_d_mean** | `≥ 1.65 Å` | Exp5' 1.687 Å baseline,允许小幅退步 |
| **val_step5_3_n_pred_shells_zero_ratio** | `≤ 0.20` | L4 lesson,绕开 Exp5'' 候选 A 坑 |

#### 11.1.3 双套一致性 (必过)

| 指标 | v4 阈值 | 依据 |
|---|---|---|
| **\|val_cps - val_step5_3\| / max** | `< 0.30` (relative diff) | 双套不应大幅背离;> 30% 触发附录 B 19 raise |

#### 11.1.4 GAN-specific (必过)

| 指标 | v4 阈值 | 依据 |
|---|---|---|
| **mode_diversity_per_spectrum** | KMeans (K=5) 输出非全单簇 (≥ 2 cluster) | inverse design 任务必须保多样性 |
| **mode_coverage (N=50 z)** | 同 spectrum 输出至少有 2 unique cluster | 同上 |
| **pairwise_violation_rate (Phase 3)** | `≤ 0.05` epoch ≥ 150 期间 | 沿用 v2 公式 |

#### 11.1.5 流程闸门 (必过)

| 检查 | 内容 |
|---|---|
| Curriculum 完成证据 | train log 显示 epoch 0/50/100/150 各 phase 切换 + best ckpt epoch ≥ 150 |
| **用户物理 sanity 必经** | 沿用 Exp6 v8 SOP 12,见 §11.1.6 |

#### 11.1.6 用户物理 sanity 必经 (流程闸门,自动指标过不能跳过)

val + holdout predictions 交给用户跑 independent 物理统计:
1. Min pred-pred distance histogram (Phase 3 best ckpt)
2. Shell-1 / shell-2 配位数 vs GT 分布
3. Shell-1 / shell-2 距离 vs GT 分布 (CPS C1/D1, step5_3 shell-1 两套各算一次)
4. **GAN-specific**: 同 5 个 spectrum 各跑 10 个 z noise,看输出多样性是否合理 + 双套 metric 是否一致

用户独立判定通过后,verdict 才能 declare 通过。

### 11.2 部分成功 (Exp7 AMBER)

任一情况:
- 主套 (CPS) 通过但副套 (step5_3) 不通过(或反之)
- 双套差 > 30% 但 SA1 解释为已知 implementation 差异 (用户决议接受)
- val_cps ∈ [Exp5'_CPS, Exp5'_CPS + 0.05) — Exp7 与 Exp5' 接近但优势不显著
- 或: val_step5_3_composite ∈ [Exp5'_step5_3, Exp5'_step5_3 + 0.02)
- 或: 两套都过但 mode_diversity 不健康 (mode collapse 但 quality 好)
- 或: 用户物理 sanity 不通过

→ Exp7 不取代 Exp5'/Exp6,但作为"第三条路径已尝试"的论文论据保留

### 11.3 失败 (Exp7 RED)

任一情况:
- **主套 (CPS) val_cps < Exp5'_CPS** (主验收失败)
- 或: **副套 (step5_3) val_composite < Exp5'_step5_3** (副验收失败)
- 或: val_step5_3_collapse_rate > 0.15 — GAN mode collapse 严重
- 或: val_step5_3_n_pred_shells_zero_ratio > 0.50 — Discriminator 没学到 shell (Exp7 v3 假设失败)
- 或: 训练根本不收敛 (G_loss diverge 或 D_loss → -∞)
- 或: 双套 metric 差 > 50% relative 无法解释 → implementation bug

→ 实验结束。论文里把 Exp7 写为 negative result,与 Exp5'' 一起作为"loss-level / GAN 路径都触顶"的论据,论证只有 Transformer (Exp6) 或更激进架构能突破


---

## 12. 风险与回退

### 12.0 术语命名约定 ⚠️

沿用 Exp6 v8 §11.0 命名分离 + Exp7 新增 GAN 专项:

| 术语 | 含义 | 性质 |
|---|---|---|
| `query_pile-up` / `query_degeneracy` | DETR 早期 query 输出位置雷同 | Exp6 用,Exp7 无 query 不适用 |
| `pred_collapse` | Exp4/5 的 `_density_loss` 导致预测原子塌缩到中心 | 失败模式 |
| `repulsion_degradation` | lambda_rep 过大导致 model 全输出 no_object | Exp6 用,Exp7 无 no_object 不适用 |
| **`mode_collapse`** (v1 新增) | GAN 经典: G 学到 trivial 解骗过 D,所有不同 z 都输出相同结构 | Exp7 GAN 专属失败模式 |
| **`pseudo_resolution`** (Exp5'' lesson) | 训练 loss 通过但评估 metric 失败 | Exp5''→Exp7 必须主动 check |

**Exp7 训练日志和 final report 严格遵守命名分离**。`mode_collapse` 是 Exp7 专属,与 Exp4/5 `pred_collapse` 严格区分。

### 12.1 已知风险

**风险 1: GAN 训练不收敛 / G_loss diverge (GAN 经典)**
- 表现: G_loss 暴涨或负无穷,D_loss 立刻 0
- 根因: D 太强,G 学不到 / 反之
- 监控: G_loss & D_loss 每 epoch print
- 应对:
  1. 检查 spectral norm 是否启用 (D 端)
  2. 检查 TTUR lr 设置 (G 1e-4, D 4e-4)
  3. 检查 lambda_gp = 10.0 (Gulrajani 标值)
  4. n_critic 改 7-10 (D 太弱)
- **回退**: 若 50 epoch 仍不收敛,abort Exp7 P3,切到 Exp7-v2 改 DCGAN 而非 WGAN-GP

**风险 2: Mode collapse (GAN 经典)**
- 表现: `mode_diversity_per_spectrum` → 0,N 个 z 输出近相同
- 根因: G 找到捷径单一解骗过 D
- 监控: 每 epoch dump `mode_diversity_per_spectrum`,固定 5 spectrum
- 应对:
  1. 加 mini-batch discrimination (Salimans 2016 trick)
  2. 加 G output diversity loss: `-mean(std(G(z, x), axis=batch))`
  3. 降 lambda_pmin (pairwise hinge 过强可能压缩 mode)
- **回退**: 若 100 epoch mode_diversity < 0.05 且 quality OK,接受 mode collapse,Exp7 verdict AMBER

**风险 3: Discriminator 没学到 shell — 假设失败**
- 表现: `val_n_pred_shells > 0` 比例 < 50%
- 这是 Exp7 v1 的核心假设失败 (distance matrix D 隐式学 shell)
- 监控: SA1 训练 epoch 5 必 dump (沿用 Exp5'' L4 教训)
- 应对:
  1. 扩大 D 容量 (increase channels, depth)
  2. D input 加 sorted distance matrix (per row sorted)
  3. D 加 attention 机制识别 shell pattern
- **回退**: 若 P3 sample 该指标 RED,Exp7 verdict ❌ FAILURE,论文里报告"distance matrix D 不充分学 shell"作 negative result

**风险 4: Pairwise hinge loss 与 GAN 训练冲突**
- 表现: G_pmin_loss 收敛但 G_adversarial_loss 不降 (G 满足 pairwise 但骗不过 D)
- 根因: λ_pmin = 1.0 可能在 GAN 框架下过强
- 监控: G_pmin_loss / G_adversarial_loss 比值
- 应对:
  1. 降 λ_pmin 0.5 → 0.2 → 0.0
  2. 若降到 0 GAN 才能学,**仅靠 D 隐式学 pairwise** (D 应该已经学到 real distance min > 1.5 Å)
- **设计 caveat**: 即使 λ_pmin=0 也是合理的 (D 应该兜底),但 Exp5' 已证 pairwise 自启动有效,保留作冗余

**风险 5: Per-z noise 不足以覆盖 spectrum 对应的所有可能结构**
- 表现: 同 spectrum 跑 N=50 z 后,KMeans K=5 只出 1-2 簇
- 根因: 128d z 可能对该任务嫌少
- 应对: noise_dim 改 256 / 512;或加 z 的 spatially aware encoding
- **设计 caveat**: 物理上同 spectrum 可能对应 1 个唯一结构(supervised matching gives a single ground truth),所以这里"多 mode"指的是与 GT 接近但有 acceptable 微扰的多个解,不是真的多个不同结构

**风险 6 (v2 新增): Curriculum 阶段切换导致训练 oscillation**
- 表现: epoch 50/100/150 切换瞬间 G_loss / D_loss 剧烈跳变,model 振荡几个 epoch 才稳定
- 根因: `min_pdist` 从 0.33M 突跳 0.53M (e.g. 0.50 → 0.80 Å),pairwise violation 突然增多,loss 突跳 → G 梯度突变
- 监控: SA1 必须在 epoch 49/50/51, 99/100/101, 149/150/151 各跑 val 检查 loss 跳变幅度
- 应对 (按优先级):
  1. **首选**: 接受短暂 oscillation (2-3 epoch 内自稳)。这是 curriculum 设计的必然代价
  2. **次选**: 切换时降 lr `× 0.5` 持续 5 epoch,然后恢复 (Lightning LRScheduler 配 step at 50/100/150)
  3. **末选**: 改 curriculum 为更平滑的 linear ramp: `min_pdist = 0.33M + (0.67M × min(1, epoch/150))`,但失去 phase 清晰的 boundary,SA1 调试更难
- **不接受的应对**: 改 CURRICULUM_FRACTIONS 数值 (这是 SOP 锁定值)

**风险 7 (v2 新增): Phase 3 (epoch ≥ 150) 后 EarlyStop 立即触发**
- 表现: 进入 Phase 3 后第 30 epoch (即 epoch 180) EarlyStop 直接停,wait_count 从 0 直接到 30
- 根因: Phase 3 第一次 val (epoch 150) 设为 best,后续 val 因为 min_pdist 突然升至 100% 导致 metric 退步,patience 立刻消耗
- 监控: epoch 150/155/160/165 的 val 必须 verify
- 应对:
  1. **首选**: 给 Phase 3 一个 "重置 grace period" — 进入 phase 3 后 wait_count 重置到 -10 (相当于多给 10 epoch buffer)
  2. **次选**: EarlyStop patience 从 30 改 40 (单独给 Phase 3 更长 patience)
- 这个风险与风险 6 相关但不同 — 风险 6 是切换瞬间 oscillation,风险 7 是 Phase 3 metric 不再爬升的早 EarlyStop

### 12.2 回退方案

如果 Exp7 完全失败 (P3 verdict RED):

1. **Exp7-v2 候选**: 把 WGAN-GP 换成 LSGAN / DCGAN,牺牲稳定性换 generator expressiveness
2. **Exp7-v3 候选**: Hybrid diffusion + GAN — diffusion 主线 + GAN discriminator 加 reward signal (`EXPERIMENT5_SERIES_FINAL_REPORT.md §10.4` 已提示)
3. **完全放弃 GAN**: Exp7 final report 写 negative result,与 Exp6 verdict 一起进 full paper §5 Failure Analysis

### 12.3 与 Exp6 verdict 的交互

| Exp6 verdict | Exp7 调整 |
|---|---|
| Exp6 GREEN (composite > 0.30) | Exp7 不调整,继续验证 GAN 路径作 ablation |
| Exp6 AMBER (composite 0.15-0.30) | Exp7 不调整,但 verdict 阈值 = max(Exp5'_composite, Exp6_composite) + 0.02 |
| Exp6 RED (composite < Exp5'_composite) | Exp7 verdict 阈值仍 vs Exp5',但若 Exp7 也 RED,可能架构整体失败,需 Exp8 大改 |

---

## 附录 A: GAN 关键概念速查 (供 SA1 / Exp7-MA 参考)

**WGAN-GP**: Gulrajani et al. 2017. 用 Wasserstein 距离作为 GAN 训练目标,gradient penalty 强制 1-Lipschitz 约束。比原版 WGAN 的 weight clipping 稳定。是 GAN 工业级实现的事实标准。

**Spectral Normalization**: Miyato et al. 2018. 通过谱归一化限制每层权重矩阵的最大奇异值 = 1,等价限制网络 Lipschitz 常数。一般只在 discriminator 用。

**TTUR (Two-Time-Scale Update Rule)**: Heusel et al. 2017. G 和 D 用不同 learning rate,具体而言 D 学得更快 (lr 大),让 D 始终 "略强于" G。WGAN-GP + TTUR 是经典稳定方案。

**Mode Collapse**: GAN 经典失败模式。Generator 找到某个能骗过 D 的局部最优解后,所有 noise z 都输出相同结构。诊断: 同 condition 多个 noise z 看输出多样性。

**Conditional GAN (cGAN)**: 给 G 和 D 都加 condition input (本 Exp7 condition = spectrum)。让 G 学到 P(structure | spectrum),不是 marginal P(structure)。

**Gradient Penalty 1-Lipschitz**: WGAN 理论要求 D 是 1-Lipschitz 函数(任意两点输入,输出差 ≤ 输入差)。Gulrajani 用 `(||grad|| - 1)^2` penalty 软实现这个约束。

---

## 附录 B: 给 Main Agent / SA 的具体指令

执行顺序:

1. **不要**碰 Exp6 / Exp5' 的代码,Exp7 全部新建在 `experiment7/` 目录下,与 Exp6 (`experiment6/`) / Exp5' (`/home/tcat/diffcsp_exp5_prime/`) 完全隔离
2. **第一件事** (v3 clone-first):
   - **clone 3 个 vendor repos 到 `experiment7/_vendor/`**:
     ```bash
     cd experiment7/
     mkdir -p _vendor && cd _vendor
     git clone https://github.com/eriklindernoren/PyTorch-GAN.git eriklindernoren_PyTorch-GAN
     git clone https://github.com/gcucurull/cond-wgan-gp.git gcucurull_cond-wgan-gp
     git clone https://github.com/christiancosgrove/pytorch-spectral-normalization-gan.git christiancosgrove_pytorch-spectral-normalization-gan
     ```
   - 读 Gulrajani 2017 / Miyato 2018 / Heusel 2017 三篇论文 (SA1 必读)
   - 从 `_vendor/eriklindernoren_PyTorch-GAN/implementations/cgan/cgan.py` cp 一份到 `shared/cgan_base.py` (零改动)
   - 从 `_vendor/eriklindernoren_PyTorch-GAN/implementations/wgan_gp/wgan_gp.py` cp 一份到 `shared/wgan_gp_base.py` (零改动)
   - 从 `_vendor/christiancosgrove_pytorch-spectral-normalization-gan/spectral_normalization.py` cp 到 `shared/` (零改动)
   - 阅读 `_vendor/gcucurull_cond-wgan-gp/` 的 cond_wgan_gp 实现,作 SA1 合并 cgan_base + wgan_gp_base 的 cross-check 参考
   - **禁止**从 TF 仓库 `igul222/improved_wgan_training` 手写 PyTorch 版 (v1/v2 错误,v3 已修)
   - 在 §10 Phase 0 完成前**禁止**写训练代码 (即 step2.1_train_wgangp.py)
3. **Phase 1 第一个产出**: 按 §9 顺序:
   - `step1.0_cache_setup.py` (cp/symlink L=20 cache,md5 verify shell_boundaries.pkl 真值 `cf2050e4899160f5698ad2481377e94c`)
   - `step1.1_build_vocab.py` (沿用 Exp6 vocab 若已建)
   - `step1.2_calibrate_min_pdist.py` (沿用 Exp6 calibration 若已做)
   - `step1.3_baseline_step5_3.py` (**必跑** Exp5' ckpt `composite_epoch169_score0.5881.ckpt` 跑 step5_3,产出 baseline)
   - `step1.4_smoke_test.py`: 5 样本 G + D forward + 4 loss + L3 训练 active + **L4 评估 active 双层验证**
   - 在 step1.4 通过之前**禁止**写训练脚本
4. **必须 logging** (新增 GAN-specific,沿用 Exp6 v8 全部 §附录 B.5 公式):
   - G_loss, D_loss, GP_loss, pairwise hinge, type CE (核心 5 项)
   - `mode_diversity_per_spectrum` (固定 5 spectrum 各跑 10 z, 每 5 epoch dump)
   - `pairwise_violation_rate` (硬指示器版,沿用 Exp6 v8 公式)
   - `val_n_pred_shells_zero_ratio` (L4 教训)
5. **禁止**:
   - 任何形式的 explicit shell loss (Exp5' 鸡蛋 + Exp5'' 候选 A 双重证伪)
   - 任何形式的 `_density_loss` 类型 attractive prior
   - 重新引入 TypeClassifier head
   - 训练中途调 MIN_PDIST (沿用 Exp6 v8 SOP 9, calibration 后冻结)
   - 训练中途改 `lambda_gp` (Gulrajani 2017 标值 10.0,SA1 不许改)
   - 训练中途改 `n_critic` 5 → 其他值 (除非风险 1 触发)
6. **必须**:
   - Holdout 永久封存,Exp7 训练全程不 touch (沿用 Exp4-Exp6 全程)
   - Exp7 best ckpt 选择基于 `val_composite + 0.3 * val_gate_pass_rate` (Exp5'-MA `composite_ckpt_score` 模式,但权重不同)
   - **训练完最后必跑用户物理 sanity** (L_SERIES §10.5 + Exp6 v8 §附录 B 第 12 条)
   - 三档 ckpt 对照 (Exp5' / Exp6 if done / Exp7) 写入 final report
7. **流程闸门** (Exp6 v8 第 12 条 + Exp5 系列 L4):
   - Smoke test 必须包含**评估 active** 验证 (`n_pred_shells > 0` 比例 ≥ 80% 在 5 sample),不允许"训练 active 通过即过"
8. **数据 / 路径一致性** (沿用 Exp6 v8 SOP 4):
   - `shell_boundaries.pkl` md5 必须 verify `cf2050e4899160f5698ad2481377e94c` (来自 EXPERIMENT5_SERIES_FINAL_REPORT.md §8.2)
   - Exp5' ckpt md5 必须 verify `127afa44a850d8f7e4fcdae17e2761a1` (来自 EXPERIMENT5_SERIES_FINAL_REPORT.md §8.1)
   - 所有外部数据加载前 md5 verify,写入 `experiment7/shared/data_integrity.json`
9. **GAN-specific**:
   - GAN 训练**不可重启** (Adam state important),只能 from-scratch。若中途 OOM,必须从 epoch 0 重训,**不要 warm-start**
   - 训练前打印 G 和 D 各自参数量,目标 D 参数量 ≈ G × (0.5 - 1.0) (D 容量太小学不到,太大 G 学不动)
   - 每 50 epoch 必 dump 5 个 spectrum × 10 z noise 的输出,可视化看是否 mode collapse

10. **必须** (v2 新增,curriculum learning):
    - `criterion.min_pdist` 不允许在 `__init__` 硬编码;必须通过 `WGANGPModule.on_train_epoch_start` 每 epoch 由 `get_curriculum_min_pdist(epoch, calibrated_min_pdist)` 注入
    - **ModelCheckpoint 必须包 CurriculumCkptFilter callback**,Phase 0-2 期 (epoch < 150) 不允许 save best ckpt,只允许 save last.ckpt
    - **EarlyStopping 必须包 CurriculumEarlyStopFilter callback**,Phase 0-2 期 (epoch < 150) `wait_count` 强制保持 0
    - SA1 必须在 smoke test 阶段 verify curriculum:跑 epoch 0/49/50/99/100/149/150/151 各 1 step,确认 `criterion.min_pdist` 数值与 schedule 一致
    - **禁止**改 `CURRICULUM_FRACTIONS` 数值 (0.33, 0.53, 0.73, 1.00) 或 `CURRICULUM_EPOCH_BOUNDARIES` 数值 (50, 100, 150)。若实测发现切换太激进,通过 §11.1 风险 6/7 应对方案处理,不改 schedule 本身

11. **必须** (v2 新增,curriculum 期 metric logging):
    - 每 epoch 必 logging `train_curriculum_min_pdist` (当前 epoch 实际使用的 min_pdist) — 这是关键诊断,验证 curriculum 真在运行
    - val 期 logging 必须包括 `val_pairwise_violation_rate_at_curriculum_threshold` (按当前 epoch 的 min_pdist 算) **和** `val_pairwise_violation_rate_at_full_threshold` (按 calibration 100% 值算)
    - 两个版本同时报: 前者监控当前 phase 的 model 行为,后者监控向 Phase 3 终态的 distance — 是 curriculum 进展真实指标

12. **必须** (v4 新增,双套评估 SOP 1 fix):
    - 每 val epoch **同时**跑两套独立评估 (CPS 主 + step5_3 副),logging 各自 metric (8.1.3 + 8.2.2)
    - SA1 实现两套时**禁止共享中间状态**:CPS 从 `shell_boundaries.pkl` 独立加载 + step5_3 调用 Exp5' `step5_3_composite_score.py` 独立 import,两套各自维护 SHELL_BOUNDARIES 副本
    - **每 epoch 末必须打印 dual-eval 对比** (见 §8.3.1 格式),让 SA1 / 用户实时观察
    - **两套差 > 30% relative** 必须 raise,不允许继续训练 — 这种背离表示至少一套有 implementation bug
    - **禁止**: 任一套异常时只报另一套 ("挑高的报") — 这是 cherry-picking 反 pattern,违反 MA5 final report §6.5 lesson
    - **禁止**: 删减任一套或减项 — 两套都是 sote,SA1 不许擅自简化

13. **必须** (v4 新增,baseline_dual.json 流程):
    - SA1 step1.3 必须用 `eval_dual_runner.py` 跑 Exp5' best ckpt,产出 `baseline_dual.json` 含**两套 baseline 数值**
    - `acceptance_thresholds.json` 必须包含**两套各自的 GREEN/AMBER/RED 阈值** (§11.1.1 + §11.1.2)
    - 任一套阈值未填则训练不许启动 (流程闸门)

14. **必须** (v5 新增, SOP 4 路径锁定):
    - **`step1.0_cache_setup.py` 必须先做 md5 verify** 才能 symlink:
      - `shell_boundaries.pkl` md5 = `cf2050e4899160f5698ad2481377e94c` (`/home/tcat/diffcsp_exp4/data/shell_boundaries.pkl`)
      - Exp5' ckpt md5 = `127afa44a850d8f7e4fcdae17e2761a1` (`/home/tcat/diffcsp_exp5_prime/checkpoints/composite_epoch169_score0.5881.ckpt`)
      - `cache_metadata.json` 必须含 `"L_VIRTUAL": 20.0`(Exp5' STEP1-FIX-C 产出)
    - **写入 `experiment7/data/data_integrity.json`** 记录所有 verify 结果 + 时间戳
    - **Verify 失败禁止继续**,SA1 raise 给用户决议
    - **禁止**: SA1 假定 md5 已知就跳过 verify (v1-v4 流程漏洞)

15. **必须** (v5 新增, Exp6 复用 SOP):
    - SA1 step1 阶段直接 cp 以下 5 个 Exp6 产出 (Exp6 step1 已完成):
      1. `/home/tcat/experiment6_v7/shared/min_pdist_calibration.json` → `shared/`
      2. `/home/tcat/experiment6_v7/shared/shell_integrity_report.json` → `shared/`
      3. `/home/tcat/experiment6_v7/shared/exp6_element_vocab.json` → `shared/exp7_element_vocab.json` (改名)
      4. `/home/tcat/experiment6_v7/shared/baseline_cps.json` → `shared/` (Exp4 baseline 部分)
      5. `/home/tcat/experiment6_v7/shared/composite_score.py` → `shared/eval_cps.py`
    - **禁止**: SA1 自己重新跑 RDF calibration / shell integrity check / build vocab。Exp6 已 verify 的数据 SA1 不许重做
    - **例外**: 若 Exp6-MA 之后修订上述任一文件 (e.g. composite_score.py bug fix),Exp7-MA 必须 ack 并重新 cp,SA1 不许擅自决定是否同步

16. **必须** (v5 新增, Day 0 必做 diff 验证):
    - SA1 Day 0 必须 diff 两组潜在歧义文件:
      1. `step5_3_composite_score.py` vs `step5_3_composite_score_exp5_prime.py` — 选与 Exp5' final report §3.2 verdict 0.0801 一致的版本 (默认假设 `_exp5_prime` 后缀版,但 SA1 verify)
      2. `/home/tcat/diffcsp_exp5_prime/code/step2/spectrum_encoder.py` vs `/home/tcat/experiment6_v7/shared/spectrum_tokenizer.py` — verify 实质实现一致后,选 Exp5' 版命名为 Exp7 spectrum_encoder.py (sote spectrum 是 condition 不是 token)
    - 任一 diff 结果出乎预期 (e.g. 文件实质内容不同),SA1 必须 raise 给用户决议,**禁止**擅自选

17. **必须** (v5 新增, Holdout cache 流程闸门):
    - Exp7 启动时 `/home/tcat/diffcsp_exp5_prime/data/` 不含 `holdout_structure_cache.pt`
    - **SA1 Phase 1-3 阶段禁止构建 holdout cache** (避免 holdout 提前接触污染)
    - **Phase 4 开始前**,SA1 必跑 `step3.0_build_holdout_cache.py`:
      ```bash
      cp /home/tcat/diffcsp_exp5_prime/code/step3/precompute_structure_cache_exp5_prime.py experiment7/step3/
      python step3/precompute_structure_cache_exp5_prime.py --split=holdout --L_VIRTUAL=20.0
      ```
    - 估计 5-10 分钟,产出 `experiment7/data/holdout_structure_cache.pt`
    - 必须 verify cache 文件含 3025 sample (与 `holdout_samples_v2.csv` 一致)

18. **必须** (v6 新增, 数据处理 verify 流程闸门, SA1 Day 0 决策):
    - SA1 Day 0 必须读 `/home/tcat/diffcsp_exp5_prime/code/step3/xas_local_dataset_v2.py` 代码 + 跑 5 个 sample 实测,verify §5.5.1 的 V1-V5 共 5 项数据 contract
    - 产出 `experiment7/data/dataset_contract_audit.json`,**没产出此 json 不许进 Phase 1 implementation** (流程闸门)
    - SA1 必须按 §5.5.2 决策矩阵 (V1-V5 verify 结果 → distance matrix 计算位置 + atom_types 排序方案) **严格选择**,不许擅自创新
    - **禁止** SA1 在没有 verify 的情况下假定 Exp5'/Exp6 dataset 的某种行为 (e.g. 假定 atom_types 已 sorted)。所有假定必须 verify
    - **禁止** SA1 选 Option B (改 dataset) — 这违反 SOP 12 "dataset 是用户决议红线项"。Option B 仅在 Option A 完全不可行 + 用户决议后才允许
    - **任一 V 项 verify 结果出乎预期** (e.g. frac_coords 不是 [-0.5,0.5] 也不是 [0,1]) 必须 RAISE,不许 patch
    - SA1 Day 0 完成 verify 后,**默认假设 §5.6.2 情况 B (V1=unsorted)**,即写 `collate_fn_resort`。若 V1 实测 = sorted,SA1 可省略 collate 但必须在 audit json 显式 mark `V1_atom_types_sorted: true`

---

*Exp7-MA 撰写,2026-05-10 v1 → v2 → v3 → v4 → v5 → v6 update 2026-05-10*
*基于:*
*- `EXPERIMENT5_SERIES_FINAL_REPORT.md` (Exp5'-MA + Exp5''-MA 撰写 2026-05-10) — Exp5 系列三阶段完整 lessons learned (14 条 ExpN+ 不变量级 + 6 份 errata + 三档 baseline ckpt + step5_3 7 项复合分 — v4 副套来源)*
*- **`EXP6_PROPOSAL_v8.md` (Exp6-MA 撰写 2026-05-01 → v8 2026-05-01) — Exp6 Transformer 路径的设计;Exp7 v3 沿用其 clone-first 原则;v4 沿用其 CPS 公式 (主套来源);v5 直接复用其 step1 全部产出 (5 个 json/py 文件)***
*- Gulrajani et al. 2017 NeurIPS (WGAN-GP) — 主算法依据,arxiv 1704.00028*
*- Miyato et al. 2018 ICLR (Spectral Normalization) — 稳定性设计,arxiv 1802.05957*
*- Heusel et al. 2017 NeurIPS (TTUR) — Two-time-scale update rule,arxiv 1706.08500*
*- Mirza & Osindero 2014 (Conditional GAN) — 任务范式基础,arxiv 1411.1784*
*- **PMC11501759 "Inverse design of structural color: finding multiple solutions via cGAN"** — v3 §1.4 任务本质参考 (one-to-many inverse design 与本任务范式直接对应)*
*- **eriklindernoren/PyTorch-GAN (GitHub, 16k+ stars)** — v3 主 clone 仓库 (取代 v1/v2 错误推荐的 TF 仓库)*
*- **gcucurull/cond-wgan-gp (GitHub)** — v3 cGAN + WGAN-GP 合并 cross-check 参考*
*- **christiancosgrove/pytorch-spectral-normalization-gan (GitHub)** — v3 SN 模块来源*
*- 用户 2026-05-10 round 1 决议: Exp5 系列 wrap up, 放弃 diffusion 路径, Exp7 GAN 与 Exp6 Transformer 同期并行验证*
*- 用户 2026-05-10 round 2 决议: 加入 curriculum learning,解决 Exp5 系列 EarlyStop 触发问题 (v2)*
*- 用户 2026-05-10 round 3 决议: 站在巨人肩膀上 — v3 改 clone-first 策略,取消手写 ~700 行 PyTorch 的 v1/v2 设计;同时强调任务本质是 inverse design + one-to-many*
*- **用户 2026-05-10 round 4 决议: 评估口径双套并报,主验收用 Exp6 v8 CPS (与 Exp6 架构对照可比),副指标用 Exp5' step5_3 (与 Exp5 系列历史 baseline 可比);v4 修复 v1-v3 的 SOP 1 公式锁定漏洞**
