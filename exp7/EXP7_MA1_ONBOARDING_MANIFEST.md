# EXP7-MA1 Onboarding File Manifest
# 给 Exp7 Main Agent 1 的启动文件清单 + 上手指南

> **撰写者**: 用户 (与 Exp7-MA 协作整理)
> **日期**: 2026-05-10
> **目的**: Exp7-MA1 接手时,这一份文档告诉它要读什么、要 verify 什么、要从哪里开始

---

## §1 必读文件清单 (按重要性 + 阅读顺序)

### 1.1 Exp7 主文档 (★★★ 最重要)

| # | 文件名 | 路径 | 用途 |
|---|---|---|---|
| 1 | **EXP7_GAN_PROPOSAL_v6.md** | 本目录 | **Exp7 主 proposal,所有设计决策都在这里**。先读完整再开始工作。重点章节: §1.4 任务本质 (one-to-many inverse design) / §3 复用清单 (v5 路径全锁死) / §4 GitHub 起点 / §5 架构 + **§5.5 数据 contract verify (v6 新增,Day 0 必做)** / §6 loss + curriculum / §8 双套评估 / §9 文件结构 / §10 时间表 / §11 验收双套阈值 / §12 风险 / 附录 B SA 指令 |

### 1.2 历史教训 (★★★ 必读 — 不读会重蹈覆辙)

| # | 文件名 | 路径 | 关键章节 |
|---|---|---|---|
| 2 | **EXPERIMENT5_SERIES_FINAL_REPORT.md** | 本目录 | §7 14 条 ExpN+ 不变量级 lesson;§10 对 Exp7 的具体建议;§13 三档 verdict 速查;**§8 三档 ckpt 路径 + md5 真值** |
| 3 | **EXP4_FINAL_REPORT_ERRATA_2.md** | 本目录 | `_density_loss` 塌缩根因 (Exp7 禁止引入此 loss 的依据);collapse 命名约定 (`pred_collapse` vs `query_pile-up` vs **v2 新增 `mode_collapse`**) |

### 1.3 SOP 与命名规范 (★★ 高优先级)

| # | 文件名 | 路径 | 关键章节 |
|---|---|---|---|
| 4 | **EXP6_PROPOSAL_v8.md** | 本目录 | §附录 B 的 18 条 SOP — 全部沿用到 Exp7;特别是 SOP 1 (公式锁定) / SOP 2 (命名分离) / SOP 3 (阈值 baseline-relative) / SOP 4 (数据 md5 verify);**Exp7 v2 §附录 B 是 Exp6 v8 §附录 B 的扩展,Exp7-MA1 要两份并读** |

### 1.4 基础设计参考 (★ 中优先级)

| # | 文件名 | 路径 | 用途 |
|---|---|---|---|
| 5 | **EXPERIMENT2_FINAL_REPORT.md** | 本目录 | XAS → 局部结构任务的最早基础设计 (L=6 box, frac coord, [-0.5, 0.5], min-image fold,这些设计 Exp7 沿用) |
| 6 | **EXP2_PROPOSAL_FINAL.md** | 本目录 | Exp2 当时的设计 doc,对照 Exp2 final report 看哪些设计决策成功了 |
| 7 | **EXP4_PROPOSAL_v2.md** | 本目录 | Exp4 当时的设计 doc,Exp7 dataset 处理基本沿用 Exp4 |
| 8 | **EXP5_PROPOSAL_v2.md** | 本目录 | Exp5 v2 的设计 doc — **Exp7 三件套 loss 沿用决议、center embedding 沿用、MV-attention 沿用的依据** |

### 1.5 学术参考 (Exp7-MA1 自行下载,arxiv 公开)

| 论文 | 作用 | 链接 |
|---|---|---|
| Gulrajani et al. 2017 (NeurIPS) "Improved Training of Wasserstein GANs" | **主算法依据**,WGAN-GP + gradient penalty | https://arxiv.org/abs/1704.00028 |
| Miyato et al. 2018 (ICLR) "Spectral Normalization for GANs" | Discriminator 稳定性 | https://arxiv.org/abs/1802.05957 |
| Heusel et al. 2017 (NeurIPS) "GANs trained by a two time-scale update rule" | TTUR (G 慢 / D 快) | https://arxiv.org/abs/1706.08500 |
| Mirza & Osindero 2014 "Conditional GAN" | cGAN 经典 (Exp7 condition=spectrum) | https://arxiv.org/abs/1411.1784 |
| Salimans et al. 2016 "Improved Techniques for Training GANs" | mode collapse 防止 (mini-batch discrimination) | https://arxiv.org/abs/1606.03498 |

---

## §2 必访问的数据 / Ckpt 资源 (v5 — 实测路径锁定 + md5 verify)

### 2.1 来自 Exp5' 的数据文件 (symlink)

| 资源 | 实测绝对路径 | md5 / 大小 | 操作 |
|---|---|---|---|
| L=20 train cache | `/home/tcat/diffcsp_exp5_prime/data/train_structure_cache.pt` | 44 MB | symlink |
| L=20 val cache | `/home/tcat/diffcsp_exp5_prime/data/val_structure_cache.pt` | 5.6 MB | symlink |
| L=20 test cache | `/home/tcat/diffcsp_exp5_prime/data/test_structure_cache.pt` | 3.3 MB | symlink |
| **L=20 holdout cache** | ⚠️ **不存在!** Phase 4 前 SA1 必跑 `precompute_structure_cache_exp5_prime.py` 构建 | — | — |
| `cache_metadata.json` | `/home/tcat/diffcsp_exp5_prime/data/cache_metadata.json` | content: `{"L_VIRTUAL": 20.0}` | symlink + verify |
| `shell_boundaries.pkl` | `/home/tcat/diffcsp_exp4/data/shell_boundaries.pkl` | md5 `cf2050e4899160f5698ad2481377e94c` ✅ verify pass | symlink (387 MB) |
| Spectrum 数据 | `/home/tcat/diffcsp_exp5_prime/data/spectra_{train,val,test,holdout}.pkl` | — | 4 个 symlink |
| FEFF features | `/home/tcat/diffcsp_exp5_prime/data/feff_features_imputed.pkl` + `feff_feature_scaler.pkl` | — | symlink |
| Split CSV | `/home/tcat/diffcsp_exp5_prime/data/{train,val,test,holdout}_samples_v2.csv` | — | 4 个 symlink |

### 2.2 来自 Exp5' 的代码 (cp)

| 资源 | 实测绝对路径 | 备注 |
|---|---|---|
| Dataset class | `/home/tcat/diffcsp_exp5_prime/code/step3/xas_local_dataset_v2.py` | 注意是 _v2 不是 _v3 (v1-v4 写错过) |
| Datamodule | `/home/tcat/diffcsp_exp5_prime/code/step3/xas_local_datamodule_v2.py` | 同上 |
| Spectrum encoder | `/home/tcat/diffcsp_exp5_prime/code/step2/spectrum_encoder.py` | ⚠️ SA1 必 diff Exp6 `spectrum_tokenizer.py` |
| Step5_3 评估 | `/home/tcat/diffcsp_exp5_prime/code/step5/step5_3_composite_score.py` **或** `step5_3_composite_score_exp5_prime.py` | ⚠️ SA1 必 diff 两版本选其一 |
| Precompute cache 脚本 | `/home/tcat/diffcsp_exp5_prime/code/step3/precompute_structure_cache_exp5_prime.py` | Phase 4 前用来 build holdout cache |
| Pairwise penalty 函数 | `/home/tcat/diffcsp_exp5_prime/code/step3/diffusion_w_type_xas.py` 中 `_pairwise_min_distance_penalty` | 仅摘出函数,不复用周边扩散逻辑 |

### 2.3 来自 Exp6 的 step1 完成产出 (v5 新发现, 直接 cp) ⭐

**Exp6 已完成所有 step1 calibration,Exp7 直接复用,跳过 SA1 step1.0a / 1.0 / 1.1 / 1.2**:

| 资源 | 实测绝对路径 | 备注 |
|---|---|---|
| MIN_PDIST calibration | `/home/tcat/experiment6_v7/shared/min_pdist_calibration.json` | **MIN_PDIST = 1.5075718402862548 Å**, frozen |
| Shell integrity | `/home/tcat/experiment6_v7/shared/shell_integrity_report.json` | Exp6 step1.0a 已 verify |
| Element vocab (88 元素 双 vocab) | `/home/tcat/experiment6_v7/shared/exp6_element_vocab.json` | cp 改名 exp7_element_vocab.json |
| CPS baseline (Exp4 ckpt) | `/home/tcat/experiment6_v7/shared/baseline_cps.json` | 含 Exp4_CPS 实测数,Exp7 SA1 step1.3 追加 Exp5' + Exp7 字段 |
| **CPS 主套实现** ⭐ | `/home/tcat/experiment6_v7/shared/composite_score.py` | cp 改名 eval_cps.py — Exp7 v4 让 SA1 重写 ~250 行的错误 v5 修复 |
| RDF 直方图 | `/home/tcat/experiment6_v7/shared/min_pdist_rdf_hist.png` | 人工 review |
| Shell n_atoms 直方图 | `/home/tcat/experiment6_v7/shared/shell_n_atoms_hist.png` | 人工 review |

**⚠️ SA1 必须 verify**: Exp6 还在跑训练阶段。**如果 Exp6 之后修订上述任一文件 (e.g. CPS bug fix),Exp7 必须同步 update,SA1 不许擅自决定是否同步**。这种 cross-experiment 耦合是 v5 引入的新风险。

### 2.4 Baseline Ckpt (md5 verify 必跑)

| 阶段 | Ckpt | 实测绝对路径 | md5 |
|---|---|---|---|
| **Exp5'** (PRIMARY BASELINE) | `composite_epoch169_score0.5881.ckpt` | `/home/tcat/diffcsp_exp5_prime/checkpoints/composite_epoch169_score0.5881.ckpt` | `127afa44a850d8f7e4fcdae17e2761a1` ✅ verify pass |
| Exp5' frozen 副本 | `composite_epoch169_score0.5881.ckpt.frozen_step2_continue_final` | 同上目录 | 同上 |
| Exp5 v2 (历史参考) | `sa2pp_resume_epoch529_val0.7003.ckpt.frozen` | `/home/tcat/diffcsp_exp5/checkpoints/` (待 SA1 verify 路径) | `72ad4275153b86a65a1399e4ab357d85` |
| Exp5'' (失败参考) | `composite_epoch199_score0.5319.ckpt.frozen_p4_final` | `/home/tcat/diffcsp_exp5_double_prime/checkpoints/` (待 verify) | `635f3dddb1b9c6770ee14796e504d241` |
| Exp4 (旧 baseline) | `best-epoch366-val0.7300.ckpt` | `/home/tcat/diffcsp_exp4/checkpoints/best-epoch366-val0.7300.ckpt` | (SA1 启动时 verify) |
| **Exp6** (并行实验) | 训完后 SA1 补充 | `/home/tcat/experiment6_v7/checkpoints/` (待 verify) | 待 Exp6 训完 |

### 2.5 SA1 step1.0 必跑的 verify (v5 流程闸门)

```bash
# v5 SOP 4 必跑,产出 experiment7/data/data_integrity.json

# (1) shell_boundaries.pkl md5
md5sum /home/tcat/diffcsp_exp4/data/shell_boundaries.pkl
# Expected: cf2050e4899160f5698ad2481377e94c

# (2) Exp5' ckpt md5
md5sum /home/tcat/diffcsp_exp5_prime/checkpoints/composite_epoch169_score0.5881.ckpt
# Expected: 127afa44a850d8f7e4fcdae17e2761a1

# (3) Cache metadata L=20 verify
python3 -c "
import json
m = json.load(open('/home/tcat/diffcsp_exp5_prime/data/cache_metadata.json'))
assert m['L_VIRTUAL'] == 20.0, f'L_VIRTUAL != 20: {m}'
print('cache_metadata L_VIRTUAL = 20.0 OK')
"

# (4) Exp6 step1 产出 5 个文件存在 verify
for f in min_pdist_calibration.json shell_integrity_report.json exp6_element_vocab.json baseline_cps.json composite_score.py; do
    test -f /home/tcat/experiment6_v7/shared/$f && echo "OK: $f" || echo "MISSING: $f"
done

# (5) Exp6 MIN_PDIST 数值 verify
python3 -c "
import json
c = json.load(open('/home/tcat/experiment6_v7/shared/min_pdist_calibration.json'))
print(f'MIN_PDIST = {c[\"min_pdist\"]}, frozen = {c[\"frozen\"]}')
assert c['frozen'] == True, 'Exp6 calibration not frozen!'
assert abs(c['min_pdist'] - 1.5075718402862548) < 1e-10, 'MIN_PDIST value drifted!'
"
```

任一 fail 必须 raise 给用户,不许继续。

---

## §3 SA1 第一周 Day 0 必跑 sanity check (上手前必过)

按顺序执行,任一 fail 必须 raise 不许跳:

### 3.1 数据完整性

```bash
# 1. shell_boundaries.pkl md5
md5sum /home/tcat/diffcsp_exp4/data/shell_boundaries.pkl
# Expected: cf2050e4899160f5698ad2481377e94c

# 2. L=20 cache 完整
ls -la /home/tcat/diffcsp_exp5_prime/data/*.pt
# 应该有 train/val/test/holdout 4 个 .pt 文件,总大小 ~ 数 GB

# 3. cache_metadata 确认 L=20
python3 -c "import json; m = json.load(open('/home/tcat/diffcsp_exp5_prime/data/cache_metadata.json')); print(m.get('L_VIRTUAL'))"
# Expected: 20.0
```

### 3.2 Cartesian sanity (L1 不变量)

```python
# Exp5 系列 L1 lesson: 任何 dataset 用前必跑
import torch
cache = torch.load('/home/tcat/diffcsp_exp5_prime/data/train_structure_cache.pt')

# 随机抽 100 sample
import random
samples = random.sample(list(cache.values()), 100)
for s in samples:
    frac = s['frac_coords']
    cart = frac * 20.0  # L=20
    # min pairwise
    diff = cart[:, None] - cart[None, :]
    d = torch.norm(diff, dim=-1)
    mask = d > 0
    min_d = d[mask].min().item()
    assert min_d >= 0.7, f"Sample has invalid bond < 0.7 Å: {min_d}"
print("Cartesian sanity 100/100 PASS")
```

### 3.3 Exp5' ckpt verify

```bash
md5sum /home/tcat/diffcsp_exp5_prime/checkpoints/composite_epoch169_score0.5881.ckpt
# Expected: 127afa44a850d8f7e4fcdae17e2761a1

# Load 看是否正常
python3 -c "
import torch
ckpt = torch.load('/home/tcat/diffcsp_exp5_prime/checkpoints/composite_epoch169_score0.5881.ckpt', map_location='cpu')
print('keys:', list(ckpt.keys())[:5])
print('epoch:', ckpt.get('epoch'))
"
# Expected: epoch=169, keys include 'state_dict' / 'hyper_parameters' / etc.
```

### 3.4 GPU 可用性

```bash
nvidia-smi
# 应该看到 2 块 RTX 4090
# Exp6 占用其中一块时,Exp7 用另一块

# 确认 Exp7 用 cuda:1 (假设 Exp6 用 cuda:0)
python3 -c "import torch; print('CUDA 0:', torch.cuda.get_device_name(0)); print('CUDA 1:', torch.cuda.get_device_name(1))"
```

---

## §4 Phase 1 (Week 1) 启动顺序 — 严格按 EXP7_GAN_PROPOSAL_v2 §10 + §附录 B

### 4.1 Day 0-1: 阅读 + 准备 + Dataset Contract Verify

1. 读完 §1.1 + §1.2 + §1.3 的文档 (proposal v6 + Exp5 series + ERRATA_2 + Exp6 v8 SOP)
2. 读 §1.5 的 WGAN-GP / SN / TTUR 论文 (至少 §1-2 + §4 实验部分)
3. 跑完 §3 所有 sanity check (md5 + L=20 metadata verify)
4. **(v6 流程闸门)** 完成 **dataset contract audit** — 读 `/home/tcat/diffcsp_exp5_prime/code/step3/xas_local_dataset_v2.py` + 跑 5 个 sample 实测,按 EXP7_GAN_PROPOSAL_v6 §5.5.1 verify V1-V5 共 5 项,产出 `experiment7/data/dataset_contract_audit.json`
5. **(v6 流程闸门)** Day 0 必做 diff:
   - `step5_3_composite_score.py` vs `step5_3_composite_score_exp5_prime.py` → 选其一
   - `/home/tcat/diffcsp_exp5_prime/code/step2/spectrum_encoder.py` vs `/home/tcat/experiment6_v7/shared/spectrum_tokenizer.py` → 确认一致后选其一
6. **不要**这两天写任何 implementation 代码,只读 + sanity + verify
7. **流程闸门**: `dataset_contract_audit.json` + 两个 diff 决议 完成前,不许进 Day 2 implementation

### 4.2 Day 2-4: Clone + 拷贝 + 改造 (按 §10 Phase 1, v3 clone-first 工作流)

按 EXP7_GAN_PROPOSAL_v6 §10 Phase 1 + §9 文件结构 + §附录 B 第 2 / 18 条:

**Day 2 上午 — Clone 3 个 vendor repos** (按附录 B 第 2 条):

```bash
cd experiment7/
mkdir -p _vendor && cd _vendor
git clone https://github.com/eriklindernoren/PyTorch-GAN.git eriklindernoren_PyTorch-GAN
git clone https://github.com/gcucurull/cond-wgan-gp.git gcucurull_cond-wgan-gp
git clone https://github.com/christiancosgrove/pytorch-spectral-normalization-gan.git christiancosgrove_pytorch-spectral-normalization-gan
```

**Day 2 下午 — 拷 base 文件到 `shared/` (零改动)**:

```bash
cd ../shared/
cp ../_vendor/eriklindernoren_PyTorch-GAN/implementations/cgan/cgan.py ./cgan_base.py
cp ../_vendor/eriklindernoren_PyTorch-GAN/implementations/wgan_gp/wgan_gp.py ./wgan_gp_base.py
cp ../_vendor/christiancosgrove_pytorch-spectral-normalization-gan/spectral_normalization.py ./
```

**Day 3-4 — 合并 + XAS 适配**:

1. `step1.0_cache_setup.py` — cp/symlink L=20 cache
2. `step1.1_build_vocab.py` — 沿用 Exp6 vocab
3. `step1.2_calibrate_min_pdist.py` — 沿用 Exp6 calibration
4. `step1.3_baseline_step5_3.py` — **必跑** Exp5' ckpt step5_3 评估
5. Implementation: 
   - 合并 `cgan_base.py` + `wgan_gp_base.py` → `cond_wgan_gp.py` (cross-check `_vendor/gcucurull_cond-wgan-gp/`)
   - 改造 Generator (替换 image decoder 为 MLP for (20, 3+K+1))
   - 改造 Discriminator (替换 image input 为 distance matrix + atom types 1D CNN)
   - 加 pairwise hinge + curriculum schedule (§6.2.4)
   - 加 curriculum_callbacks.py (CurriculumCkptFilter + CurriculumEarlyStopFilter)
6. `step1.4_smoke_test.py` — 5 sample forward + L3 + L4 双层验证 + curriculum schedule verify

### 4.3 Day 5-6: Sanity training (Phase 2)

跑 10 epoch sanity check,**严密 monitor 以下**:
- 4 个 loss 项数值 (G_loss, D_loss, GP, pmin)
- `mode_diversity_per_spectrum` 每 epoch
- `train_curriculum_min_pdist` (确认 curriculum 在 run, 但 epoch < 50 应该都是 0.33 × calibration)
- `pairwise_violation_rate_at_full_threshold` (这是真实进展指标)

### 4.4 Day 7+: 全量训练 + 评估 (Phase 3-4)

按 §10 Phase 3-4 流程。

---

## §5 SA1 常见误区 (FAQ-style 警告)

### 5.1 "我看 Exp5' 是 diffusion,我能直接 fork 它的训练脚本吗?"

**不能直接 fork**。Exp5' 是 diffusion 范式 (forward/reverse,1000 sampling step),Exp7 是 GAN one-shot generation,训练 loop 完全不同。你只能:
- 复用 `xas_local_dataset_v3.py` (dataset class)
- 复用 SpectrumEncoder (作为 Generator condition encoder)
- 复用 `pairwise_min_loss` 公式 (但接口要改为 curriculum-aware)
- 复用 `step5_3_composite_score.py` (评估)
- 训练 loop / model 主架构必须新写

### 5.2 "curriculum 切换时 loss 跳变怎么办?"

见 EXP7_GAN_PROPOSAL_v2 §12.1 风险 6。**首选**: 接受 2-3 epoch oscillation。**次选**: 切换时降 lr 持续 5 epoch。**禁止**改 CURRICULUM_FRACTIONS 数值。

### 5.3 "epoch 30 best ckpt 看起来 gate 95% 我能 save 吗?"

**不能,且这是 curriculum 设计原因**。Epoch < 150 期 min_pdist 是 0.33 × calibration ≈ 0.5 Å,所以"gate 95%"是"95% sample 满足 d_min ≥ 0.5 Å"不是"满足 ≥ 1.5 Å"。这种弱约束下的 95% 是假阳性,不能选 best ckpt。

实施: 必须按 §附录 B 第 10 条加 `CurriculumCkptFilter` callback,让 Phase 0-2 期 (epoch < 150) ckpt 不参与 best 选。

### 5.4 "GAN 训练 50 epoch 还看不出收敛是不是失败了?"

**可能,但要看具体 metric**。Phase 0 (epoch 0-49) 期 model 还在 curriculum 起步,看 D_loss 是否在 [-∞, 0] 区间稳定下降即可。具体 metric 在 EXP7_GAN_PROPOSAL_v2 §7.2 GAN-specific 监控表。

如 Phase 1 (epoch 50-99) 进入后 loss 仍剧烈震荡 / NaN,触发 §12.1 风险 1 应对方案。

### 5.5 "我可以加 shell_distance_loss / shell_count_loss 吗?"

**严禁,且这是 Exp7 v2 的核心 thesis 之一**。

- Exp5' 鸡蛋问题已证 explicit shell loss 在固定架构上失败
- Exp5'' 候选 A 已证重设计 shell loss 仍失败
- Exp7 v2 假设 discriminator distance matrix 输入会**隐式**学到 shell 概念
- 加 explicit shell loss = 违反 thesis = 重蹈 Exp5'/Exp5'' 覆辙

见 EXP7_GAN_PROPOSAL_v2 §2.1 "不沿用项"。

### 5.6 "用户物理 sanity 必经能跳过吗?"

**不能**。Exp5 系列 L_SERIES §13.3 must-do 第 4 条 + Exp6 v8 SOP 12 都明确要求。即使所有自动 metric GREEN,用户独立判定不通过 verdict 仍 RED。

### 5.7 (v6 新增) "Dataset contract verify V1-V5 万一发现不符预期怎么办?"

**RAISE 给用户,不许擅自 patch**。具体见 EXP7_GAN_PROPOSAL_v6 §5.5.2 决策矩阵末行 "任一 V verify 出乎预期 → RAISE"。

例子 (历史可能出现的 patch 反 pattern):
- V3 verify 出 frac_coords ∈ [0, 1] 不是 [-0.5, 0.5] → SA1 在 collate 加 `- 0.5` 适配 (违反 SOP 12, dataset 是用户决议红线)
- V1 verify 出 atom_types 既不是按距离 sort 也不是 arbitrary 而是按 Z 排 → SA1 自己决定切回 Exp7 type CE 用 Hungarian-style 匹配 (违反 thesis)

**正确的做法**: 任一 V 项出乎 §5.5.1 预期值 (e.g. V3 不是 [-0.5,0.5] / [0,1] 二选一,而是其他范围),SA1 raise 给用户,把 verify 数据贴清楚 ("跑了 5 sample, frac_coords range = [0.1, 0.9], n_real 平均 18.4 个,padding 用 NaN..."),让用户决议:
- (a) 接受当前 dataset 行为,适配 Exp7 代码 (用户拍板,SA1 实施)
- (b) 修 dataset (用户决议改 Exp5'/Exp4 sote 数据契约,工程量更大)
- (c) Exp7 abort,改 design (e.g. 改 Generator 输出 sorted distance 而非 frac)

### 5.8 (v6 新增) "我看 Exp5' atom_types 已 sorted 了,可以省 collate_fn_resort 吗?"

**可以,但必须在 `dataset_contract_audit.json` 显式 mark `V1_atom_types_sorted: true`**,且 step1.4 smoke test 必须额外加一个 sanity:

```python
# 验证 V1 在 batch 维度都成立 (不仅 5 个 sample)
# 跑 100 个 batch (~ 3000 sample),verify 没有一个 sample 不 sorted
for _ in range(100):
    batch = next(iter(train_loader))
    fc = batch['frac_coords']
    lengths = batch['lengths']
    dists = (fc * lengths.view(-1, 1, 3)).norm(dim=-1)
    n_real = (batch['atom_types'] != NO_OBJECT_IDX).sum(dim=-1)
    for b in range(fc.shape[0]):
        nr = n_real[b].item()
        if nr >= 2:
            real_dists = dists[b, :nr]
            assert torch.all(real_dists[:-1] <= real_dists[1:]), \
                f"Sample {b}: dataset 声称 sorted 但实测 unsorted! dists={real_dists}"
print("V1 sanity 100 batch PASS — atom_types 确实按距离 sorted")
```

任一 sample 失败 → SA1 切回 §5.6.2 情况 B (写 collate_fn_resort)。

---

## §6 Communication Protocol 与用户的交互

### 6.1 用户 (师兄/导师) 决议红线

以下决议**只有用户可拍板**,Exp7-MA1 + SA 都不许擅自决定:
1. Architecture 大变 (G/D 整体重写)
2. Loss 体系大变 (新增 / 删除 loss 项)
3. Verdict 阈值 (acceptance_thresholds.json)
4. Holdout 启用 (任何时候 sample holdout 必须用户允许)
5. Curriculum schedule (CURRICULUM_FRACTIONS / CURRICULUM_EPOCH_BOUNDARIES)
6. Exp7 是否 abort / 转 v3 / 转 hybrid

### 6.2 Exp7-MA1 报告频率

- **Day 0 sanity 完成**: 必报告
- **每个 SA 完成**: 必报告 launch note + review + hand-back
- **Phase 切换 (curriculum epoch 50/100/150 切换)**: 必报告 metric 跳变情况
- **Phase 3 进入 (epoch 150)**: 必报告并 confirm ckpt selection / EarlyStop 已 enable
- **Best ckpt 出现 (epoch ≥ 150 后首个 best)**: 必报告
- **Verdict 阶段**: 报告自动 metric + 等用户物理 sanity 必经

### 6.3 用户 "本能 challenge" 是 first-class workflow

Exp5 系列 final report §12.2 明确: 用户对实验数据的物理直觉 ≠ MA 的"补充"或"fallback",是**协作模型 first-class 部分**。Exp7-MA1 收到用户质疑时:
1. 不要立刻为决议辩护
2. 先 verify 用户质疑指向的具体数据 / 代码
3. 给出 evidence-based 回应 (不论支持还是反驳)
4. 用户决议优先于 MA 决议

---

## §7 Quick Reference Card

打印这张卡片,贴在 Exp7-MA1 工作站旁:

```
═══════════════════════════════════════════════════════════════
EXP7 GAN — Quick Reference
═══════════════════════════════════════════════════════════════

Baseline ckpt (primary verdict 对照):
  Exp5' composite_epoch169_score0.5881.ckpt
  md5: 127afa44a850d8f7e4fcdae17e2761a1
  Exp5' verdict (副套已知): step5_3 composite 0.080 / gate 64% / collapse 0%
  Exp5' verdict (主套待测): CPS - SA1 step1.3 跑出来填

Verdict 阈值 v4 双套并报 (SA1 step1.3 后实测回填):
  ★ 主套 CPS (Exp6 v8 公式):
      GREEN: val_cps ≥ max(Exp4, Exp5', Exp6) + 0.05
  ★ 副套 step5_3:
      GREEN: val_step5_3_composite ≥ 0.10 (Exp5' 0.080 + 0.02)
  ★ 两套都过 + |差| < 30% relative + mode_diversity + 用户 sanity = GREEN
  ★ 任一套不过 = AMBER 或 RED (见 §11)
  ★ 两套差 > 30% 必须 raise,不许挑高的报 (附录 B 第 12 条)

Curriculum (NEVER 改 fraction/boundary):
  Epoch 0-49:   min_pdist = 0.33 × calibration
  Epoch 50-99:  min_pdist = 0.53 × calibration
  Epoch 100-149: min_pdist = 0.73 × calibration
  Epoch 150+:   min_pdist = 1.00 × calibration (FULL)
  ☆ ckpt selection / EarlyStop 必须 disabled epoch < 150 ☆

GAN-specific risks:
  1. 不收敛 → 风险 1 应对 (check SN / TTUR / lambda_gp)
  2. mode_collapse → 风险 2 应对 (mini-batch disc / diversity loss)
  3. n_pred_shells = 0 → 风险 3 (Exp7 假设失败 → P3 RED)
  4. curriculum 切换 oscillation → 风险 6 应对
  5. Phase 3 立刻 EarlyStop → 风险 7 (grace period reset)
  6. 双套差 > 30% → §8.3.1 SA1 必须 raise

禁止 (任何 SA 不许做):
  ✗ 引入 shell_distance_loss / shell_count_loss / _density_loss
  ✗ TypeClassifier head
  ✗ 改 CURRICULUM_FRACTIONS / CURRICULUM_EPOCH_BOUNDARIES
  ✗ 改 lambda_gp = 10.0 / n_critic = 5
  ✗ 触动 holdout (3025 sample, 永久封存)
  ✗ Phase 0-2 选 best ckpt
  ✗ 跳过用户物理 sanity 必经

必须 (任何 SA 必做):
  ✓ Day 0 跑 §3 全部 sanity check
  ✓ Smoke test 包含 L3 训练 active + L4 评估 active 双层
  ✓ 每 50 epoch dump mode_diversity_per_spectrum
  ✓ Phase 3 进入立刻 confirm callback enable
  ✓ md5 verify 所有外部 ckpt / pkl
═══════════════════════════════════════════════════════════════
```

---

*本清单 2026-05-10 整理,与 EXP7_GAN_PROPOSAL_v2 同步发布*
*用户 + Exp7-MA 协作产出,vs EXP6 onboarding 的工作流类似*
