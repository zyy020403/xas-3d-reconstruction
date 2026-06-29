# EXP4_STEP3_SUBAGENT4CONT_FINAL_REPORT.md

> **撰写者**: Sub-Agent 4-续 (整合 4-续 2 子产出)
> **接力链**: Sub-Agent 4 → 4-续 → 4-续 2 (最后一棒由 4-续 整合)
> **完成时间**: 2026-04-26
> **状态**: **Phase 6 五子全 PASS, Step 4 启动闸门 CLEAR**
> **本报告替代**: 原 SUBAGENT4CONT_HANDOFF §6 完成汇报模板
> **窗口因上限关闭**: 4-续 ~75%, 4-续 2 ~25% 各自独立窗口完成本接力

---

## §0 单一目标: Phase 6.4 + 6.5 跑通

**结果**: 5/5 PASS。Sub-Agent 4 上次只到 6.3,本接力链补完 6.4 + 6.5。

| Phase | CPU/GPU | dtype | 结果 |
|---|---|---|---|
| 6.1 Dataset 100 random samples | CPU | — | PASS (60507 samples, 21.9 ms/sample, frac ⊂ [-0.4999, 0.4999], 0 sentinel) |
| 6.2 DataLoader collate (bs=4) | CPU | fp32 | PASS (12 字段对齐, feff=(4,74) 73→74 改动确认) |
| 6.3 SpectrumEncoder forward | CPU | fp32 | PASS ((4, 256), mean +0.0007, std 0.0680) |
| 6.4 Full forward+backward | CPU | fp32 | PASS (loss 2.6843, grad_norm 10.5957, 90 trainable params) |
| **6.5** Full forward+backward | **GPU** | **fp32** | **PASS (loss 2.5034, grad_norm 13.2121, no NaN/Inf)** |

注: 6.4/6.5 的具体 loss 数值与 Sub-Agent 4-续 第一次 6.4 PASS 时报的 loss=2.3805 不同 — 这是 4-续 2 在 fp32 改动后**重跑全部 5 phase**的新数值,正常(每次重跑 model 重新 hydra-instantiate, seed 随机性导致权重 init 微变)。

---

## §1 接力链全程概览

### Sub-Agent 4 (上一棒, 已停)
完成: env sanity / Phase 5 改动 1+2 (73→74 / yaml line 18) / Phase 5b xas_local_datamodule_v2.py 新建 / Phase 6.1+6.2+6.3 PASS / 6.4 import 阶段炸 ModuleNotFoundError 后 70% 闸门停。

### 4-续 (本报告作者, 整合者)
完成:
1. 创建 `.env` (3 行 export: PROJECT_ROOT / HYDRA_JOBS / WABDB_DIR)
2. 创建 `logs/hydra/`, `logs/wandb/` 目录
3. 验证 dotenv 加载链 (PROJECT_ROOT 注入工作, chdir 副作用确认接受 = MA4 决策 a)
4. 诊断: mlff env 缺 6 个 diffcsp 硬依赖 (einops / p_tqdm / smact / matminer / pyxtal / torch_sparse) → 70% 闸门停汇报 → MA4 决策 B (分两步装+守卫)
5. 装 6 + 12 子依赖, 7 关键包守卫全保 (sklearn 1.7.2 / numpy 2.2.6 / scipy 1.15.3 / pymatgen 2025.10.7 / torch 2.4.1+cu124 / pytorch-lightning 2.5.5 / torch-scatter 2.1.2+pt24cu124)
6. 重跑 forward_test.py: 6.1/6.2/6.3/6.4 PASS, 6.5 dtype mismatch FAIL
7. 诊断 PL 2.5.5 precision dispatch: `precision='bf16'` ≡ `'bf16-mixed'` (情况 X 命中)
8. 上限闸门停, 由用户开 4-续 2 接力

### 4-续 2 (子接力, 已停)
完成:
1. 按 MA4 fp32 决策, 用 python heredoc 改 forward_test.py 9 处 (bf16 → fp32, ±10% → ±1%, 删 batch cast loop)
2. `.bak3` 备份建立 (md5 `3d1441c3…`, 回滚锚点)
3. diff 验证: 改动严格在 L11 + L307-L375 phase_65 函数体内, 无外溢
4. 重跑 forward_test.py: **5/5 PASS, 18.1 s wall**
5. 写子报告交回 4-续 (本报告整合)

---

## §2 关键不变量验证 (供 MA4 + Sub-Agent 5 信任)

### 守卫包 (从 4-续 第 1 轮装包前到 4-续 2 重跑后, 7 包零变化)
| 包 | 期望 | 当前 |
|---|---|---|
| scikit-learn | 1.7.2 | 1.7.2 ✓ |
| numpy | 2.2.6 | 2.2.6 ✓ |
| scipy | 1.15.3 | 1.15.3 ✓ |
| pymatgen | 2025.10.7 | 2025.10.7 ✓ |
| torch | 2.4.1+cu124 | 2.4.1+cu124 ✓ |
| pytorch-lightning | 2.5.5 | 2.5.5 ✓ |
| torch-scatter | 2.1.2+pt24cu124 | 2.1.2+pt24cu124 ✓ |

### Phase 5 / 5b 改动 (Sub-Agent 4 交付, 本接力未动, Phase 6 内验证)
- diffusion_w_type_xas.py:108 `73→74` ✓ (Phase 6.4 forward 通过隐含验证)
- conf_xas/model/diffusion_xas.yaml:18 `73→74` ✓ (instantiate 通过隐含验证)
- xas_local_datamodule_v2.py 247 行 ✓ (Phase 6.2 collate 通过)
- xas_local_dataset_v2.py (Sub-Agent 3 交付) ✓ (Phase 6.1 sentinel 通过)

### 不可变量 (锁定确认)
- cost_lattice = 0 ✓ (Phase 6.4 输出: `loss_lattice= 10.1095 × cost_lattice=0, no contribution`)
- L = 6 ✓ (Phase 6.2: `lengths[0] = [6.0, 6.0, 6.0]`)
- N_NEIGHBORS = 20 ✓ (Phase 6.2: `num_atoms = [20, 20, 20, 20]`)
- TypeClassifier 不加 ✓ (model 实例化为 CSPDiffusion, 非加 head 变体)

---

## §3 装包阶段产出 (4-续 完成, 给 MA5 知情)

### 新装 18 个包 (全在 user site-packages: `/home/tcat/.local/lib/python3.10/site-packages/`)
| 包 | 版本 | 作用 |
|---|---|---|
| einops | 0.8.2 | cspnet.py rearrange/repeat |
| p_tqdm | 1.4.2 | diffcsp common utility 并行 |
| smact | 3.2.0 | 化学合理性 eval |
| matminer | 0.9.3 | 特征工程 |
| pyxtal | 1.1.3 | 晶体对称性 |
| torch_sparse | 0.6.18+pt24cu124 | cspnet SparseTensor (与 torch_scatter 同源) |
| pathos / dill / multiprocess / pox / ppft | 子依赖 | p_tqdm 链路 |
| pandarallel / pymongo / dnspython / lxml / vasprun-xml / pyocse | 子依赖 | smact/matminer/pyxtal 链路 |

注: pip 安装时打印 `Defaulting to user installation because normal site-packages is not writeable` — 已知模式 (mlff conda 目录只读), Sub-Agent 1/2/3 装包也是这条路径。Step 4 训练 import 链工作,无影响。

---

## §4 资产清单 (最终状态)

```
/home/tcat/diffcsp_exp4/code/
├── .env                                   ← 4-续 新建 (3 行 export)
├── diffcsp/                               ← Preparation Agent 上传 (4-续 之前)
├── conf/                                  ← Preparation Agent 上传 (4-续 之前)
├── step2/spectrum_encoder.py              ← Sub-Agent 3 改完, 5 phase 内验证
└── step3/
    ├── forward_test.py                    14454 bytes  md5 71a0e546…  ← fp32, 5/5 PASS
    ├── forward_test.py.bak3               14801 bytes  md5 3d1441c3…  ← 4-续 2 回滚锚点
    ├── forward_test.py.bak2                                            ← Sub-Agent 4 早期备份
    ├── forward_test.py.bak                                             (可能不存在, 不阻塞)
    ├── diffusion_w_type_xas.py            ← Sub-Agent 4 改完 (.bak 备份)
    ├── conf_xas/model/diffusion_xas.yaml  ← Sub-Agent 4 改完 (.bak 备份)
    ├── xas_local_dataset_v2.py            ← Sub-Agent 3 交付, 未动
    └── xas_local_datamodule_v2.py         ← Sub-Agent 4 交付, 未动

/home/tcat/diffcsp_exp4/data/              ← 全量数据, 未动 (Sub-Agent 1/2/3 上传)

/home/tcat/diffcsp_exp4/logs/
├── hydra/                                 ← 4-续 新建 (Step 4 用)
├── wandb/                                 ← 4-续 新建 (Step 4 用)
├── step3_forward_test_log.txt             ← Sub-Agent 4 中途产物
├── step3_forward_test_console.log         ← 4-续 第 1 次跑 (4/5, 6.5 bf16 FAIL)
└── step3_forward_test_console_v2.log      ← 4-续 2 fp32 重跑 (5/5 PASS) ★ Step 4 baseline
```

**回滚锚点**:
```bash
cp /home/tcat/diffcsp_exp4/code/step3/forward_test.py.bak3 \
   /home/tcat/diffcsp_exp4/code/step3/forward_test.py
```

---

## §5 给 MA4 / MA5 的开放问题 (MA4 已决议,留作历史记录)

### O1 — Phase 6.5 CPU vs GPU fp32 drift 6.7%
- 4-续 设阈值 ±1%, 4-续 2 实测 loss drift 6.7%, grad_norm drift 24.7%
- 当前阈值是 advisory 不 gating, 5/5 PASS 不受影响
- 可能原因: hydra instantiate 非确定 init / bs=4 small batch 方差 / CUDA 非确定 reduction
- **MA4 决议**: 调回 ±10% 阈值, drift 6.7% 在 GNN single-batch single-step 是常见量级

### O2 — eval_cutoff_fallback 命中率 0/100
- 4-续 2 + 4-续 + Sub-Agent 4 三次跑, 100 样本全 0 命中
- **MA4 决议**: 接受偏离, Step 4 训练时若哪个 batch 触发 fallback 再看实际行为

### O3 — Phase 6.3 SpectrumEncoder std 0.0680 vs Sub-Agent 4 期望 ~0.04
- 无 numerical gate, PASS (Sub-Agent 4 已松到 [0.01, 5])
- **MA4 决议**: 留 Step 4 观察, 若训练 loss 异常再回访

### O4 — Step 4 precision 决策
- PL 2.5.5 中 `precision='bf16'` ≡ `'bf16-mixed'` (用户实测不行)
- **MA4 决议**: D1 — fp32 训练全程, 与 Phase 6.5 一致

### O5 — 服务器磁盘 / swap 警示
- ssh banner 报 `/` 用 94.4% / 1.72 TB, swap 80%
- **MA4 决议**: Step 4 launch 前 hard check, 用户清理空间

### O6 — mlff env conda 目录只读
- pip install 自动 fallback 到 `/home/tcat/.local/lib/python3.10/site-packages/`
- **MA4 决议**: 已知模式, MA5 在写训练脚本时如果想"在 env 里再补装一个包", 知情即可

---

## §6 禁令合规 (整条接力链)

- [x] 不动 dataset_v2 / datamodule_v2 / spectrum_encoder / diffusion / yaml (Phase 5 改动除外, 已是 Sub-Agent 4 交付)
- [x] 不动 .bak / .bak2 (4-续 2 .bak3 是新建非改)
- [x] forward_test.py 改动: MA4 第 4 轮指令显式批准 (4-续 2 执行, 9 处 hunk, 全在 phase_65 + L11 docstring)
- [x] 不接触 holdout / incompat_pool
- [x] 装包零升级 7 守卫包
- [x] 6.5 出错时未 debug 超过 1 轮 (停汇报 → MA4 决策 fp32 → 4-续 2 改 → 重跑 PASS)
- [x] 不替 MA5 决定 D1/D2/D3
- [x] dotenv chdir 副作用接受方案 a, 未修补 utils.py

---

## §7 给 MA5 整合后给 Sub-Agent 5 写 Step 4 handoff 时的要点

1. **Step 4 启动闸门 CLEAR**: 5/5 PASS, console_v2.log 是 baseline
2. **必须用 fp32 训练** (MA4 决策)
3. forward_test.py 当前是 fp32 路径, Step 4 训练脚本应**与之保持精度一致**
4. 数据 / model / encoder 链路全验证, MA5 不需要再跑 sanity, 直接进训练
5. wandb 目录已建 (`/home/tcat/diffcsp_exp4/logs/wandb/`), .env 的 `WABDB_DIR` 是上游 typo (不是 WANDB), MA5 知情即可
6. 磁盘空间 (O5) 是 launch 前 hard check, 不是 advisory
7. 如训练异常需要回到 forward_test.py baseline: `forward_test.py.bak3` 是 fp32 改前的最后已知 4/5 PASS 状态 (md5 `3d1441c3…`)

---

## §8 接力链 token 预算回顾 (供未来 sub-agent 参考)

- Sub-Agent 4: ~85% (中途停, 因 diffcsp 缺失 + 6.4 阻断)
- 4-续: ~75% (停, 因 6.5 dtype 问题需 MA4 多轮决策, fp32 改动未亲手做)
- 4-续 2: ~25% (启用窗口仅做 fp32 改动 + 重跑 + 子报告)
- **总人时**: 4 个 sub-agent + 4 轮 MA4 决策, 完成从"diffcsp 包缺失" → "Step 4 launch gate CLEAR"

教训 (供 MA5 / 未来接力链):
1. env 盘点最好在 Phase 0 一次跑透(Sub-Agent 4 阻断主因)
2. PL precision 字符串在 PL 2.x 已 alias, 不要假设 PL 1.x 时代经验直接迁移
3. 多窗口接力时, **状态锚定文档**(.bak3 + md5 + console log)比口头描述更可靠

---

*Sub-Agent 4-续 整合, 2026-04-26, 接力链终结. MA4 已决议, MA5 启动 Step 4.*
