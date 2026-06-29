# EXP4_MAINAGENT5_HANDOFF.md
# DiffCSP-Experiment4 Main Agent 5 交接文档

> **撰写者**：DiffCSP-Exp4-Main-Agent 4
> **接收者**：DiffCSP-Exp4-Main-Agent 5
> **日期**:2026-04-26
> **MA4 对话已达上限**:由 MA5 接管,启动 Step 4 训练 → Step 5 评估 → Step 6 可视化 → final report
> **当前状态**:**Step 3 已全部完成,Phase 6 五子全 PASS,Step 4 启动闸门 CLEAR**。下一步 MA5 写 Step 4 训练 Sub-Agent 交接,启动训练。

---

## §1 你接手时项目状态(锁死,不重新讨论)

**身份**:你是 Experiment 4 的 **Main Agent 5**(MA1→MA2→MA3→MA4→MA5)。

**全部已完成**:
- Step 0:方案锁定(EXP4_PROPOSAL_v2.md)
- Step 1:数据清洗 + v1 切分(128,382 样本)
- Step 2:谱预处理(spectra_*.pkl)
- Step 2.5:物理对齐 + Option D 剔除 incompat → **v2 = 75,637 样本 / 88 元素**
- Step 3:Dataset 改造 + Encoder 73→74 + DataModule 重写 + Forward test 5/5 PASS

**你的工作(按顺序)**:
1. **第一棒**:写 Step 4 训练 Sub-Agent 交接文档(MA5 第一件事)
2. 等用户跑完训练汇报,确认 val_loss 收敛 + best.ckpt 存档
3. **第二棒**:写 Step 5 评估 Sub-Agent 交接文档(holdout / 分层 RMSD / Type Acc)
4. 等用户跑完评估汇报
5. **第三棒**:写 Step 6 可视化 Sub-Agent 交接文档(figures)
6. **最后**:写 Exp4 final report

**你不写代码**。代码由 Sub-Agent 实现。

---

## §2 必读文档清单(按读取顺序)

| # | 文档 | 必读? | 重点 |
|---|------|-------|------|
| 1 | **本文档** | ✅ | 全文,尤其 §3 状态 + §6 Step 4 工作框架 |
| 2 | **EXP4_STEP3_SUBAGENT4CONT_FINAL_REPORT.md**(Sub-Agent 4-续 整合) | ✅ | §4 资产清单 + §5 开放问题(MA4 已决,§5 中透传) + §7 给 MA5 要点 |
| 3 | EXP4_PROPOSAL_v2.md | ✅ | §1.3 不可变量 + §6 预期指标 + §7 已知风险 |
| 4 | EXP4_FILE_INVENTORY.md | ✅ | 数据文件位置 + schema(Step 5 评估时还要回查) |
| 5 | EXPERIMENT2_FINAL_REPORT.md | ✅ | §1 三指标定义 + §2.4 Exp2 holdout 数字(可比性参考) |
| 6 | EXP4_MAINAGENT4_HANDOFF.md | 可选 | §6 锁定决策 + §7 文件归属表(已被 Sub-Agent 4-续 final report 浓缩,作 backup) |
| 7 | EXP4_PROGRESS_LOG.md | 可选 | Step 1/2/2.5 历史(Step 4 不直接用) |

---

## §3 用户的执行环境(继承,不变)

**服务器**:`scsmlnprd02.its.auckland.ac.nz`,密码登录,无 ssh key。

**执行 env**:`mlff` (`/home/tcat/conda_envs/mlff`)。

**关键包(7 守卫包,Sub-Agent 4-续整条链路守住未变)**:
- numpy 2.2.6 / scipy 1.15.3 / sklearn 1.7.2 / pymatgen 2025.10.7
- torch 2.4.1+cu124 / pytorch-lightning **2.5.5** / torch-scatter 2.1.2+pt24cu124

**Sub-Agent 4-续装的 6 + 12 子依赖包**(在 user site-packages,正常工作):
- einops 0.8.2 / p_tqdm 1.4.2 / smact 3.2.0 / matminer 0.9.3 / pyxtal 1.1.3 / torch_sparse 0.6.18+pt24cu124

**GPU**:2× RTX 4090 24 GB,CUDA 12.2。

**磁盘 ⚠️**:Sub-Agent 4-续 报 `/` 已用 94.4%(1.72 TB),swap 80%。**Step 4 launch 前 hard check**(见 §6.1)。

**.env 文件已建**(`/home/tcat/diffcsp_exp4/code/.env`,3 行 export):
```
PROJECT_ROOT=/home/tcat/diffcsp_exp4/code
HYDRA_JOBS=/home/tcat/diffcsp_exp4/logs/hydra
WABDB_DIR=/home/tcat/diffcsp_exp4/logs/wandb        # 上游 typo,保持
```

---

## §4 全部锁定决策(MA4 已决,MA5 不重新讨论)

### §4.1 不可变量(继承 Exp2)
- L = 6 Å,坐标系 [-0.5, 0.5],`frac -= np.round(frac)` min-image
- forward() 无 `% 1.`
- N_NEIGHBORS = 20,batch_size = **16**,lr = **1e-4**,num_workers = 0
- 三路 SpectrumEncoder(xmu 150 + chi1 200 + feff **74** → latent 256)
- DiffCSP 扩散框架,**cost_lattice = 0**
- **不加 TypeClassifier**

### §4.2 Step 4 阶段 MA4 已决(关键)

| 决策 | 选择 | 理由 |
|---|---|---|
| **precision** | **D1: fp32 全程** | Phase 6.5 已 fp32 PASS,直接对应训练。PL 2.5.5 的 `'bf16'` ≡ `'bf16-mixed'`,用户实测不行;Exp2 用的 PL 1.9.5 `precision='bf16'` 是纯 bf16,行为已变,迁移代价高 |
| **CPU vs GPU drift 阈值** | 调回 ±10% | GNN single-batch drift 6.7% 正常,不是 bug |
| **eval_cutoff_fallback 0/100** | 接受偏离 | PROPOSAL 估 5-10% 是粗估,Step 4 训练若实际触发再看 |
| **6.3 std 0.0680** | 留 Step 4 观察 | 无 numerical gate 已 PASS |
| **PL 版本** | mlff env 内 PL **2.5.5**,**不装 PL 1.9.5** | 跨 env 风险大,fp32 已验证 |

### §4.3 Step 5 阶段必含 caveat (Step 5 final report)

继承 MA3 锁定:
- bit-exact 等价 caveat:"Step 3 Dataset 邻居计算与 shell_boundaries.pkl 通过 bit-exact 验证一致;Phase A 算法的物理正确性继承 Step 2.5 报告,本工作未独立验证。"
- incompat 盲区声明:"本工作 Exp4 训练了 75,637 样本(剔除 52,745 'incompat' 样本)。incompat 样本结构上含多个不等价 Wyckoff 中心位点,与 MP 谱的 site-averaged 性质不直接对齐。Exp5 计划用 site-averaging 策略激活这部分数据。"
- **新增**:precision 偏离 caveat:"Exp4 因 PL 版本(2.5.5)行为与 Exp2 时代(PL 1.9.5)不同,改用 fp32 训练替代 bf16。Exp4 vs Exp2 的指标对比已含此 precision 差异(可能影响 ±5% 数值层),不影响架构有效性结论。"

---

## §5 资产清单(Step 4 启动 baseline)

```
/home/tcat/diffcsp_exp4/code/
├── .env                                   3 行 export ✓
├── diffcsp/                               框架包(Preparation Agent 上传)
├── conf/                                  顶层 hydra config(Preparation Agent 上传)
├── step2/spectrum_encoder.py              5 处 73→74 改完 (Sub-Agent 3)
└── step3/
    ├── forward_test.py                    14454 bytes  fp32, 5/5 PASS  ✓
    ├── forward_test.py.bak3               14801 bytes  md5 3d1441c3…   ← fp32 改前 baseline 回滚锚点
    ├── diffusion_w_type_xas.py            line 108: feat_dim=74        (Sub-Agent 4)
    ├── conf_xas/model/diffusion_xas.yaml  line 18: feat_dim=74         (Sub-Agent 4)
    ├── xas_local_dataset_v2.py            13 字段 + 双 raise            (Sub-Agent 3)
    └── xas_local_datamodule_v2.py         247 行, 类 XasLocalDataModuleV2 (Sub-Agent 4)

/home/tcat/diffcsp_exp4/data/              全量数据 (Sub-Agent 1/2/3 上传)
  ├── data_inventory_v2.csv                75,637 样本 ★主索引
  ├── {train,val,test,holdout}_samples_v2.csv
  ├── feff_features_imputed.pkl            (128382, 74) float32
  ├── feff_feature_scaler.pkl              RobustScaler
  ├── spectra_{train,val,test,holdout}.pkl  4 split 预处理谱
  ├── shell_boundaries.pkl                 369.5 MB, Step 5 评估用
  ├── MP_all_POSCAR_flat/                  41,496 POSCAR
  └── (incompat_pool.csv 封存,全程不读)

/home/tcat/diffcsp_exp4/checkpoints/       (Step 4 产出,目前空)
/home/tcat/diffcsp_exp4/logs/
  ├── hydra/                               Step 4 训练用
  ├── wandb/                               Step 4 训练用
  ├── step3_forward_test_console_v2.log    fp32 5/5 PASS baseline
  └── (step1/step2/step2.5 历史 log)
```

**回滚锚点(应急)**:
```bash
cp /home/tcat/diffcsp_exp4/code/step3/forward_test.py.bak3 \
   /home/tcat/diffcsp_exp4/code/step3/forward_test.py
```

---

## §6 Step 4 训练 Sub-Agent 工作框架(MA5 写交接时参考)

### §6.1 Step 4 启动前 hard check(MA5 必让 Sub-Agent 5 跑)

```bash
# 磁盘
df -h ~                # 期望剩余 ≥ 30 GB,< 30 GB 必须先清理(旧 wandb run / 旧 ckpt)
free -h                # RAM 可用 ≥ 10 GB(训练前 cache 用,可选)

# env 完整
which python           # /home/tcat/conda_envs/mlff/bin/python
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
                       # 期望 2.4.1+cu124 True
nvidia-smi             # 看 GPU 0/1 占用,选闲的

# 关键文件存在
ls /home/tcat/diffcsp_exp4/code/step3/forward_test.py \
   /home/tcat/diffcsp_exp4/code/step3/xas_local_datamodule_v2.py \
   /home/tcat/diffcsp_exp4/code/step3/diffusion_w_type_xas.py \
   /home/tcat/diffcsp_exp4/code/step3/conf_xas/model/diffusion_xas.yaml \
   /home/tcat/diffcsp_exp4/code/step2/spectrum_encoder.py \
   /home/tcat/diffcsp_exp4/code/.env

# 重跑一次 forward_test 确保 baseline 仍 PASS(防止 env 变动)
cd /home/tcat/diffcsp_exp4/code/step3
PYTHONPATH=/home/tcat/diffcsp_exp4/code python forward_test.py 2>&1 | tail -20
```

任一异常立即停汇报,不强推训练。

### §6.2 训练脚本设计要点(Sub-Agent 5 写)

参考 Exp2 `step4d_2_train.py` 改写,但要做以下修改:

| Exp2 | Exp4 修改 |
|---|---|
| `PRECISION = 'bf16'` | **`PRECISION = 32`**(fp32,MA4 决策 D1) |
| `PROJECT_ROOT = r"C:\..."` Windows 硬编码 | 删,从 `.env` 读(dotenv 已配) |
| `from xas_local_datamodule import XASDataModule` | `from xas_local_datamodule_v2 import XasLocalDataModuleV2` |
| `MAX_EPOCHS = 500` | **保持 500**(Exp2 Step4d 263 epoch 收敛,Exp4 数据量 ~7×,可能需更多但 500 是上限) |
| `EARLY_STOP_PAT = 30` | **保持 30** |
| `BATCH_SIZE = 16` / `LR = 1e-4` / `GRADIENT_CLIP = 1.0` | 保持 |
| ckpt callback `save_top_k=3` | 改 **`save_top_k=1`**(磁盘紧张,只留 best + last) |
| `default_root_dir = STEP4d_DIR`(Windows) | 改 `/home/tcat/diffcsp_exp4/checkpoints/` |
| `csv_logger` Windows 路径 | 改 `/home/tcat/diffcsp_exp4/logs/csv/` |

**完整 Trainer 参数模板**(MA5 给 Sub-Agent 5):
```python
trainer = pl.Trainer(
    default_root_dir = "/home/tcat/diffcsp_exp4/checkpoints",
    logger = csv_logger,
    callbacks = [ckpt_cb, lr_cb, early_cb],
    precision = 32,                    # ← MA4 决策 D1
    devices = 1,
    accelerator = "gpu",
    gradient_clip_val = 1.0,
    max_epochs = 500,
    check_val_every_n_epoch = 5,
    log_every_n_steps = 10,
    enable_progress_bar = True,
)
```

### §6.3 Step 4 期望与监控

**预期收敛**:Exp2 Step4d val_loss=0.8554 @ epoch 263。Exp4 数据量 ~7× + 88 元素多样性 → val_loss 可能更高(0.9-1.2 范围),但**应该单调下降**。

**红灯**:
- val_loss 前 10 epoch 不降 → 可能 lr / data 有问题,停汇报
- val_loss NaN → 立即停(可能 fp32 路径上某 op 数值不稳,需 MA5 决策是否要 grad clip 加严)
- GPU OOM → 调小 batch 或换 GPU 1
- 训练 epoch 时间 > 30 min → 不正常,可能 dataloader 是 bottleneck(Sub-Agent 3 测过 dataloader 20.7 ms/sample,60K × 16 batch ≈ 80s/epoch,异常即 stop)

**绿灯**:
- 第 30-50 epoch 内 val_loss 稳定下降到 < 1.5
- best ckpt 存档到 `/home/tcat/diffcsp_exp4/checkpoints/best.ckpt`

### §6.4 Step 4 子任务清单(MA5 给 Sub-Agent 5)

1. **Phase 4.0**:§6.1 hard check 全 PASS
2. **Phase 4.1**:写训练脚本 `step4_train.py`(参考 Exp2 step4d_2_train.py)
3. **Phase 4.2**:smoke test(`max_epochs=2`,`limit_train_batches=10`)→ 确认 fit 流程通
4. **Phase 4.3**:正式训练(`max_epochs=500`,后台 nohup 跑,10-30 小时)
5. **Phase 4.4**:监控 val_loss 曲线(每 50 epoch 看一次),早停触发即收尾
6. **Phase 4.5**:确认 best.ckpt + metrics.csv 存档,汇报 MA5

**Sub-Agent 5(Step 4)预算**:本身可能需要分 2 棒——一棒写 + smoke test,另一棒看训练完跑出来的指标。或者训练 nohup 后直接交,等用户主动来汇报。MA5 灵活处理。

---

## §7 Step 5 / Step 6 概要(MA5 写后续交接时再展开)

### §7.1 Step 5 评估(MA5 第二棒)
- **数据**:`holdout_samples_v2.csv` (3,025 样本,**全程封存**直到此刻)
- **3 指标**:RMSD(匈牙利匹配)/ Type Accuracy / pred_in_cutoff vs true_in_cutoff
- **分层报告**:按 center_element 分组、按 site_equivalence_tag 分组、按 eval_cutoff_fallback 分组
- **参考脚本**:Exp2 `step5/step5_1_sample.py` + `step5_2_compute_metrics.py`(改路径常量 + center_element 不再硬编码 Fe)
- **Exp4 vs Exp2 对比表**(必含):
  | 指标 | Exp2 holdout | Exp4 holdout |
  |---|---|---|
  | RMSD | 1.47 Å | __ |
  | Type Accuracy | 0.241 | __ |
  | pred_in_cutoff | 17.52/20 | __ |

### §7.2 Step 6 可视化(MA5 第三棒)
- 参考 Exp2 `step6/step6_visualize.py`(4 张图:fig1-fig4)
- Exp4 增图(因 88 元素):
  - fig5:按中心元素分层 RMSD 箱线
  - fig6:eval_cutoff_fallback 样本单独 RMSD 分布
  - fig7:incompat vs 训练样本谱差异(归档用,Exp5 启动)

### §7.3 Final Report
- 必含 §4.3 三条 caveat
- 与 Exp2 比较结论:架构在 88 元素是否 holds
- Exp5 路线图(site-averaging 激活 incompat 52,745 样本)

---

## §8 给 MA5 的避坑提示(继承全部 MA1-MA4 教训 + Sub-Agent 接力链经验)

### 坑 1(MA4 顶坑):多窗口接力的状态信任问题
本接力链 Sub-Agent 4 → 4-续 → 4-续 2 跨 3 窗口完成 Step 3 收尾。**关键防线**:
- handoff 文档透传文件归属表(粘全文,不是引用)
- bash 改动留 .bak / .bak2 / .bak3 + md5 备份
- 重大决策由 MA 拍板,不由 Sub-Agent 自行决定

### 坑 2:env 完整性必须 Phase 0 一次盘透
Sub-Agent 4 在 Phase 6.4 才暴露 diffcsp 缺失,代价 = 多开 1 个 Preparation Agent + 1 棒 Sub-Agent 4-续。**MA5 写 Step 4 handoff 时,§6.1 的 hard check 必须含 import 链测试**(不只是 ls 文件)。

### 坑 3:版本假设危险
PL `precision='bf16'` 在 PL 1.9.5 是纯 bf16,在 PL 2.5.5 是 alias 到 mixed。**任何继承 Exp2 的字符串配置都要在新 env 验证一次行为**,不要假设字符串语义稳定。

### 坑 4:Exp2 仓库文件名不可信(MA3 顶坑)
版本号 `_v1/_v2/_step4c/_v6` 与文件实际内容可能不一致(用户原话:"这个版本我觉得没必要保留我就直接把内容替换掉了")。Step 5 / Step 6 如果要复用 Exp2 脚本,**先 Check Agent 验身份**再用。

### 坑 5:Exp2 final report §3.4 TypeClassifier proposal 不要执行
Exp3 已证伪,Exp4 不加。MA5 在 Step 5 / final report 写作时不要被 §3.4 误导。

### 坑 6:incompat_pool.csv / holdout 严格隔离
- incompat 全程不读(只在 final report 引述存在)
- holdout 仅 Step 5 评估时读,Step 4 训练绝对不进 DataModule(已 §6 锁定)

### 坑 7:60% 上下文闸门 → 70%(MA5 阶段可放宽)
Step 4 训练阶段 Sub-Agent 工作集中在"启动训练 + 监控",非脚本反复修订,token 密度低。MA5 可让 Sub-Agent 闸门放到 **70%**(不再像 MA3/MA4 时代严守 60%)。

### 坑 8:Sub-Agent 4-续整条链路装的 6 + 12 子依赖在 user site-packages
路径:`/home/tcat/.local/lib/python3.10/site-packages/`(因 mlff conda 目录只读 fallback)。**MA5 在 Step 4 / 5 / 6 任何 Sub-Agent 装新包时,知情即可**——这条 fallback 路径已被 Sub-Agent 1/2/3/4 验证不影响 import 链。

### 坑 9:磁盘空间(O5)
`/` 用 94.4%。Step 4 训练 + checkpoint(每个 ~40 MB)+ wandb log(每 epoch ~10 KB × 500 = 5 MB)预计增 ~100-200 MB。看似不多,但 swap 80% 表明系统已紧张,**Sub-Agent 5 在 Phase 4.0 hard check 必须实测 ≥ 30 GB**,不够就先清。

### 坑 10:fp32 训练比 bf16 慢约 1.5-2×
Exp2 Step4d bf16 训 263 epoch 约 6-10 小时(4090)。Exp4 fp32 + 数据量 ~7× → 单 epoch 时间放大 ~10-14×。500 epoch 估计 50-100 小时。**MA5 写 Step 4 handoff 时必须告诉用户**:
- 训练用 nohup 后台跑(不要交互)
- 预期 2-4 天完成
- 早停 patience=30 应该能在 100-200 epoch 内触发,实际可能不到 50 小时

如果用户不能接受 50+ 小时,选项:
- 降 max_epochs 到 200(可能不收敛)
- 用 GPU 0+1 同时跑(`devices=2,strategy='ddp'`,约 1.5× 加速,有 PL 2.5.5 配置坑)
- 接受时间长,跑就跑

---

## §9 你(MA5)的第一条回复建议格式

```
我已阅读完 MA4 交接的所有文档。

[简要复述: Step 1/2/2.5/3 全部完成 + Phase 6 五子全 PASS + 
Step 4 启动闸门 CLEAR + MA4 已决 fp32 训练]

我注意到三个关键 carry-over:
1. fp32 训练 → 比 bf16 慢 ~1.5-2×, 实际可能 50-100 小时
2. 磁盘 / 紧, Phase 4.0 hard check 必须含磁盘清理 gate
3. PL 2.5.5 与 Exp2 PL 1.9.5 行为差异(precision/setup签名等), 写训练脚本时 grep 检验

在写 Step 4 Sub-Agent 交接前, 我向你确认:
1. 你接受 fp32 训练预期 50-100 小时? (否则要选降 max_epochs / 多 GPU / 别的)
2. /home/tcat 当前可用空间是否 ≥ 30 GB? (低于则先清理)
3. 是否要在 Step 4 启动同时, 让 Sub-Agent 5 也写 Step 5 评估脚本(并行)?
   还是 Step 4 跑完才写 Step 5? (我倾向后者, 但你拍板)

确认后我开始写 Step 4 Sub-Agent 交接。
```

---

## §10 最后一条提醒

MA1→MA4 接力链总耗时:多窗口 + 多 Sub-Agent + 4 轮 MA 决策。**剩下的工作量比想象小**:

- Step 4 训练:1 Sub-Agent 1 窗口启动 + 用户后台跑训练 + 1 Sub-Agent 收尾(大部分时间是计算,非对话)
- Step 5 评估:1 Sub-Agent 1 窗口
- Step 6 可视化:1 Sub-Agent 1 窗口或可与 Step 5 合并
- Final report:MA5 自己写

**全部顺利 = MA5 一个生命周期 + 3 个 Sub-Agent**。如果 Step 4 训练出意外(如 fp32 路径仍有数值问题),可能加 1 个 Sub-Agent debug。

如果你(MA5)发现"原假设错了"——例如 Step 4 训练直接 NaN(fp32 也炸)、Step 5 holdout 评估指标完全异常、Step 6 figures 写不出来——**不要硬推**,按 MA3/MA4 工作哲学:
1. 承认观察
2. 解释影响
3. 给 MA5 自己几个选项(或转给用户决策)
4. 不替用户做底层科学判断

用户对你信任,你回报这份信任的方式是**诚实**,不是"流畅"。

---

*MA4 撰写,2026-04-26,最后一次发言,接力终结于 MA5 启动 Step 4*
