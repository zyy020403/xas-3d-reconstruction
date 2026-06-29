# EXP4_MAINAGENT4_HANDOFF.md
# DiffCSP-Experiment4 Main Agent 4 交接文档

> **撰写者**：DiffCSP-Exp4-Main-Agent 3
> **接收者**：DiffCSP-Exp4-Main-Agent 4
> **日期**：2026-04-25
> **背景**：MA3 对话上下文耗尽,由 MA4 接管。Step 3 Phase 0/3/4 已完成,Sub-Agent 3 卡满需立即停。下一步:写 Sub-Agent 4 handoff 推进 Phase 5 / 5b / 6,Phase 6 全过即可启动 Step 4。

---

## 1. 你是谁,你接手时项目什么状态

你是 Experiment 4 的 **Main Agent 4**(MA1 → MA2 → MA3 → MA4)。

**当前阶段**:Step 3 进行中,**Phase 0 / 3 / 4 已完成**,Phase 5 / 5b / 6 待 Sub-Agent 4 接管。

**你的工作**:

1. **第一棒**:写 **Sub-Agent 4 交接文档**(`STEP3_SUBAGENT4_HANDOFF.md`),让它接 Phase 5 / 5b / 6
2. 等用户跑完 Sub-Agent 4 汇报,确认 Phase 6 五子 phase 全 PASS,**才能启动 Step 4**
3. 然后写 **Step 4 交接文档**(训练阶段)
4. 等用户跑完 Step 4 汇报,确认 val_loss 收敛后,写 **Step 5 交接文档**(评估 + holdout)

**你不写代码**。代码由各 Sub-Agent 实现。

**Step 0 / 1 / 2 / 2.5 / Step 3 Phase 0/3/4 的所有决策已锁死**,不要重新讨论。

---

## 2. 必读文档清单(按读取顺序)

用户会传给你下面这些文档:

| # | 文档 | 必读? | 作用 |
|---|------|-------|------|
| 1 | **本文档**(EXP4_MAINAGENT4_HANDOFF.md) | ✅ | 你的工作全景图 + 完整 action/result |
| 2 | EXP4_PROPOSAL_v2.md | ✅ | 项目方案 v2(Step 0 锁死 + Step 2.5 修正) |
| 3 | EXP4_FILE_INVENTORY.md | ✅ | 数据文件位置 + schema |
| 4 | EXPERIMENT2_FINAL_REPORT.md | ✅ | Exp2 物理 + 4 版本演化(注意 §3.4 与 Exp4 矛盾,见 §8 避坑 5)|
| 5 | EXP4_STEP3_SUBAGENT_HANDOFF.md | ✅ | Step 3 Sub-Agent 总入口(MA3 写,Sub-Agent 1/2/3 都用过) |
| 6 | STEP3_SUBAGENT2_REPORT_TO_MA3.md | ✅ | Sub-Agent 2 中途汇报(包含 §1 仓库审计,内含废弃文件清单) |
| 7 | EXP4_PROGRESS_LOG.md | ✅ | Step 1/2/2.5 完整 action+result(MA1-2 时代) |
| 8 | EXP4_MAINAGENT3_HANDOFF.md | 可选 | MA2→MA3 接力文档 |
| 9 | STEP1_COMPLETION_REPORT.md | 可选 | Step 1 详细 |
| 10 | STEP2_5_FINAL_REPORT.md | 可选 | Step 2.5 详细 |

---

## 3. 用户的执行环境

- **本地**(Windows):Step 1/2/2.5 已全部完成,~700 MB 数据已上传服务器
- **服务器**(Step 3+ 在这里跑):
  - Host:`scsmlnprd02.its.auckland.ac.nz`,密码登录(无 ssh key 权限)
  - OS:Ubuntu 22.04.4 LTS
  - **执行 env**:`mlff`(`/home/tcat/conda_envs/mlff`),**不是** jhub_env(jhub_env 是 CPU-only,共享 env 已损坏)
  - GPU:2× RTX 4090 24 GB,CUDA 12.2(mlff 内 torch 2.4.1+cu124,cuda=True)
  - 关键包(MA3 已逐项 sanity):numpy 2.2.6 / pymatgen 2025.10.7 / torch 2.4.1+cu124 / pytorch-lightning 2.5.5 / sklearn 1.7.2 / pandas 2.3.3 / hydra 1.3.2 / omegaconf 2.3.0 / scipy 1.15.3 / joblib 1.5.2 / **torch_scatter 2.1.2+pt24cu124(MA3 决策 R2 修订版,env-local 安装到 mlff 内部,5/5 测试 PASS)**
  - 磁盘:根盘 1.72 TB 共享,~27-29 GB 可用(全服务器共享,不是用户 quota)
  - `/home/tcat/diffcsp_exp4/` 目录已搭好

---

## 4. Step 0 / 1 / 2 / 2.5 简要回顾(详见 EXP4_PROGRESS_LOG)

### 4.1 数据集(锁死)

- 原始 133,718 → Step 1 清洗后 128,382 → Step 2.5 剔除 incompat → **v2 = 75,637 样本 / 35,445 mp_ids / 88 元素**
- 52,745 incompat 封存到 `incompat_pool.csv`,**Exp4 全程不读**(Exp5 留)
- v2 split:train 60,507 / val 7,624 / test 4,481 / holdout 3,025

### 4.2 关键产物(已上传服务器 `/home/tcat/diffcsp_exp4/data/`)

- `data_inventory_v2.csv` 33.5 MB(主索引,75,637 × 15 列)
- `{train,val,test,holdout}_samples_v2.csv`(共 4 个,4 列 schema)
- `feff_features_imputed.pkl` 40.3 MB(DataFrame, index=sample_name, 128382 × 74 float32, 零 NaN)
- `feff_feature_scaler.pkl` 1.6 KB(RobustScaler, sklearn 1.6.1 训, 1.7.2 unpickle warning 已 catch_warnings 静默)
- `spectra_{train,val,test,holdout}.pkl`(4 split 已预处理,xmu 150 + chi1 200 float32)
- `shell_boundaries.pkl` 369.5 MB(全 128K 样本,9 字段 schema,Step 5 评估用)
- `MP_all_POSCAR_flat/` 41,496 个 POSCAR(`mp-{ID}_POSCAR` 命名)

### 4.3 sample_name 格式(全 4 数据源同格式)

`mp-{ID}__mp-{ID}-EXAFS-{Element}-K`(注意 `mp-{ID}__mp-{ID}` 重复,by design)

---

## 5. Step 3 详细 action+result(MA3 时代,你必须知道)

### 5.1 环境路径决策(关键)

**踩过的坑**:用户最初在 `(jhub_env)` prompt 跑 pip freeze,看到的版本来自 user-level Python 3.10 包(jhub_env 是 Python 3.12,完全不同),误以为 jhub_env 是 ML env。

Sub-Agent 实际 import 验证暴露真相:
- jhub_env: torch 2.4.1.post100 (CPU-only conda-forge build), cuda=False
- sklearn 在 jhub_env 内 ABI broken (numpy.dtype size 96 vs 88)

MA3 让用户跑 `conda env list` + 各 env `python -c "import torch"`,发现 mlff env 是干净的 GPU 路径,锁定为执行环境。

**教训**:**任何 env 验证必须走 `python -c "..."` 而非 `pip freeze`**。

### 5.2 Sub-Agent 1(MA3 第一棒)

**Action**:收到 EXP4_STEP3_SUBAGENT_HANDOFF v1。

**Result**:
- Phase 2 仓库审计完成,识别 step3/ 实际文件结构(分子目录,不是扁平)
- 发现 EXP2 仓库实际 fork 源:Step4d 时代的 `xas_local_dataset_L6.py` + `xas_local_datamodule.py` + `diffusion_w_type_xas.py`(无后缀)
- 提出 3 个决策点 A / B / C

**MA3 拍板**:
- A=R2 修订版:torch_scatter env-local 装到 mlff 内部(不加 `--user`),命令 `pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.1+cu124.html`,5/5 测试 PASS
- B=B1:重写 datamodule 为 `xas_local_datamodule_v2.py`,类名 `XasLocalDataModuleV2`
- C=C3 修订版:frac sentinel 触发 raise RuntimeError(不 clamp 不返回 None),epsilon=1e-6,Phase 6.1 改成 100 随机样本遍历

**Sub-Agent 1 上下文耗尽**,转 Sub-Agent 2。

### 5.3 Sub-Agent 2(MA3 第二棒)

**Action 1**:POSCAR 上传

服务器原本有用户传到 `/home/tcat/mp-9_POSCAR` 的旧路径,但实际是无关化学计算残留。Sub-Agent 用 `find` 确认 zero `mp-XXXX_POSCAR`,**POSCAR 必须从 Windows 重传**。

**Result**:Windows 端 `tar -czf` 打包(18.26 MB,gzip 压缩比 ~5×)→ scp(秒传)→ 服务器 `tar -xzf`。

POSCAR 数 41,496(打包 41,497 文件 + 1 目录 entry = 41,498,删 missing_poscar_list.csv 后 41,496)。

**MA3 handoff §1.4 写 ~41,431 是错的**(那是 v1 inventory 切分后的 mp_id 数,不是磁盘 POSCAR 文件数)。**真值 41,496**,POSCAR 是 v1 35,445 mp_ids 的超集 + 65 个 Step 1 处理过程中被剔除整 mp_id 但 POSCAR 还在的样本。Sub-Agent 3 Phase 0.5 已确认 v2 unique mp_ids ⊂ POSCAR(missing=0)。

**Action 2-4**:Phase 0.2 / 0.3 / 0.3b / 0.3c / 0.4 完成

| Phase | 任务 | Result |
|-------|------|--------|
| 0.2 | numpy 1.x 别名 grep | 零命中(主 grep + 自补 `_` 后缀分支也零命中)|
| 0.3 | sklearn RobustScaler unpickle | 74 维对齐,sklearn 1.6.1→1.7.2 unpickle 触发 InconsistentVersionWarning 但功能 OK(transform(zeros) 输出无 NaN/Inf)|
| 0.3b | feff DataFrame 真实维度 | (128382, 74) float32, 三件套全 74 ✅ |
| 0.3c | DataFrame index/columns 结构 | index=sample_name(object, unique, 零 null), 格式 `mp-{ID}__mp-{ID}-EXAFS-{Element}-K`, 零 NaN |
| 0.4 | Lightning import grep | 21 处全 `pytorch_lightning`,零 `lightning`(dev) → 用 PL 2.5.5 |

**Action 5**:提交 7 个待决策点(7.1-7.7)。

**MA3 拍板**(全部 7 项):
- 7.1 numpy grep 模式补丁 → errata,handoff 修订时加
- 7.2 encoder 改动 → **B**(default + grep instantiate + docstring + 完成后 grep "73" 零残留)
- 7.3 sklearn warning → **D**(局部 catch_warnings + 注释说明 ABI 兼容)
- 7.4 sample_name 跨源对齐 → **B**(先 inspect 4 源格式 + 顺便算 set 数学,任一不一致立刻停)
- 7.5 DataFrame.loc → **A 现在 + B 后置**,`__init__` 末尾加 1000 次 timing benchmark
- 7.6 POSCAR 41,496 vs 41,431 → **A**(接受 41,496 真值,Phase 0.5 顺便验 missing=0)
- 7.7 missing_poscar_list.csv → **A**(已删 move on)

**Sub-Agent 2 上下文耗尽**,转 Sub-Agent 3。

### 5.4 Sub-Agent 3(MA3 第三棒,最后一棒,当前已停)

#### Phase 0.5(全过 ✅)

5 道闸:
- G1: v2 splits sum = 75,637 ✓
- G1-strong: incompat_pool = 52,745 ✓
- G2: v2 + incompat = 128,382 = feff DataFrame 行数 ✓
- G3: 4 源 sample_name 100% 同格式(无需映射函数)✓
- G4: v2 unique mp_ids ⊂ POSCAR on-disk(missing=0)✓

#### Phase 0.1(全过 ✅,bit-exact)

10/10 max|diff|=**0.000e+00 Å** 严格相等(5 single_site + 5 multi_site)。

**结论**:pymatgen `SpacegroupAnalyzer(symprec=0.1)` + `get_neighbors(r=10.0)` + first-matching-site 在 Linux 工作正常,**与 shell_boundaries.pkl bit-exact 对齐**。

**Phase 3 Dataset 用 pymatgen 在线计算,不启用 brute-force fallback,`exp4_utils/neighbors.py` 不创建**。

**认识论 caveat**(Sub-Agent 3 提出,MA3 已记入 Step 5 论文 caveat 段):bit-exact 等价 ≠ 物理正确性。如果 Phase A 算错了 Phase 0.1 会一起错。Phase A 物理正确性归 Step 1/2.5 阶段责任,Step 3 不独立验证。

#### Phase 3:Dataset 改造(完成 ✅)

**交付**:`/home/tcat/diffcsp_exp4/code/step3/xas_local_dataset_v2.py`

**关键设计**(MA3 已批 Q1-Q4 + L_VIRTUAL=6.0 + 6 推断点 + 双补充防御 raise):

return dict 12 字段(权威 schema):
```python
{
    "xmu":           Tensor (150,) float32,    # Step 2 预处理后
    "chi1":          Tensor (200,) float32,    # Step 2 预处理后
    "feff":          Tensor (74,)  float32,    # scaler.transform 后(catch_warnings 静默 InconsistentVersionWarning)
    "frac_coords":   Tensor (20, 3) float32, ∈ [-0.5, 0.5],
    "atom_types":    LongTensor (20,) ∈ [1, 109],
    "sample_name":   str,
    "mp_id":         str,
    "center_element": str,
    "eval_cutoff":   float,                    # Step 5 用
    "eval_cutoff_fallback": bool,              # Step 5 用
    "n_center_sites": int,                     # 审计用
    "site_equivalence_tag": str,               # Step 5 分组用
}
```

不可变量:
- L_VIRTUAL = 6.0 / N_NEIGHBORS = 20 / CUTOFF_R = 10.0 / SYMPREC = 0.1
- frac 流程:中心 cart → 邻居 cart - 中心 cart → /L_VIRTUAL → `frac -= np.round(frac)` → frac sentinel raise(epsilon=1e-6)
- atom_types/frac_coords **不含中心**(中心在虚拟晶格原点 0,通过 center_element 字段透传)
- 邻居数 < 20 → **raise RuntimeError**(不 padding)
- 显式 `np.argsort(dists)[:20]`,不依赖 pymatgen 返回顺序
- center_idx StopIteration → 显式 RuntimeError(prim 内位点元素列表打出来)

双 raise 防御:
- A 类(init defensive): `__init__` 抽 5 样本对齐 4 源,miss 立刻 RuntimeError
- B 类(§2.C frac sentinel): `__getitem__` frac 越界立刻 RuntimeError

shell_info **不进嵌套字段**,只透传标量(eval_cutoff / eval_cutoff_fallback / n_center_sites)给 Step 5。

**测试结果**:10/10 unit sanity PASS。

**Benchmarks**(`__init__` 末尾,记入 forward_test_log):
- feff.loc avg: **58.16 µs/sample**(< MA3 200 µs 阈值,初版不需 dict cache)
- POSCAR + SGA avg: **20.70 ms/sample**(Step 4 训练 60K × 50 epoch 该值是 dataloader 主开销)

#### Phase 4:SpectrumEncoder(完成 ✅)

**改动**:`step2/spectrum_encoder.py` 5 处 73 → 74

| 行 | 上下文 |
|----|-------|
| 2 | docstring header |
| 28 | docstring forward 描述 |
| 37 | docstring `__init__` 参数 |
| 41 | code: `def __init__(..., feat_dim=73, ...)` |
| 80 | docstring forward Args |

**完成后 grep sanity**(命令 `grep -rn "feat_dim\s*=\s*73"`):

| 残留位置 | 解读 |
|---------|------|
| `step3/diffusion_w_type_xas.py:108` | **Phase 5 target,Sub-Agent 4 改** |
| `step3/diffusion_w_type_xas_v1.py:103` | 废弃 |
| `step3/diffusion_w_type_xas_v2.py:99` | 废弃 |
| `step3/step3_5_e2e_forward_test.py:84` | 废弃 |
| `step3/step3_5_nan_debug.py:58` | 废弃 |

5 处全在合理位置,Phase 4 收尾 ✅。

#### Sub-Agent 3 顺手发现(透传给 MA4)

**3 个 Dataset 废弃文件**(MA3 上次没列):
- `step3/xas_local_dataset_step4c.py` = Step4c (L=12) 废弃
- `step3/xas_local_dataset_v6.py` = Step4b bug fix 走死,废弃
- `step3/xas_local_dataset_L6.py` = Step4d (L=6) 即 EXP2 权威版,**已被 Exp4 的 xas_local_dataset_v2.py 替代**

**step2/ 下含 73 的脚本**(Sub-Agent 3 不动,透传):
- `step2/step2_2_preprocess_validation.py`(2 处 73)— Step 2 验证脚本,不复用
- `step2/step2_4_encoder_test.py`(3 处 73)— Step 2 encoder 旧测试,不复用(Phase 6 写新 forward_test.py)
- `step2/spectrum_preprocessor.py`(1 处 73)— Step 2 谱预处理工具,不复用(已被 spectra_*.pkl 替代)

#### Sub-Agent 3 已停

用户报告"非常卡"(Sub-Agent 3 自报 62-65% 但实际更高)。**MA3 决定立即停 Sub-Agent 3**,handoff §6 Sub-Agent 4 的工作由 MA4 自己写新 handoff。

---

## 6. 全部锁定决策清单(MA4 必须沿用,不要重新讨论)

### 6.1 不可变量(继承 Exp2,绝对不能改)

- L = 6 Å(虚拟晶格)
- 坐标系 [-0.5, 0.5],`frac -= np.round(frac)` min-image
- forward() **无** `% 1.`
- N_NEIGHBORS = 20
- batch_size = 16, lr = 1e-4, bf16
- 三路 SpectrumEncoder(xmu 150 + chi1 200 + feff **74** → latent 256)
- DiffCSP 扩散框架,cost_lattice = 0
- **不加 TypeClassifier**(Exp3 已证伪,Exp2 final report §3.4 的 proposal **不要执行**)

### 6.2 Step 3 阶段锁定(MA3 时代)

| 决策 | 内容 |
|------|------|
| Q1 | 12 字段 return dict schema(三路谱并行,frac_coords 20 邻居不含中心,shell 仅标量透传)|
| Q2 | 双 raise(init defensive + frac sentinel)|
| Q3 | 不加 cache 初版,Step 4 profile 后再决定 |
| Q4 | 每 split 一个 Dataset 实例,holdout 不进 DataModule |
| Q5 | step3/ 5+3 个废弃文件来自 Exp2 仓库 scp 上传时整体带过来 |
| Q6 | feat_dim 73→74 仅改 spectrum_encoder.py 5 处 + diffusion_w_type_xas.py(无后缀):108 |
| Q7 | Phase 5 target = diffusion_w_type_xas.py(无后缀版,Step4d 时代权威)|
| 7.1 | numpy grep 加 `_` 后缀分支(errata,handoff 后续修订加)|
| 7.2 | encoder 改 default + grep instantiate + docstring 全改 |
| 7.3 | sklearn warning 局部 catch_warnings(InconsistentVersionWarning + ABI 注释)|
| 7.4 | Phase 0.5 先 inspect 格式 + set 数学 + missing 验证 |
| 7.5 | df.loc 现在 + dict cache 后置 + timing benchmark |
| 7.6 | POSCAR 41,496 真值,v2 mp_ids ⊂ on-disk(missing=0 已确认)|
| 7.7 | missing_poscar_list.csv 已删 move on |
| L_VIRTUAL | 6.0 |
| 6 推断 | 邻居数<20 raise(不 padding); fixed CUTOFF=10.0; SGA primitive |
| 双防御 | 显式 argsort + center_idx StopIteration → RuntimeError |
| Phase 6.1 | 100 random samples 遍历(不是单 ds[0])|

---

## 7. **Exp2 仓库文件归属总表(最关键的一段)**

> **目的**:让任何后续 Sub-Agent 零推断直接知道每个文件该不该动。
> **来源**:Sub-Agent 1 仓库审计 + EXP2 FINAL §2.3 4 版本演化 + Sub-Agent 3 grep 确认 + MA3 拍板 Q5/Q6/Q7。
> **使用方式**:MA4 写 Sub-Agent 4 handoff 时**必须完整粘贴本表**,避免 Sub-Agent 4 重蹈"无主文件踩坑"。

### 7.1 step3/ 目录(Step 3 主战场)

| 文件 | EXP2 阶段 | 状态 | Exp4 命运 |
|------|----------|------|----------|
| `xas_local_dataset_L6.py` | Step4d (L=6,最终有效) | EXP2 权威 | 不动(已被 `xas_local_dataset_v2.py` 替代)|
| `xas_local_dataset_step4c.py` | Step4c (L=12,失败) | 废弃 | 不动,保留作 Exp2 历史快照 |
| `xas_local_dataset_v6.py` | Step4b bug fix 走死 | 废弃 | 不动,保留 |
| `xas_local_datamodule.py` | Step4d datamodule | EXP2 权威 | **Phase 5b Sub-Agent 4 重写为 v2** |
| `diffusion_w_type_xas.py`(无后缀) | Step4c/4d 共用扩散逻辑 | EXP2 权威 | **Phase 5 Sub-Agent 4 改 4 项** |
| `diffusion_w_type_xas_v1.py` | Step4 (L=12 + `%1.` bug) | 废弃 | 不动,保留 |
| `diffusion_w_type_xas_v2.py` | Step4c density loss 实验 | 废弃 | 不动,保留 |
| `step3_5_e2e_forward_test.py` | Step4 时代 e2e debug | 废弃 | **Phase 6 不复用,Sub-Agent 4 写新 forward_test.py** |
| `step3_5_nan_debug.py` | Step4 时代 NaN debug | 废弃 | 不复用 |

### 7.2 step2/ 目录(Sub-Agent 3 透传)

| 文件 | 状态 | Exp4 命运 |
|------|------|----------|
| `spectrum_encoder.py` | EXP2 权威 + Exp4 已改 5 处 73→74 | **已用,Phase 4 完成** |
| `step2_2_preprocess_validation.py` | Step 2 验证脚本(含 2 处 73) | 不动,不复用 |
| `step2_4_encoder_test.py` | Step 2 encoder 旧测试(含 3 处 73) | 不动,不复用 |
| `spectrum_preprocessor.py` | Step 2 谱预处理工具(含 1 处 73) | 不动,不复用(已被 spectra_*.pkl 替代)|

### 7.3 Exp4 新建文件

| 文件 | 阶段 | 状态 |
|------|------|------|
| `step3/xas_local_dataset_v2.py` | Sub-Agent 3 Phase 3 | **完成**,10/10 unit sanity PASS |
| `step3/xas_local_datamodule_v2.py` | 待 Sub-Agent 4 Phase 5b | 未开始 |
| `step3/forward_test.py`(或类似名) | 待 Sub-Agent 4 Phase 6 | 未开始 |

### 7.4 不变文件(DiffCSP 框架核心,绝对不动)

继承 Exp2,Step 1/2/2.5/3 都未触碰:
- `gnn.py`、`cspnet.py`、`diff_utils.py`(扩散数学层)
- `run.py`(训练入口,Step 4 时再判断要不要改路径常量)
- 所有数学层、采样器、loss 函数

---

## 8. Sub-Agent 4 待办(Phase 5 / 5b / 6)

### 8.1 Phase 5:`diffusion_w_type_xas.py` 4 项改动

target: `step3/diffusion_w_type_xas.py`(无后缀版)

1. **line 108**: `feat_dim=73` → `feat_dim=74`
2. **数据路径常量**: 改成 `os.environ.get("EXP4_DATA_DIR", "/home/tcat/diffcsp_exp4/data")` 形式
3. **Dataset import**: `from xas_local_dataset_L6 import XASLocalStructureDataset` → `from xas_local_dataset_v2 import XasLocalDatasetV2`
4. **DataModule import**: 改成新 `from xas_local_datamodule_v2 import XasLocalDataModuleV2`

**不动**:
- `cost_lattice = 0`(必须保持)
- 扩散数学(β schedule、DDPM/DDIM)
- forward / training_step / validation_step 主体
- **不加 TypeClassifier**

### 8.2 Phase 5b:重写 `xas_local_datamodule_v2.py`

按 MA3 决策 B1 6 边界:
1. 命名:`xas_local_datamodule_v2.py`,类名 `XasLocalDataModuleV2`
2. import 改:`from xas_local_dataset_v2 import XasLocalDatasetV2`
3. 路径常量:从环境变量读
4. 删 v1 73 维 FEFF CSV 路径常量(Exp4 用 feff_features_imputed.pkl,无需此 CSV)
5. DataModule 结构不动:setup / train_dataloader / val_dataloader / test_dataloader
6. PL 2.5.5 兼容:grep `setup()` 签名是否依赖 1.x(`if stage is None`)

**holdout 不进 DataModule**(Step 5 单独实例化)。

### 8.3 Phase 6:写新 forward test

target: `step3/forward_test.py`(新建,**不复用** `step3_5_*` 废弃文件)

按 EXP4_STEP3_SUBAGENT_HANDOFF.md §9 五子 phase:

| 子 Phase | 目的 | 期望 |
|----------|------|------|
| 6.1 | Dataset 100 random samples 遍历(MA3 §2.C 增强版)| 0 frac sentinel 触发,12 字段 schema 全对齐 |
| 6.2 | DataLoader collate (bs=4) | dict batch 含 list/tensor 字段无 collate 错误 |
| 6.3 | SpectrumEncoder forward | (4, 256), no NaN, mean/std 合理 |
| 6.4 | CPU full forward+backward | loss ∈ [2, 6], grad_norm ∈ (0, 1e4), no NaN grad |
| 6.5 | GPU bf16 forward+backward (cuda:0) | loss 类似 CPU 范围(±10% bf16 漂移),no NaN grad |

**Step 4 启动闸门**:Phase 6 五子全 PASS + 100 random frac filter 零触发 + GPU bf16 loss/grad 全正常。任何一项 FAIL 立刻停汇报。

---

## 9. 给 MA4 的避坑提示(继承 MA3 已踩的坑)

### 坑 1:Sub-Agent 接力时不能只转上一棒结论

MA3 给 Sub-Agent 3 的指令漏转了 Sub-Agent 1/2 的废弃文件审计 → Sub-Agent 3 Phase 4 时碰到 5 个无主文件再次卡住,**整整一轮回炉**。

**修法**:MA4 给 Sub-Agent 4 的指令必须**完整粘贴本文档 §7 文件归属总表**(不是引用,是粘贴全文)。

### 坑 2:env 验证必须走 `python -c "..."` 而非 `pip freeze`

用户最初的 jhub_env pip freeze 输出是 user-level Python 3.10 包(jhub_env 是 Python 3.12),误导整整一轮。

**修法**:任何 env 探查必须 `which python && python --version && python -c "import torch; print(torch.cuda.is_available())"`。

### 坑 3:诚实 > 流畅

Sub-Agent 1/2/3 都在不确定时停下汇报,没有硬推。MA4 必须延续。

如果 Sub-Agent 4 跑 Phase 6 出 NaN/Inf,**不要**"为了进度推 Step 4",立刻停。如果发现"原假设错了"(类似 Step 2.5 Phase D 的 site-averaged vs site-specific 发现),停下来给 MA4 几个选项,不替 MA4 决定。

### 坑 4:handoff 写完用户 review 再发新窗口

MA3 上次明确"handoff 写完转给我先 review 再发新窗口"。MA4 沿用。Sub-Agent 4 handoff 写完贴回用户,用户审完才发到新窗口给 Sub-Agent 4。

### 坑 5:Exp2 final report §3.4 的 TypeClassifier 不要执行

EXP2 final report §3.4 是 Exp3 proposal,但 **Exp3 已证伪**,EXP4_PROPOSAL_v2 §1.3 不可变量明示"不加 TypeClassifier"。

同理:§3.4 提到的 maml.rfxas 与 Exp4 完全无关,Sub-Agent 4 不要触碰。

### 坑 6:incompat_pool.csv 全程不读

唯一例外是 Sub-Agent 3 Phase 0.5 set 数学验证算了一次行数。Phase 5 / 5b / 6 全部禁止 import 任何 incompat_pool 路径。

### 坑 7:holdout 全程不进 DataModule

Step 4 训练只用 train + val。Phase 6 forward test 用 train ds[0]/random 100。holdout 留 Step 5。

### 坑 8:上下文预算把闸门提前到 60%

Sub-Agent 3 自估 62-65% 但用户报告"非常卡",真实消耗超出自估。MA4 给 Sub-Agent 4 的指令应把"主动停"闸门提前到 **60%**,留 buffer 写交接文档。

### 坑 9:Step 5 final report 必须的 caveat 段

Phase 0.1 bit-exact 等价的认识论 caveat MA3 已记入 Step 5 论文 caveat 段:

> "Step 3 Dataset 邻居计算与 shell_boundaries.pkl 通过 bit-exact 验证一致;Phase A 算法的物理正确性继承 Step 2.5 报告,本工作未独立验证。"

加上原 EXP4_PROPOSAL_v2 §7.6 的 incompat 盲区声明:

> "本工作 Exp4 训练了 75,637 样本(剔除 52,745 'incompat' 样本,详见 Step 2.5 Phase D 报告)。incompat 样本结构上含多个不等价 Wyckoff 中心位点,与 MP 谱的 site-averaged 性质不直接对齐。Exp5 计划用 site-averaging 策略激活这部分数据。"

MA4 写 Step 5 交接时这两条都要透传。

### 坑 10:torch_scatter env-local 安装

MA3 决策 R2 修订版,torch_scatter 装到 mlff env 内部(不加 `--user`,不污染共享路径)。如果重新装,命令:

```bash
conda activate mlff
which pip   # 必须 /home/tcat/conda_envs/mlff/bin/pip
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.1+cu124.html
```

---

## 10. 用户应该给你的文件清单

### 必传(按读取顺序)

1. **本文档** EXP4_MAINAGENT4_HANDOFF.md
2. EXP4_PROPOSAL_v2.md
3. EXP4_FILE_INVENTORY.md
4. EXPERIMENT2_FINAL_REPORT.md(注意 §3.4 与 Exp4 矛盾,见 §9 避坑 5)
5. EXP4_STEP3_SUBAGENT_HANDOFF.md(MA3 写,Sub-Agent 1/2/3 用过的总入口)
6. STEP3_SUBAGENT2_REPORT_TO_MA3.md
7. EXP4_PROGRESS_LOG.md(MA1-2 时代历史)

### 可选深 context

8. EXP4_MAINAGENT3_HANDOFF.md(MA2→MA3 接力)
9. STEP1_COMPLETION_REPORT.md
10. STEP2_5_FINAL_REPORT.md

### Sub-Agent 3 的工作汇报(用户对话历史里,如能找到)

- Sub-Agent 3 Phase 0.5 通过汇报
- Sub-Agent 3 Phase 0.1 通过汇报(10/10 max|diff|=0.000e+00 Å)
- Sub-Agent 3 Phase 3 完成汇报(benchmarks: 58µs / 20.7ms)
- Sub-Agent 3 Phase 4 完成汇报(grep sanity 5 处残留全在合理位置)

如果用户对话历史完整,这些是最详细的 action+result 记录,优于本文档的概述。

---

## 11. 你的第一条回复(建议格式)

```
我已阅读完 MA3 交接的所有文档。

[简要复述: Step 1/2/2.5 完成 + Step 3 Phase 0/3/4 完成 + 
当前待办 Phase 5/5b/6 由 Sub-Agent 4 接管]

我注意到 Exp2 final report §3.4 的 TypeClassifier proposal 与 Exp4 
"不加 TypeClassifier"决策矛盾,我会在 Sub-Agent 4 handoff 里特别强调,
避免它被误导。

在写 Sub-Agent 4 handoff 之前,我需要向你确认几件事:

1. Sub-Agent 3 已经停止了对吗? 最后状态是 Phase 4 完成,
   spectrum_encoder.py 5 处已改,grep sanity 通过(残留 5 处全在
   废弃文件)。这个状态可信吗?
   
2. 服务器上 /home/tcat/diffcsp_exp4/code/step3/xas_local_dataset_v2.py
   是否能 ls 看到? (Sub-Agent 3 自报已交付,需要简单确认文件存在性)

3. mlff env 的 torch_scatter env-local 安装是否还在? 
   (Sub-Agent 4 Phase 6 forward test 需要)

4. 服务器根盘可用空间还剩多少? 
   (Step 4 启动前 cache 到 /tmp 需要核实空间)

确认后我开始写 Sub-Agent 4 handoff。
```

---

## 12. 给你的最后一条提醒

**Step 3 比原计划多走了 5 个回合**(Sub-Agent 1 → 2 → 3 + 中间 MA3 多次 review + 决策迭代)。这不是浪费——是因为发现了真实的环境问题(jhub_env CPU-only)、版本兼容性问题(numpy 2.x、sklearn ABI、Cython buffer)、Exp2 仓库文件归属混乱。**最终方案的稳定性是基于这些发现的**。

如果你接下来在 Phase 5/5b/6 中又发现"原假设错了"的情况,**不要硬推**。停下来:

1. 先承认观察("我发现 X 与文档假设 Y 不一致")
2. 解释影响("如果按现状继续,会导致 Z")
3. 给 MA4 几个选项(A/B/C),写各自代价
4. 不替 MA4 做决定,让 MA4 决策

用户对你信任,你回报这份信任的方式是**诚实**,不是"流畅"。

---

*MA3 撰写,2026-04-25,最后一次发言*
