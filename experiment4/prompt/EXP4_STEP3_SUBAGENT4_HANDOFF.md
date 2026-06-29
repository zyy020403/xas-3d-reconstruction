# EXP4_STEP3_SUBAGENT4_HANDOFF.md
# DiffCSP-Experiment4 Step 3 Sub-Agent 4 交接文档

> **撰写者**：DiffCSP-Exp4-Main-Agent 4
> **接收者**：DiffCSP-Exp4-Step3-SubAgent-4
> **日期**：2026-04-26
> **接力关系**：MA1→MA2→MA3→MA4 / Sub-Agent 1→2→3→4
> **前置完成**：Phase 0/3/4（Sub-Agent 1/2/3 接力）+ 5 关键脚本身份核对（Check Agent，5/5 文件确认）
> **你的任务**：Phase 5 + 5b + 6（一窗口完成全部 Step 3 收尾，过 Phase 6 五子闸门后 MA4 启动 Step 4 训练）

---

## §0 快速定位

| 你做 | 你不做 |
|---|---|
| Phase 5：改 `step3/diffusion_w_type_xas.py` 4 项 | 不改任何已被 Check Agent PASS 的文件结构 |
| Phase 5b：新建 `step3/xas_local_datamodule_v2.py`，**类名 XasLocalDataModuleV2** | 不动 `xas_local_dataset_v2.py`（Check Agent 13/13 marker PASS）|
| Phase 6：写 `step3/forward_test.py`，跑 6.1-6.5 五子 phase | 不训练、不进 epoch、不评估、不接触 holdout |
| 完成后向 MA4 汇报，按 §10 模板 | **不改不可变量、不加 TypeClassifier、不读 incompat_pool.csv** |

**单窗口预算**：MA4 估算 80-105K token（顺利）/ 120-150K token（一处需 debug）。**70% 上下文闸门**触发即停汇报，buffer 留给写汇报，不留给"再试一次"。

---

## §1 你必须读的文档（按顺序）

| # | 文档 | 必读? | 重点章节 |
|---|------|-------|---------|
| 1 | **本文档**（你正在读）| ✅ | 全文 |
| 2 | EXP4_CHECK_AGENT_REPORT（Check Agent 报告）| ✅ | F1/F2/F5 三处 SUSPICIOUS 详情（影响你的工作）|
| 3 | EXP4_MAINAGENT4_HANDOFF.md | ✅ | §6 锁定决策清单 + §8 Phase 5/5b/6 任务说明 + §9 避坑提示（10 条）|
| 4 | EXP4_STEP3_SUBAGENT_HANDOFF.md | ✅ | §9.2 Phase 6 测试代码模板（你**直接复用**，不重写）|
| 5 | EXP4_PROPOSAL_v2.md | ✅ | §1.3 不可变量列表 |
| 6 | F1 文件内容（`step3/diffusion_w_type_xas.py` 全文）| ✅ | Phase 5 改 4 项前必须 view |
| 7 | F2 文件内容（`step3/xas_local_datamodule.py` 全文）| ✅ | Phase 5b 重写蓝本 |
| 8 | F5 文件内容（服务器 `step3/xas_local_dataset_v2.py` 全文）| ✅ | Phase 5b 必须知道它的 `__init__` 接口（**data_dir 是构造参数，不是 env var**）|
| 9 | EXP4_PROGRESS_LOG.md | 可选 | 历史 context |
| 10 | EXPERIMENT2_FINAL_REPORT.md | 可选 | §2.3 四版本演化（避免误将 _v1/_v2/_step4c/_v6 当权威）|

---

## §2 Check Agent 报告关键发现（影响你的执行）

这 3 项 SUSPICIOUS 不是错误，是 MA3 假设的细节偏差。你必须按下面理解执行：

### §2.1 F1 SUSPICIOUS：模型文件不 import dataset

**事实**：`step3/diffusion_w_type_xas.py` 全文**不引用**任何 `xas_local_dataset_*`。Dataset 通过 DataModule 注入（架构解耦）。

**影响 MA4 handoff §8.1 item 3**：原文写"Dataset import: from xas_local_dataset_L6 → from xas_local_dataset_v2"——**这条作废**。Phase 5 不要在 diffusion 文件里找 dataset import，找了也找不到。

**真正的 dataset import 在 F2**（DataModule），那是 Phase 5b 的事。

### §2.2 F2 SUSPICIOUS：DataModule 真实类名是 `XASDataModule`（无 "Local"）

**事实**：F2 line 49 类名 `class XASDataModule(pl.LightningDataModule):`。MA3/MA4 之前以为叫 `XasLocalDataModule`。

**影响**：
- Phase 5（如果 diffusion 文件 import DataModule）：要找 `XASDataModule`，不是 `XasLocalDataModule`。
- Phase 5b：你**新写**的类用 MA3 钦定的 `XasLocalDataModuleV2`（不是延续 `XASDataModule` 的命名风格）。这是 MA3 在 §6 决策 B1 锁死的，不要改。

### §2.3 F5 SUSPICIOUS（关键）：dataset_v2 不读 EXP4_DATA_DIR 环境变量

**事实**：`XasLocalDatasetV2.__init__(self, split, data_dir, verbose_init_benchmark=True)`——`data_dir` 是**构造参数**，dataset 内部**不**调 `os.environ.get("EXP4_DATA_DIR", ...)`。

**这是 Sub-Agent 3 的合法设计选择**，已通过 13/13 marker 检查 + 10/10 unit sanity。**不要改 dataset 让它读 env var**——Phase 6 forward test 已基于这个接口设计过了。

**你（Phase 5b）的责任**：在新 `XasLocalDataModuleV2.__init__` 里读 `os.environ`，**显式透传**给 dataset：

```python
# 推荐模式（写在 datamodule_v2 里）
data_dir = os.environ.get("EXP4_DATA_DIR", "/home/tcat/diffcsp_exp4/data")
self.train_ds = XasLocalDatasetV2(split="train", data_dir=data_dir)
self.val_ds   = XasLocalDatasetV2(split="val",   data_dir=data_dir)
# 注意 holdout 不进 DataModule（MA3 §8.2 锁死）
```

---

## §3 环境基线（不变，但你要 sanity check 一次）

承自 MA3 EXP4_MAINAGENT4_HANDOFF §3：

- 服务器 `scsmlnprd02.its.auckland.ac.nz`，密码登录
- 执行 env：**`mlff`**（`/home/tcat/conda_envs/mlff`），不是 jhub_env
- GPU：2× RTX 4090 24 GB，CUDA 12.2，torch 2.4.1+cu124（cuda=True）
- 关键包：pymatgen 2025.10.7 / numpy 2.2.6 / pytorch_lightning 2.5.5 / sklearn 1.7.2 / **torch_scatter 2.1.2+pt24cu124（env-local 已装）**

### §3.1 你接手时第一件事（不超过 5 分钟）

```bash
ssh tcat@scsmlnprd02.its.auckland.ac.nz
conda activate mlff
which python   # 期望: /home/tcat/conda_envs/mlff/bin/python
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 期望: 2.4.1+cu124 True
python -c "import torch_scatter; print(torch_scatter.__version__)"
# 期望: 2.1.2+pt24cu124（确认 Sub-Agent 1 的 env-local 安装还在）

ls /home/tcat/diffcsp_exp4/code/step3/xas_local_dataset_v2.py
ls /home/tcat/diffcsp_exp4/code/step2/spectrum_encoder.py
# 两个文件都必须存在（Sub-Agent 3 已交付，Check Agent 已 PASS）

ls /home/tcat/diffcsp_exp4/data/ | head -20
# 期望看到 data_inventory_v2.csv / spectra_*.pkl / shell_boundaries.pkl / MP_all_POSCAR_flat/ 等

df -h ~  # 根盘可用空间，期望 ≥ 25 GB
```

任一项异常**立即停汇报 MA4**（不要 pip install 任何东西，不要 conda activate 别的 env）。

---

## §4 锁定决策（继承 MA1/MA2/MA3，**不要重新讨论**）

这些是不可变量，违反任一项即作废重做：

### §4.1 Exp4 物理 / 数学层（不可改）

- L_VIRTUAL = 6.0（虚拟晶格边长 Å）
- 坐标系 [-0.5, 0.5]，`frac -= np.round(frac)` min-image
- forward() **无** `% 1.`（这是 Step4 buggy 版本的特征）
- N_NEIGHBORS = 20，CUTOFF_R = 10.0，SYMPREC = 0.1
- batch_size = 16，lr = 1e-4，bf16，num_workers = 0
- 三路 SpectrumEncoder：xmu 150 + chi1 200 + feff **74** → latent 256
- DiffCSP 扩散框架，**cost_lattice = 0**
- **不加 TypeClassifier**（Exp3 已证伪）

### §4.2 Step 3 阶段决策（MA3 锁定）

- Q1：12 字段 return dict schema（已实现于 dataset_v2，不改）
- Q2：双 raise（init defensive + frac sentinel epsilon=1e-6）—— 已实现
- Q3：不加 cache 初版（Step 4 profile 后再决定，**Phase 5b/6 不引入 cache**）
- Q4：每 split 一个 Dataset 实例，**holdout 不进 DataModule**
- 邻居数 < 20 → raise RuntimeError（**不 padding**）

---

## §5 文件归属总表（MA3 §7 确认 + Check Agent 验证后版本）

> **关键**：动手前对照本表确认文件身份。Check Agent 已对 5 个核心文件做完身份核对，本表是**经验证后**的最新版。

### §5.1 step3/ 目录

| 文件 | 身份 | Check Agent 判定 | Phase 5/5b/6 命运 |
|------|------|----------------|------------------|
| `diffusion_w_type_xas.py`（无后缀） | Step4c/4d 共用扩散逻辑 | ✅ PASS（核心 marker M2/M6 干净，line 108 `hparams.get('feat_dim', 73)`）| **Phase 5 改 4 项** |
| `xas_local_datamodule.py`（无后缀） | Step4d datamodule | ⚠️ SUSPICIOUS（仅命名 `XASDataModule`，非阻塞）| **Phase 5b 重写蓝本** |
| `xas_local_dataset_L6.py` | Step4d (L=6) 权威 | ✅ PASS（L=6, min-image, 73 维）| 不动（已被 v2 替代）|
| `xas_local_dataset_v2.py`（Sub-Agent 3 新建）| Exp4 v2 dataset | ✅ PASS（13/13 markers + 5/5 反 marker 干净）| **不动**（Phase 5b/6 调用，Phase 5b 不修改其内部实现）|
| `xas_local_dataset_step4c.py` | Step4c (L=12) 失败 | 未检（声称废弃）| 不动 |
| `xas_local_dataset_v6.py` | Step4b 死路 | 未检（声称废弃）| 不动 |
| `diffusion_w_type_xas_v1.py` | Step4 (L=12 + `%1.` bug) | 未检（声称废弃）| 不动 |
| `diffusion_w_type_xas_v2.py` | Step4c density loss | 未检（声称废弃）| 不动 |
| `step3_5_e2e_forward_test.py` | Step4 时代 e2e debug | 废弃 | **不复用，Phase 6 写新 forward_test.py** |
| `step3_5_nan_debug.py` | Step4 时代 NaN debug | 废弃 | 不复用 |

### §5.2 step2/ 目录

| 文件 | 身份 | Check Agent 判定 | 命运 |
|------|------|----------------|------|
| `spectrum_encoder.py`（服务器版本）| Sub-Agent 3 已改 5 处 73→74 | ✅ PASS（M1-M5 全 ✓，全文 0 个 73 残留）| **Phase 6 forward test 直接 import** |

### §5.3 不变文件（DiffCSP 框架核心，绝对不动）

`gnn.py` / `cspnet.py` / `diff_utils.py` / `run.py` / 所有数学层 / 采样器 / loss 函数。

---

## §6 Phase 5：`diffusion_w_type_xas.py` 4 项改动

### §6.1 改动前先 grep 探查（5 分钟）

```bash
cd /home/tcat/diffcsp_exp4/code/step3
# 1. feat_dim 出现位置（核对 line 108 + hydra config 是否 override）
grep -n "feat_dim" diffusion_w_type_xas.py
grep -rn "feat_dim" /home/tcat/diffcsp_exp4/code/  # 含 yaml 配置

# 2. 数据路径常量（找出实际硬编码位置）
grep -n -E "(EXP4_DATA|/home/tcat|data_dir|DATA_DIR)" diffusion_w_type_xas.py

# 3. DataModule import（确认 Phase 5 是否需要改 import）
grep -n -E "(import|from).*datamodule" diffusion_w_type_xas.py

# 4. Dataset import（Check Agent 报告说不该有；验证一遍）
grep -n -E "(import|from).*dataset" diffusion_w_type_xas.py
```

把 4 条 grep 输出**完整记录**到你的工作日志（汇报时透传给 MA4）。

### §6.2 改动 1：line 108 默认 feat_dim

**原文**（Check Agent 已确认）：
```python
feat_dim = self.hparams.get('feat_dim', 73)
```

**改为**：
```python
feat_dim = self.hparams.get('feat_dim', 74)
```

**配套**：grep 输出里如果 yaml 配置文件（`conf*/*.yaml`）有 `feat_dim: 73`，**也要改成 74**，否则 hparams.get 会拿到 yaml 里的 73 覆盖默认值。如果 yaml 完全不写 feat_dim，默认 74 生效。

### §6.3 改动 2：数据路径常量 → 环境变量

**改动意图**：原 Exp2 硬编码（如 `/path/to/exp2/fe_oxide` 或类似）→ 从 `EXP4_DATA_DIR` 环境变量读，**不读时 fallback 到 `/home/tcat/diffcsp_exp4/data`**。

**实现模式**：
```python
import os
DATA_DIR = os.environ.get("EXP4_DATA_DIR", "/home/tcat/diffcsp_exp4/data")
```

**注意**：
- 如果 §6.1 grep 显示 diffusion 文件**没有**硬编码数据路径（路径全在 hydra config 里），这条改动改 yaml 即可。
- 如果有硬编码，按上面模式替换。
- 不要把 §6.4 的 DataModule import 路径替换成 env var——那是 Python module 路径，不是数据路径。

### §6.4 改动 3：DataModule import

如果 §6.1 grep 输出显示 diffusion 文件 import 了 DataModule（如 `from xas_local_datamodule import XASDataModule`）：

**改为**：
```python
from xas_local_datamodule_v2 import XasLocalDataModuleV2
```

并把后续实例化代码 `XASDataModule(...)` 改为 `XasLocalDataModuleV2(...)`。

如果 §6.1 grep 显示 diffusion 文件**不直接** import DataModule（DataModule 通过 hydra config 实例化），改 yaml 配置文件里的 `_target_` 字段即可。

### §6.5 改动 4：Dataset import（已确认不存在，跳过）

按 §2.1，`diffusion_w_type_xas.py` **不引用** dataset 文件。这条 MA4 handoff §8.1 item 3 作废，跳过。

如果你 grep 后发现 Check Agent 错了，**立即停汇报 MA4**——这意味着 Check Agent 的 §5.1 F1 分析有遗漏，需要重新评估。

### §6.6 Phase 5 完成后 grep sanity

```bash
grep -n "feat_dim" diffusion_w_type_xas.py
# 期望：所有 73 改成 74，行号没变（line 108 附近）

grep -rn "from xas_local_datamodule\b" /home/tcat/diffcsp_exp4/code/step3/diffusion_w_type_xas.py
# 期望：零命中（已改成 _v2 或 yaml 配置）

grep -rn "73" /home/tcat/diffcsp_exp4/code/step3/diffusion_w_type_xas.py
# 期望：剩余 73 应该都在无关上下文（如其他维度、注释中提及历史）
```

把每条 grep 输出粘到日志。

---

## §7 Phase 5b：新建 `xas_local_datamodule_v2.py`

### §7.1 关键约束（MA3 §6 锁定 + Check Agent 修正）

1. **文件名**：`step3/xas_local_datamodule_v2.py`
2. **类名**：`XasLocalDataModuleV2`（**不要**沿用 `XASDataModule`，MA3 钦定 v2 命名）
3. **Dataset import**：`from xas_local_dataset_v2 import XasLocalDatasetV2`
4. **数据路径**：DataModule 内**显式**读 `os.environ.get("EXP4_DATA_DIR", "/home/tcat/diffcsp_exp4/data")`，透传给 dataset 构造函数
5. **删除** v1 73 维 FEFF CSV 路径常量（如果 F2 模板里有 `FEFF_CSV` / `feff_features_all_csv_75cols(in).csv` 等 Windows 路径或 v1 csv 路径，全删——Exp4 用 `feff_features_imputed.pkl`）
6. **PL 2.5.5 兼容**：`setup()` 签名 F2 是 `def setup(self, stage: Optional[str] = None)`，PL 2.x 兼容，不动；但 grep 一遍：

   ```bash
   grep -n "if stage is None" /home/tcat/diffcsp_exp4/code/step3/xas_local_datamodule.py
   ```
   有命中 → 这是 PL 1.x 行为依赖，重写时改成 `if stage in (None, "fit"):`（PL 2.x 习惯）。
7. **DataModule 结构不动**：保留 `setup` / `train_dataloader` / `val_dataloader` / `test_dataloader` 四个方法。
8. **holdout 绝对不进**——不要写 `holdout_dataloader`、不要 setup 时 load `spectra_holdout.pkl`。

### §7.2 实现骨架（参考意图，不是抄）

```python
import os
from typing import Optional
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from xas_local_dataset_v2 import XasLocalDatasetV2

class XasLocalDataModuleV2(pl.LightningDataModule):
    def __init__(self, batch_size: int = 16, num_workers: int = 0):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = os.environ.get(
            "EXP4_DATA_DIR", "/home/tcat/diffcsp_exp4/data"
        )

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit"):
            self.train_ds = XasLocalDatasetV2(
                split="train", data_dir=self.data_dir
            )
            self.val_ds = XasLocalDatasetV2(
                split="val", data_dir=self.data_dir
            )
        if stage in (None, "test"):
            self.test_ds = XasLocalDatasetV2(
                split="test", data_dir=self.data_dir
            )
        # holdout 故意不 load——Step 5 单独实例化

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=self.batch_size,
            num_workers=self.num_workers, shuffle=True,
            collate_fn=self._collate,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, batch_size=self.batch_size,
            num_workers=self.num_workers, shuffle=False,
            collate_fn=self._collate,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, batch_size=self.batch_size,
            num_workers=self.num_workers, shuffle=False,
            collate_fn=self._collate,
        )

    @staticmethod
    def _collate(batch):
        # dataset 返回 dict 含 string 字段，需要自定义 collate
        # 完整实现见 EXP4_STEP3_SUBAGENT_HANDOFF.md §9.2 Phase 6.2
        ...
```

**collate 函数的实现** 直接参考 EXP4_STEP3_SUBAGENT_HANDOFF.md §9.2 Phase 6.2，**那段就是给 datamodule 用的**——把它抽进 DataModule 作为 staticmethod 或 module-level 函数。

### §7.3 Phase 5b 完成后 sanity

```bash
# 类名 + import 验证
grep -n "class XasLocalDataModuleV2" /home/tcat/diffcsp_exp4/code/step3/xas_local_datamodule_v2.py
# 期望: 1 命中

grep -n "XasLocalDatasetV2" /home/tcat/diffcsp_exp4/code/step3/xas_local_datamodule_v2.py
# 期望: 至少 4 命中（import + train + val + test 三处实例化）

grep -n "holdout" /home/tcat/diffcsp_exp4/code/step3/xas_local_datamodule_v2.py
# 期望: 0 命中（holdout 不进 datamodule）

grep -n "EXP4_DATA_DIR" /home/tcat/diffcsp_exp4/code/step3/xas_local_datamodule_v2.py
# 期望: ≥ 1 命中（os.environ.get 调用）

grep -n "FEFF_CSV\|75cols\|MP_all_EXAFS" /home/tcat/diffcsp_exp4/code/step3/xas_local_datamodule_v2.py
# 期望: 0 命中（v1 路径常量已删）
```

---

## §8 Phase 6：写 `forward_test.py` 跑 5 子 phase

### §8.1 复用原则

**完整代码模板**已在 EXP4_STEP3_SUBAGENT_HANDOFF.md §9.2（Phase 6.1-6.5）写好。**你直接复用**那段代码，按下面 4 条修订：

1. **Phase 6.1 改 100 random samples**（MA3 决策 C3 修订版，不是单 ds[0]）：
   ```python
   ds = XasLocalDatasetV2(split="train", data_dir=os.environ.get("EXP4_DATA_DIR", "/home/tcat/diffcsp_exp4/data"))
   import random
   random.seed(42)
   indices = random.sample(range(len(ds)), 100)
   for i in indices:
       sample = ds[i]
       fc = sample["frac_coords"]
       assert fc.min() >= -0.5 - 1e-6 and fc.max() <= 0.5 + 1e-6, (i, fc.min(), fc.max())
       # 同时验证 12 字段全在
       expected_keys = {"xmu", "chi1", "feff", "frac_coords", "atom_types",
                        "sample_name", "mp_id", "center_element",
                        "eval_cutoff", "eval_cutoff_fallback",
                        "n_center_sites", "site_equivalence_tag"}
       assert set(sample.keys()) == expected_keys, (i, set(sample.keys()) ^ expected_keys)
   ```

2. **Phase 6.2 collate**：从 datamodule_v2 import 同一个 `_collate`，验证 4 样本 batch 出来的 dict 字段对齐（tensor 字段 stack 成 batch dim，str 字段进 list）。

3. **Phase 6.4 / 6.5 model 类名**：F1 文件里的 model 类名你要先 grep 确认（不一定叫 `DiffusionWithTypeXAS`）：
   ```bash
   grep -n "^class .*pl\.LightningModule" /home/tcat/diffcsp_exp4/code/step3/diffusion_w_type_xas.py
   grep -n "^class " /home/tcat/diffcsp_exp4/code/step3/diffusion_w_type_xas.py
   ```
   用 grep 出来的真实类名 import + 实例化。

4. **Phase 6.5 GPU bf16**：模型和 batch 都 `.to(torch.bfloat16)`。**non-floating-point tensor**（如 `atom_types` 是 long）保持 long 不要 cast bf16，会报错。EXP4_STEP3_SUBAGENT_HANDOFF.md §9.2 Phase 6.5 的代码模板已处理这个分支，照抄即可。

### §8.2 五子 phase 期望（Step 4 启动闸门）

| 子 Phase | 期望 PASS 条件 |
|---|---|
| 6.1 | 100 个 random sample 全部 frac ∈ [-0.5, 0.5]，零 frac sentinel 触发；12 字段集合严格等于预期 |
| 6.2 | bs=4 collate 不报错；tensor 字段 batch dim 正确；str 字段是 list of 4 |
| 6.3 | SpectrumEncoder forward 输出 (4, 256)；no NaN；mean ∈ [-5, 5]；std ∈ [0.1, 5] |
| 6.4 | CPU model.training_step(batch) loss ∈ [2, 6]；backward 后 grad_norm ∈ (0, 1e4)；no NaN grad |
| 6.5 | GPU bf16 loss 与 CPU 同范围 ±10%；grad_norm 正常；no NaN grad |

**全 PASS 才进 Step 4**。任一子 phase FAIL → 立即停汇报，**不要尝试 debug 超过 1 轮**（参考 §9.1 停汇报触发条件）。

### §8.3 forward_test.py 文件位置

`/home/tcat/diffcsp_exp4/code/step3/forward_test.py`（**新建**，不复用 `step3_5_e2e_forward_test.py` 或 `step3_5_nan_debug.py`，那是 Step4 时代废弃文件）。

测试日志写到 `/home/tcat/diffcsp_exp4/logs/step3_forward_test_log.txt`。

---

## §9 停汇报触发条件（MA3 工作哲学：诚实 > 流畅）

下面任一情况**立即停下汇报 MA4**，不要硬推、不要"再试一次"、不要替 MA4 决定：

### §9.1 强制停汇报

| 触发 | 原因 |
|---|---|
| §3.1 环境 sanity 任一项失败（torch_scatter 不在 / cuda 不可用 / 文件丢失） | 服务器状态变了，需要 MA4 决策 |
| §6.1 grep 输出与 Check Agent 报告不一致（如 dataset import 出现在 diffusion 文件） | Check Agent 漏检，重新评估 |
| Phase 5 改动后 import 报错（如 `ModuleNotFoundError: xas_local_datamodule_v2`） | 顺序错了，应先 5b 后 5——重新规划 |
| Phase 6.1 frac sentinel 触发任一次 | dataset 物理 bug 或 incompat 漏剔，**绝对不就地修 dataset** |
| Phase 6.1 12 字段集合不等于预期 | dataset_v2 实际接口与 Check Agent 报告不一致 |
| Phase 6.4 CPU loss 是 NaN/Inf 或越出 [2, 6] 范围超过 ±50% | 数学层有问题，可能是 feat_dim 改不全或 yaml override |
| Phase 6.5 GPU bf16 出 NaN grad | bf16 数值稳定性问题，可能要 fp32 训练或加 `set_grad_to_none` 等——**这是 MA4 决策点，不是 Sub-Agent 4 决策点** |
| 上下文用量到 70% | 闸门，立即停 |

### §9.2 汇报格式（即使中途停也按这个）

```
# Phase 5/5b/6 中途停汇报

## 已完成
- §3.1 环境 sanity: [全 ✓ / 部分 ✓]
- Phase 5: [完成 / 中途 / 未开始]，改了 ___ 项
- Phase 5b: [完成 / 中途 / 未开始]
- Phase 6: [6.1=__ / 6.2=__ / 6.3=__ / 6.4=__ / 6.5=__]

## 触发停因
[精确描述：哪一步、什么期望、什么实际]

## 我已观察的事实（不解读）
[grep 输出 / 报错 traceback / 数值 / 任何客观信息]

## 我没做的事
[避免误判：我没改 dataset、没装新包、没用 holdout 等]

## 给 MA4 的选项（不替你决定）
A. ___（代价 ___）
B. ___（代价 ___）
C. ___（代价 ___）

## 上下文消耗估算
约 __ %
```

---

## §10 Phase 6 全 PASS 后的完成汇报模板

```
# Sub-Agent 4 完成汇报（Phase 5 + 5b + 6 全 PASS）

## §3.1 环境 sanity
torch_scatter / cuda / 文件存在性 / 磁盘空间: 全 ✓

## Phase 5（diffusion_w_type_xas.py）
- §6.1 grep 输出（4 条）: ___（粘贴）
- §6.2 line 108 改 73→74: ✓ / yaml 是否 override: ___
- §6.3 数据路径常量: ___（实际改了几行 / 改在了 yaml 还是 .py）
- §6.4 DataModule import: ___（改了 / 不需要改原因）
- §6.5 跳过（按 §2.1 / §6.5 确认）: ✓
- §6.6 grep sanity: ___（粘贴）

## Phase 5b（xas_local_datamodule_v2.py）
- 文件位置: /home/tcat/diffcsp_exp4/code/step3/xas_local_datamodule_v2.py
- 行数: ___
- 类名: XasLocalDataModuleV2 ✓
- §7.1 七项约束逐项: ___ / ___ / ___ / ___ / ___ / ___ / ___ ✓
- §7.3 grep sanity: ___（粘贴）

## Phase 6（forward_test.py）
- 6.1 100 random samples 12 字段 + frac 范围: PASS / 用时 ___ s
- 6.2 DataLoader collate (bs=4): PASS
- 6.3 SpectrumEncoder forward: 输出 shape ___, mean ___, std ___, PASS
- 6.4 CPU forward+backward: loss ___, grad_norm ___, PASS
- 6.5 GPU bf16 forward+backward: loss ___, grad_norm ___, PASS

## Phase 6 启动闸门（Step 4 训练前置）
五子全 PASS: ✓
零 NaN/Inf: ✓
预期范围内: ✓

## 我没做的事（防御性陈述）
- 没改 dataset_v2 内部实现（M6 env var 在 datamodule 解决）
- 没接触 incompat_pool.csv
- 没接触 holdout（除 Phase 0.5 时代由 Sub-Agent 3 做过的 key alignment）
- 没装新 pip / conda 包
- 没改 cost_lattice / TypeClassifier / 任何不可变量

## 给 MA4 的开放问题
[如果有，列出；没有就写"无"]

## 上下文消耗估算
约 __ %
```

---

## §11 Sub-Agent 4 工作哲学（继承 Sub-Agent 1/2/3）

1. **诚实 > 流畅**：不确定就停下汇报，给 MA4 选项不替决定
2. **改动前先 view + grep**，不靠记忆改文件
3. **每个改动后立即 sanity check**，不积累到最后一起验
4. **每条命令 + 每条 grep 输出**写到日志，汇报时透传 MA4
5. **70% 上下文闸门是闸门，不是参考值**——预算用完 = 停（不是慌）
6. **不创新，不优化"顺手"做的事**——你的工作就是 Phase 5 + 5b + 6 三件事
7. **不主动跳到 Step 4**——即使 Phase 6 全 PASS，也是 MA4 启动 Step 4，不是你

---

*MA4 撰写,2026-04-26,等待用户 review 后投入新窗口*
