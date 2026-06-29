# EXP4_STEP4_SUBAGENT5_HANDOFF.md
# DiffCSP-Experiment4 Step 4 训练 Sub-Agent 5 交接文档

> **撰写者**: DiffCSP-Exp4-Main-Agent 5
> **接收者**: Step 4 Sub-Agent (本接力链编号 Sub-Agent 5)
> **日期**: 2026-04-26
> **背景**: Step 3 Phase 6 五子全 PASS,Step 4 启动闸门 CLEAR。MA5 已与用户对齐 4 项执行约束(见 §3),你接手启动训练。
> **核心约束**: 你不做架构判断、不替 MA5 决策、不深 debug。出意外停下来汇报。

---

## §0 你是谁,你的工作边界

你是 DiffCSP-Exp4 接力链的 **Sub-Agent 5**(Sub-Agent 1→2→3→4→4-续→4-续 2→**5**)。

**你做什么**:
1. Phase 4.0 hard check(disk/env/import 链)
2. Phase 4.1 写 `step4_1_smoke_test.py` + `step4_2_train.py`(参考 Exp2 step4d 但有关键修改)
3. Phase 4.2 跑 smoke test(2 epoch × 10 batch)→ 通过才进 4.3
4. Phase 4.3 nohup 启动正式训练 + 写后台启动指令到 README
5. Phase 4.4 留 ~30 min 监控窗口,看早期 loss / GPU util / epoch 时间
6. Phase 4.5 写中期报告交回 MA5 + 关窗口(训练后台继续 30-100 小时)

**你不做什么**:
- 不动 dataset_v2 / datamodule_v2 / spectrum_encoder / diffusion / yaml / forward_test.py
- 不深 debug(任何 phase FAIL 立刻停,把观察 + 候选解释交给 MA5)
- 不替 MA5 改架构变量(L=6 / N_NEIGHBORS=20 / batch_size=16 / lr=1e-4 / num_workers=0)
- 不动 holdout / incompat_pool(任何 read 路径都不能含这两个文件名)
- 不加 TypeClassifier(Exp3 已证伪,EXP4_PROPOSAL_v2 §1.3 锁定)
- 不装新包(PROPOSAL 用的 7 守卫包 + Sub-Agent 4-续 装的 18 子依赖已够;万一 import 缺包,停下汇报,不自行 pip install)

**上下文闸门**: **70%**(MA5 阶段已放宽,见 EXP4_MAINAGENT5_HANDOFF §8 坑 7)。到 70% 必须停,把"未做完事项 + 当前状态"交回 MA5。

---

## §1 必读文档清单(按读取顺序)

用户会一次性传给你所有文档。按下表读取,**第 1-3 必精读,第 4-6 速读**:

| # | 文档 | 必读? | 重点章节 |
|---|------|-------|---------|
| 1 | **本文档** | ✅ 精 | 全文,尤其 §3-§9 |
| 2 | **EXP4_MAINAGENT5_HANDOFF.md** | ✅ 精 | §3 执行环境 + §6 Step 4 框架 + §8 坑 1-10 |
| 3 | **EXP4_STEP3_SUBAGENT4CONT_FINAL_REPORT.md** | ✅ 精 | §2 守卫包不变量 + §4 资产清单 + §5 Open issues |
| 4 | EXP4_PROPOSAL_v2.md | ✅ 速 | §1.3 不可变量 + §6 预期指标 |
| 5 | EXP4_FILE_INVENTORY.md | ✅ 速 | 数据文件位置 + schema(训练 dataloader 路径要对) |
| 6 | EXPERIMENT2_FINAL_REPORT.md | 速 | §2.3 Step4d 配置(你写训练脚本要参考) |

**额外**: 用户应当在服务器 `/home/tcat/diffcsp_exp4/code/` 下让你看到 Exp2 fork 的 `step4d_2_train.py`(如果在 step3 同级目录或 step4 待建目录)。如果**找不到**,停下来问用户路径,不要凭空写训练脚本。

---

## §2 当前项目状态(继承,不重新讨论)

**Step 0/1/2/2.5/3 全部完成**:
- 75,637 v2 样本 / 88 元素 / split 60507+7624+4481+3025
- 服务器路径 `/home/tcat/diffcsp_exp4/`(data/code/checkpoints/logs)
- Step 3 Phase 6 五子全 PASS,`step3/forward_test.py` 当前 fp32 路径 5/5 PASS

**关键已固定**:
- `step2/spectrum_encoder.py` Linear 73→74 改完
- `step3/diffusion_w_type_xas.py` line 108 `feat_dim=74` 改完
- `step3/conf_xas/model/diffusion_xas.yaml` line 18 `feat_dim=74` 改完
- `step3/xas_local_dataset_v2.py` 12 字段 + 双 raise 防御
- `step3/xas_local_datamodule_v2.py` 247 行,类 `XasLocalDataModuleV2`

**回滚锚点**(应急,你不主动用):
```bash
cp /home/tcat/diffcsp_exp4/code/step3/forward_test.py.bak3 \
   /home/tcat/diffcsp_exp4/code/step3/forward_test.py
```

---

## §3 锁定执行约束(MA5 与用户对齐)

| 项 | 决策 | 来源 |
|---|---|---|
| precision | **fp32** 全程 | MA4 决策 D1(Phase 6.5 fp32 5/5 PASS) |
| max_epochs | **500** | EXP2 Step4d 同值;Exp4 数据 ~7×,500 是上限 |
| early_stop patience | **30** | EXP2 同值,大概率 100-200 epoch 触发 |
| batch_size | **16** | PROPOSAL §1.3 锁定 |
| lr | **1e-4** | PROPOSAL §1.3 锁定 |
| num_workers | **0** | PROPOSAL §1.3 锁定;**不要改成 >0**(Exp4 dataset 含 pymatgen SGA,workers>0 风险高) |
| gradient_clip_val | **1.0** | PROPOSAL §1.3 锁定 |
| save_top_k | **1**(替代 Exp2 的 3) | 磁盘紧,只留 best + last |
| devices | **1** GPU | 用户已确认不试 ddp |
| accelerator | `"gpu"` | |
| check_val_every_n_epoch | **1**(每 epoch val 一次) | 配合 patience=30 = 30 epoch 无改善停;若设 5 则 patience 实际放大 5×,不合理 |

**用户已 / 即将完成**:
- 清理旧 wandb / ckpt(目标 `~/` 可用 ≥ 30 GB)。**Sub-Agent 5 在 Phase 4.0 必须 `df -h ~` 复测,< 30 GB 立刻停汇报**。

**用户产出节奏要求**:
- Sub-Agent 5 写脚本 → smoke test → 启动 nohup → 留 30 min 监控早期 loss → 中期报告 + 关窗口
- 训练后台 30-100 小时,用户回来主动汇报,MA5 再开 Step 5 Sub-Agent

---

## §4 文件归属总表(完整粘贴,不引用)

> 取自 EXP4_MAINAGENT4_HANDOFF.md §7,加 Sub-Agent 4-续/4-续 2 完结状态。
> Sub-Agent 5 任何 import / 路径常量必须严格按本表,**不要凭记忆写**。

### 4.1 step3/ 目录(EXP2 fork 来源,服务器位于 `/home/tcat/diffcsp_exp4/code/step3/`)

| 文件 | 状态 | Step 4 用法 |
|------|------|------------|
| `xas_local_dataset_v2.py` | Sub-Agent 3 交付 ✓ | **`from xas_local_dataset_v2 import XasLocalDatasetV2`** |
| `xas_local_datamodule_v2.py` | Sub-Agent 4 交付 ✓ | **`from xas_local_datamodule_v2 import XasLocalDataModuleV2`** |
| `diffusion_w_type_xas.py` | Sub-Agent 4 改完 line 108 ✓ | **`from diffusion_w_type_xas import CSPDiffusion`** (类名以实际为准,grep 验) |
| `conf_xas/model/diffusion_xas.yaml` | Sub-Agent 4 改完 line 18 ✓ | hydra instantiate 入口 |
| `forward_test.py` | Sub-Agent 4-续 2 fp32 改完 ✓ | **不复用,Step 4 不 import 它** |
| `forward_test.py.bak3` | 4-续 2 留的回滚锚点 | **不动,留作应急** |
| `xas_local_dataset_L6.py` | EXP2 时代,Step4d 用 | **不复用** |
| `xas_local_dataset_step4c.py` | EXP2 Step4c 实验残留 | **不复用** |
| `xas_local_dataset_v6.py` | EXP2 命名混乱,内容不可信 | **不复用** |
| `xas_local_datamodule.py`(无 _v2) | EXP2 时代,被 v2 取代 | **不复用** |
| `diffusion_w_type_xas_v1.py` | EXP2 Step4(L=12+`%1.` bug)| **不复用,保留** |
| `diffusion_w_type_xas_v2.py` | EXP2 Step4c density loss | **不复用,保留** |
| `step3_5_e2e_forward_test.py` | EXP2 e2e debug 残留 | **不复用** |
| `step3_5_nan_debug.py` | EXP2 NaN debug 残留 | **不复用** |

### 4.2 step2/ 目录

| 文件 | 状态 | Step 4 用法 |
|------|------|------------|
| `spectrum_encoder.py` | Sub-Agent 3 改完 5 处 73→74 ✓ | **被 diffusion_w_type_xas import**,你不直接 import |
| `step2_2_preprocess_validation.py` 等其他 | EXP2 时代 | **不复用** |

### 4.3 step4/ 目录(Sub-Agent 5 待建)

| 文件 | 阶段 | 你的产出 |
|------|------|---------|
| `step4_1_smoke_test.py` | Phase 4.2 | **新建**,模仿 Exp2 `step4d_1_quick_test.py` 但改路径/precision/import |
| `step4_2_train.py` | Phase 4.3 | **新建**,模仿 Exp2 `step4d_2_train.py` 但改路径/precision/import/save_top_k |
| `step4_README.md` | Phase 4.3 | **新建**,记录 nohup 启动命令、log 位置、回顾用 |
| (`step4_3_sample.py` / `step4_4_compute_metrics.py`) | Step 5 | **不归你**,Step 5 Sub-Agent 写 |

### 4.4 不动文件(DiffCSP 框架核心)

继承 EXP2,**Sub-Agent 5 完全不碰**:
- `gnn.py` / `cspnet.py` / `diff_utils.py`(扩散数学层)
- `diffcsp/` 包内任何文件
- `conf/` 顶层 hydra config

---

## §5 Phase 子任务清单

### Phase 4.0:Hard check(必先全过)

**目的**: 确认 env / 磁盘 / 文件齐全 / 关键 import 不炸。任一项 FAIL 立刻停。

```bash
# 5.0.1 disk + memory
df -h ~                  # 期望 ≥ 30 GB,< 30 GB 停汇报(用户已答应清理,你做最后核实)
free -h                  # RAM ≥ 10 GB(tmpfs cache 用)
df /tmp                  # /tmp 是 tmpfs (Mounted on /tmp shows tmpfs),available ≥ 5 GB

# 5.0.2 env
which python             # 期望 /home/tcat/conda_envs/mlff/bin/python
python --version         # 期望 Python 3.10.x
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
                         # 期望 2.4.1+cu124 True
nvidia-smi               # 看 GPU 0/1 占用,记录哪个空闲(后面 CUDA_VISIBLE_DEVICES 用)

# 5.0.3 关键文件存在
for f in \
  /home/tcat/diffcsp_exp4/code/.env \
  /home/tcat/diffcsp_exp4/code/step3/xas_local_dataset_v2.py \
  /home/tcat/diffcsp_exp4/code/step3/xas_local_datamodule_v2.py \
  /home/tcat/diffcsp_exp4/code/step3/diffusion_w_type_xas.py \
  /home/tcat/diffcsp_exp4/code/step3/conf_xas/model/diffusion_xas.yaml \
  /home/tcat/diffcsp_exp4/code/step3/forward_test.py \
  /home/tcat/diffcsp_exp4/code/step2/spectrum_encoder.py \
  /home/tcat/diffcsp_exp4/data/data_inventory_v2.csv \
  /home/tcat/diffcsp_exp4/data/train_samples_v2.csv \
  /home/tcat/diffcsp_exp4/data/val_samples_v2.csv \
  /home/tcat/diffcsp_exp4/data/spectra_train.pkl \
  /home/tcat/diffcsp_exp4/data/spectra_val.pkl \
  /home/tcat/diffcsp_exp4/data/feff_features_imputed.pkl \
  /home/tcat/diffcsp_exp4/data/feff_feature_scaler.pkl ; do
  [ -e "$f" ] && echo "OK $f" || echo "MISS $f"
done | grep MISS  # 期望无输出

# 5.0.4 关键 import 链(MA5 坑 2 教训:必跑,不只 ls)
cd /home/tcat/diffcsp_exp4/code
PYTHONPATH=/home/tcat/diffcsp_exp4/code python -c "
import torch, pytorch_lightning as pl, hydra
from omegaconf import OmegaConf
import sys; sys.path.insert(0, 'step3'); sys.path.insert(0, 'step2')
from xas_local_dataset_v2 import XasLocalDatasetV2
from xas_local_datamodule_v2 import XasLocalDataModuleV2
from diffusion_w_type_xas import CSPDiffusion  # 类名以文件实际为准,grep 验
import torch_scatter, torch_sparse
print('IMPORTS OK')
print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())
print('pl:', pl.__version__)
"
# 期望最后一行 IMPORTS OK + 版本号

# 5.0.5 重跑 forward_test.py 一次,确认 baseline 仍 PASS
cd /home/tcat/diffcsp_exp4/code/step3
PYTHONPATH=/home/tcat/diffcsp_exp4/code python forward_test.py 2>&1 | tee /tmp/sa5_phase40_forward.log | tail -30
# 期望末尾打印 Phase 6.1-6.5 全部 PASS
```

**任一异常处理**:
- disk < 30 GB → 停,告诉用户具体值 + 建议清理目标
- env / cuda 不对 → 停,把 `python -c` 输出贴出来给 MA5
- 文件 MISS → 停,列清单给 MA5
- import 链炸 → 停,贴 Traceback 给 MA5
- forward_test.py 不再 PASS → **重大警告**,可能 env 漂移,立刻停,**不要尝试自行修复**

### Phase 4.1:写训练脚本

**两份脚本**: `step4_1_smoke_test.py` 和 `step4_2_train.py`,放在 `/home/tcat/diffcsp_exp4/code/step4/`(目录待你 mkdir)。

#### 模板参考: Exp2 `step4d_2_train.py`

服务器上应有 Exp2 fork。如果路径找不到,**停下来问用户**,不要凭空写。Exp2 fork 应在 `/home/tcat/diffcsp_exp4/code/`(顶层)或某子目录。`find / -name "step4d_2_train.py" 2>/dev/null` 可以找。

#### Exp2 → Exp4 必改清单

| Exp2 step4d_2_train.py | Exp4 step4_2_train.py |
|---|---|
| `PRECISION = 'bf16'` | **`PRECISION = 32`**(MA4 决策 D1) |
| `PROJECT_ROOT = r"C:\..."` 硬编码 | **删,改用 `from dotenv import load_dotenv; load_dotenv("/home/tcat/diffcsp_exp4/code/.env")` 或 `os.environ["PROJECT_ROOT"]`** |
| `from xas_local_datamodule import XASDataModule` | **`from xas_local_datamodule_v2 import XasLocalDataModuleV2`** |
| `from xas_local_dataset_L6 import ...` | 不要 import dataset(由 datamodule 内部 import) |
| `from diffusion_w_type_xas import ...`(EXP2 旧路径) | **同名 import,但版本是 Sub-Agent 4 改完的(line 108)** |
| `MAX_EPOCHS = 500` | **保持 500** |
| `EARLY_STOP_PAT = 30` | **保持 30** |
| `BATCH_SIZE = 16` | **保持 16** |
| `LR = 1e-4` | **保持 1e-4** |
| `GRADIENT_CLIP = 1.0` | **保持 1.0** |
| `NUM_WORKERS = 0` | **保持 0** |
| `ckpt_callback save_top_k=3` | **改 `save_top_k=1`** |
| `default_root_dir = STEP4d_DIR`(Win 路径)| `default_root_dir = "/home/tcat/diffcsp_exp4/checkpoints"` |
| csv_logger Win 路径 | `csv_logger = CSVLogger("/home/tcat/diffcsp_exp4/logs/csv", name="step4_train")` |
| `gpus=1` (旧 PL 1.x API) | **删,改 `accelerator="gpu", devices=1`**(PL 2.5.5 必须) |
| `check_val_every_n_epoch=1` 或不设 | **显式设 `check_val_every_n_epoch=1`**,与 patience=30 配套 |

#### Trainer 参数模板(直接粘到脚本)

```python
trainer = pl.Trainer(
    default_root_dir="/home/tcat/diffcsp_exp4/checkpoints",
    logger=csv_logger,
    callbacks=[ckpt_cb, lr_cb, early_cb],
    precision=32,                    # ← MA4 决策 D1
    accelerator="gpu",
    devices=1,
    gradient_clip_val=1.0,
    max_epochs=500,
    check_val_every_n_epoch=1,
    log_every_n_steps=50,
    enable_progress_bar=True,
)
```

#### Callback 三件套

```python
# best ckpt: monitor val_loss, mode min, top_k=1
ckpt_cb = ModelCheckpoint(
    dirpath="/home/tcat/diffcsp_exp4/checkpoints",
    filename="best-{epoch:03d}-{val_loss:.4f}",
    monitor="val_loss",
    mode="min",
    save_top_k=1,
    save_last=True,
)
# lr 监控
lr_cb = LearningRateMonitor(logging_interval="epoch")
# early stop
early_cb = EarlyStopping(
    monitor="val_loss",
    mode="min",
    patience=30,
    verbose=True,
)
```

#### tmpfs cache(可选,推荐)

PROPOSAL §2.4 提到:训练前 cache 数据到 `/tmp/diffcsp_cache`(tmpfs RAM)。Exp4 数据 ~650 MB 全 cache 进 tmpfs 后,POSCAR open 时间从磁盘 → RAM。但不解决 pymatgen SGA 的 ~20 ms parse cost(那是 CPU 算法,与 I/O 无关)。

**做法**: 训练脚本启动前(在 bash 一次性 cp,不放进 Python):
```bash
mkdir -p /tmp/diffcsp_cache
cp -r /home/tcat/diffcsp_exp4/data/* /tmp/diffcsp_cache/
# 然后 export EXP4_DATA_DIR=/tmp/diffcsp_cache 或在 .env 里改
```

但这会让 .env 与 forward_test 时的 PROJECT_ROOT 设定脱钩。**推荐**: 不动 .env,直接改训练脚本里的数据路径常量,加一个 `DATA_DIR = "/tmp/diffcsp_cache"`(若存在)`else "/home/tcat/diffcsp_exp4/data"` 的 fallback。

如果 `df /tmp` 显示 < 5 GB 可用,**不 cache**,直接读 `/home/tcat/diffcsp_exp4/data/`,代价是磁盘 I/O 慢一些,不是阻塞性问题。

#### PL 2.5.5 兼容检查清单(grep Exp2 step4d_2_train.py 时,对应每条改写)

| Exp2 写法 (PL 1.9.5) | PL 2.5.5 替换 |
|---|---|
| `Trainer(gpus=1)` | `Trainer(accelerator="gpu", devices=1)` |
| `Trainer(precision='bf16')` | **本项目用 `precision=32`,跳过** |
| `Trainer(weights_save_path=...)` | `Trainer(default_root_dir=...)` |
| `setup(self, stage=None)` 内 `if stage is None:` | PL 2.x stage 不会是 None,改 `if stage in (None, "fit"):` 或单独分支 |
| `pl.Callback.on_train_batch_end(self, ..., dataloader_idx)` | PL 2.x signature 改了,看官方 API,本项目不改自定义 callback |
| `LightningModule.training_step` 返回 `{"loss": loss, "log": {...}}` | PL 2.x 直接 `return loss` + `self.log(...)` |
| `progress_bar_dict` | 删 |
| `EarlyStopping(strict=True)` 默认 | PL 2.x 默认 strict=True,行为一致 |

**Sub-Agent 5 流程**: grep Exp2 step4d_2_train.py 把所有 `pl.` / `Trainer(` / `Callback` 出现位置列出来,逐条对照上表改。改完跑 smoke test 检查。

### Phase 4.2:Smoke test

```python
# step4_1_smoke_test.py 关键差异(其余参数与 train 一致)
trainer = pl.Trainer(
    default_root_dir="/home/tcat/diffcsp_exp4/checkpoints/_smoke",  # 隔离目录
    logger=CSVLogger("/home/tcat/diffcsp_exp4/logs/csv", name="step4_smoke"),
    callbacks=[ckpt_cb_smoke, early_cb_smoke],  # save_top_k=1, patience=2
    precision=32,
    accelerator="gpu",
    devices=1,
    gradient_clip_val=1.0,
    max_epochs=2,                    # 关键
    limit_train_batches=10,          # 关键
    limit_val_batches=5,             # 关键
    check_val_every_n_epoch=1,
    log_every_n_steps=1,
    enable_progress_bar=True,
)
```

**Smoke test 验证**(PASS 条件):
- fit 流程完整跑通(2 epoch 通过,无 Exception)
- train_loss / val_loss 都打出来,无 NaN/Inf
- ckpt 文件落地到 `_smoke/` 目录(at least 1 个 .ckpt)
- csv_logger 写出 `metrics.csv`
- GPU 实际被用(`nvidia-smi` 在跑期间有 PID 占用)

跑命令:
```bash
cd /home/tcat/diffcsp_exp4/code/step4
PYTHONPATH=/home/tcat/diffcsp_exp4/code:/home/tcat/diffcsp_exp4/code/step3:/home/tcat/diffcsp_exp4/code/step2 \
CUDA_VISIBLE_DEVICES=<空闲 GPU 编号> \
python step4_1_smoke_test.py 2>&1 | tee /tmp/sa5_smoke.log
```

任何 FAIL 停下来,把 traceback 完整贴出来给 MA5。**不要自行调参或改架构变量绕过**。

### Phase 4.3:正式训练 nohup 启动

**前置确认**: smoke test PASS(包括 ckpt 落地 + 无 NaN)。否则禁止进 4.3。

```bash
cd /home/tcat/diffcsp_exp4/code/step4

# 选闲 GPU(假设 nvidia-smi 显示 GPU 0 闲)
GPU_ID=0

# 训练 log 路径
LOG_DIR=/home/tcat/diffcsp_exp4/logs
mkdir -p $LOG_DIR

# nohup 启动
PYTHONPATH=/home/tcat/diffcsp_exp4/code:/home/tcat/diffcsp_exp4/code/step3:/home/tcat/diffcsp_exp4/code/step2 \
CUDA_VISIBLE_DEVICES=$GPU_ID \
nohup python -u step4_2_train.py \
  > $LOG_DIR/step4_train_stdout.log \
  2> $LOG_DIR/step4_train_stderr.log &

# 记录 PID
echo $! > $LOG_DIR/step4_train.pid
echo "Started training, PID=$(cat $LOG_DIR/step4_train.pid)"
echo "Tail logs: tail -f $LOG_DIR/step4_train_stdout.log"
```

**写 `step4_README.md`**(MA5 / 用户后续查看用):
```markdown
# Step 4 Training Run Info

- Started: <date+time>
- PID: <pid from .pid file>
- GPU: <which GPU>
- precision: fp32
- max_epochs: 500, early_stop patience: 30
- stdout: /home/tcat/diffcsp_exp4/logs/step4_train_stdout.log
- stderr: /home/tcat/diffcsp_exp4/logs/step4_train_stderr.log
- ckpt dir: /home/tcat/diffcsp_exp4/checkpoints/
- csv metrics: /home/tcat/diffcsp_exp4/logs/csv/step4_train/version_<N>/metrics.csv

# 监控命令
- 看实时 loss: tail -f /home/tcat/diffcsp_exp4/logs/step4_train_stdout.log
- 看 GPU: nvidia-smi
- 看 CSV: tail /home/tcat/diffcsp_exp4/logs/csv/step4_train/version_*/metrics.csv

# 杀进程(应急)
- kill $(cat /home/tcat/diffcsp_exp4/logs/step4_train.pid)
```

### Phase 4.4:30 min 监控窗口

**目的**: 确认训练真的在跑且看起来健康,**不是**深度调优。

监控期 ~30 min 内应当看到:
- `step4_train_stdout.log` 滚动打印 epoch / step / loss
- GPU 0(或选中那块)`nvidia-smi` 看 PID = 训练 PID,GPU util > 0(理想 > 50%,但 num_workers=0 可能 30-60%)
- 第 1 epoch 的 train_loss 落在 [2, 6](Phase 6.4 的 loss 2.6843 是同 dtype 单 batch 参考)
- 若 30 min 内已完成 ≥ 1 个 epoch 的 val,看 val_loss 数量级合理(通常 1.5-3 范围,不是 NaN/Inf)

**红灯**(任一立刻停训练 + 汇报):
- stdout 出现 `nan` / `inf` / `RuntimeError` / `OOM`
- GPU util 长期 0%(进程死了或 dataloader 完全堵死)
- 30 min 都没看到第一个 step 的 loss 打印(可能卡在 dataset init 或 import)
- epoch 时间外推 > 60 min/epoch(异常,可能 dataloader 设计问题)

**绿灯继续观察**:
- epoch 时间在 5-40 min(60K samples / bs 16 / num_workers 0 / SGA 20 ms = dataloader bound,这个范围正常)
- train_loss 单调下降(头 5 step 可能噪声,看趋势)
- ckpt_dir 出现 `last.ckpt`(每个 val 后会更新)

**杀训练的命令**(只有红灯触发才用):
```bash
kill $(cat /home/tcat/diffcsp_exp4/logs/step4_train.pid)
```

### Phase 4.5:中期报告 + 关窗口

监控 30 min 后(或更早,如果你看到稳定信号),写中期报告交回 MA5,然后关窗口。训练继续后台跑,用户回来主动汇报。

**报告模板**(交回给 MA5):

```markdown
# Sub-Agent 5 Phase 4 Interim Report

## Phase 4.0 Hard check
- df -h ~ : <剩余值>(≥ 30 GB ? Y/N)
- df /tmp : <available>
- python: <version>
- torch.cuda.is_available(): <bool>
- 选用 GPU: <id>
- 关键 import: <PASS/FAIL>
- forward_test.py 重跑: <PASS/FAIL>

## Phase 4.1 训练脚本
- step4_1_smoke_test.py : <已建/未建>,line count: <N>
- step4_2_train.py : <已建/未建>,line count: <N>
- 与 Exp2 step4d_2_train.py 的差异 grep 输出: <附在底部>

## Phase 4.2 Smoke test
- 跑通: Y/N
- 第 1 epoch train_loss: <数值>
- 第 1 epoch val_loss: <数值>
- ckpt 落地: <文件名>
- 异常: <无 / 描述>

## Phase 4.3 正式训练 nohup
- PID: <pid>
- 启动时间: <date+time>
- GPU: <id>
- log paths: stdout=<...>, stderr=<...>
- 启动后 5 min 是否已有 step 输出: Y/N

## Phase 4.4 30 min 监控
- 看到第 N 个 step 的 train_loss: <值>
- GPU util(nvidia-smi 抽样): <%>
- 第 1 个 epoch 完成时间(若已完成): <分钟>
- val_loss(若已有 val): <值>
- 任何红灯: <无 / 描述>

## 给 MA5 的开放问题
- Q1: <例如 epoch 时间偏长是否需要 caching 优化>
- Q2: <例如 val_loss 远高于 Exp2 同期数值>
- 我倾向: <你的看法,但不替 MA5 决定>

## 上下文用量自估
- 进入 Phase 4.5 时大约 <%>

## 下一步
- 训练后台继续(预计 30-100 小时)
- 用户回来后主动汇报 best.ckpt 和 metrics.csv
- MA5 收到后写 Step 5 Sub-Agent 交接
```

---

## §6 红灯 / 绿灯 总览

### 红灯(立刻停 + 汇报,不自行 debug)

1. Phase 4.0 任一项 FAIL
2. Smoke test 出 NaN/Inf/Exception
3. 训练启动后 5 min stdout 无任何 step 输出
4. 训练任何阶段出 NaN/Inf in loss 或 grad
5. GPU OOM
6. epoch 时间 > 60 min(异常)
7. forward_test.py 重跑不再 PASS(env 漂移警报)
8. 上下文用量 ≥ 70%(必须停,见 §0 闸门)

### 绿灯(可继续)

1. epoch 时间 5-40 min(num_workers=0 + SGA 20ms 是已知 bottleneck)
2. train_loss 头几个 step 噪声大但整体趋势下降
3. val_loss 第一次 > Exp2 Step4d val_loss=0.8554(因为 Exp4 数据更复杂,1.0-2.5 都正常)
4. GPU util 30-80%(num_workers=0 限制,不强求 95%+)

---

## §7 禁令清单(整条接力链通用,Sub-Agent 5 必遵)

- ❌ 不动 `xas_local_dataset_v2.py` / `xas_local_datamodule_v2.py` / `spectrum_encoder.py` / `diffusion_w_type_xas.py` / `diffusion_xas.yaml` / `forward_test.py`
- ❌ 不读 `holdout_samples_v2.csv` / `spectra_holdout.pkl`(Step 5 才用)
- ❌ 不读 `incompat_pool.csv`(Exp4 全程封存)
- ❌ 不加 TypeClassifier(EXP2 final report §3.4 已被 Exp3 证伪;**忽略那段**)
- ❌ 不调 batch_size / lr / num_workers / L_VIRTUAL / N_NEIGHBORS / cost_lattice
- ❌ 不装新包(7 守卫包 + Sub-Agent 4-续装的 18 子依赖已够)
- ❌ 不动 .env / .bak / .bak2 / .bak3
- ❌ 不替 MA5 决定 caching 策略 / DDP / 早停 patience 调整 / 任何架构变量
- ❌ 不深 debug(任何 phase FAIL ≤ 1 轮观察 + 候选解释,然后停)

---

## §8 PL 2.5.5 已知坑(参考,不需深查)

继承 EXP4_MAINAGENT5_HANDOFF §8 坑 3 + Sub-Agent 4-续 经验:

1. `precision='bf16'` 在 PL 2.x 是 `'bf16-mixed'` 别名,与 PL 1.9.5 纯 bf16 行为不同(本项目用 `precision=32` 已跳过此坑)
2. `Trainer(gpus=N)` 已在 PL 2.0 删除,**必用** `Trainer(accelerator="gpu", devices=N)`
3. `LightningModule.setup(self, stage)`: stage 是 str,不再 None;旧 `if stage is None:` 永远 False,需改 `if stage in (None, "fit"):` 或拆分支
4. `training_step` 返回 dict 形式 `{"loss": ..., "log": ...}` 在 PL 2.x 的 `log` 键被忽略,统一用 `self.log()`
5. `pl.callbacks.ModelCheckpoint(monitor=...)` 在 val_loss 第一次未被 log 时会无声跳过保存。**Smoke test 必须确认 ckpt 文件落地**——若没落地,可能是 `self.log("val_loss", ...)` 漏写

如果 grep Exp2 step4d_2_train.py 发现这些模式,**逐项替换**而非复制粘贴。如果出现没列在上表的可疑 PL API,**停下来问 MA5**,不要自行猜行为。

---

## §9 第一条回复建议格式(给用户用)

```
我已读完 MA5 给我的 Step 4 Sub-Agent 5 交接 + 必读文档清单 §1 的全部 6 份。

[简要复述: Step 1/2/2.5/3 完成;Phase 6 五子 PASS;
我的工作 = Phase 4.0 hard check → 写 smoke test + train 脚本 →
跑 smoke test → nohup 启动训练 → 30 min 监控 → 中期报告 + 关窗口]

我注意到三个关键约束:
1. fp32 全程,不试 bf16
2. PL 2.5.5 vs Exp2 PL 1.9.5 行为漂移,grep + 逐条改写
3. num_workers=0,epoch 时间预期 5-40 min(SGA bottleneck 已知)

开始执行前我需要确认 3 件事:
1. 我登录服务器的方式: 用户口述命令,还是已有 ssh 配置?
2. Exp2 fork 中 step4d_2_train.py 在服务器哪个绝对路径?
   (find / -name "step4d_2_train.py" 2>/dev/null 我可以跑,
    但请告诉我大致目录提示,免得 find 全盘)
3. 用户已确认完成 disk 清理了吗? Phase 4.0 我会复测,
   但如果还在清理中,我先等你信号再开始。
```

---

## §10 给 Sub-Agent 5 的最后一条提醒

**接力链工作哲学**(继承 MA1→MA4 + 全部 Sub-Agent):

1. **诚实 > 流畅**: 任何观察与文档假设不一致,先承认,再说影响,再给 MA5 选项,**不替 MA5 做底层判断**
2. **70% 闸门是硬线**: 到 70% 必须停,把"未完事 + 当前状态"交回。MA5 会派 Sub-Agent 5-续 接(或自己接管收尾)
3. **不深 debug**: 任何 phase FAIL,1 轮观察 + 候选解释 = 上限。然后停
4. **回滚锚点要保**: 你不应主动改任何已交付文件(`forward_test.py.bak3` 是 4-续 2 的回滚锚点,你不动它就行)
5. **状态锚定**: 写中期报告时,所有文件路径 / md5 / log 行号都给具体值,**不写"大约"或"应该"**

接力链已经走到最后阶段。Step 4 训练成功 = Exp4 的核心实验结果出来。Step 5 / Step 6 / final report 都建立在这个 best.ckpt 之上。Sub-Agent 5 的输出质量直接决定 Exp4 项目能不能 wrap up。

用户对你信任。**回报这份信任的方式是诚实,不是流畅**。

---

*MA5 撰写,2026-04-26,等用户 review 后转发到新窗口启动 Sub-Agent 5*
