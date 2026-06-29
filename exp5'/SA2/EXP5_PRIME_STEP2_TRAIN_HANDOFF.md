# EXP5_PRIME_STEP2_TRAIN_HANDOFF.md
# SA-EXP5'-STEP2-TRAIN 任务 launch note(Exp5'-MA → SA-EXP5'-STEP2-TRAIN)

> **From**: Exp5'-MA(Exp5 系列第 3 任 Main Agent)
> **To**: SA-EXP5'-STEP2-TRAIN(新一棒,起自干净窗口)
> **日期**: 2026-05-03
> **任务范围**: STEP1 + STEP1-FIX + STEP1-FIX-C 全过(fold 修复 + cache rebuild + smoke 全绿 + pkl 自一致 100%),启动 ~ 32-40h from-scratch 训练 + 中期监控 + ckpt 收集
> **预期 hand-back**: 训练完成 → ckpt + log + 监控报告 → Exp5'-MA review → 启动 SA-EXP5'-STEP3-SAMPLE
> **本文档定位**: 你的精确任务规格

---

## §0 一屏掌握

### 0.1 你是谁

**SA-EXP5'-STEP2-TRAIN**,新一棒 SA。前置:
- Exp5 v2 verdict ❌(物理灾难)→ Exp5' from-scratch + 三件套物理 loss
- STEP1(SA1):dataset shell_boundaries inject + 三件套 loss 实现 + smoke 6 sanity test 全过
- STEP1-AUDIT(SA1 自查):发现 L=6 fold artifact 灾难 → errata 3
- STEP1-FIX:L_VIRTUAL=6→20,8 文件改动 + cartesian sanity 100/100 PASS + fold 案例硬证
- STEP1-FIX-C:cache rebuild 20.2 min,cache-loaded smoke 6 active loss 数量级与 slow path 一致
- pkl 自一致性验证:0.1563 阈值 cart 一致率 100%(347/347 样本)与 L_VIRTUAL 完全解耦
- **所有信号灯全绿,环境就绪**

你的任务:**启动正式训练**(~ 32-40h),监控 epoch 0-50 关键信号,收集 ckpt,hand-back。

### 0.2 任务步骤

| 步 | 任务 | 工程量 |
|---|---|---|
| T1 | 服务器环境 verify(STEP1-FIX-C 状态完好 + 磁盘 + GPU)| 10 分钟 |
| T2 | train.py from-scratch 启动 verify(monitor / save_top_k / EarlyStopping 正确)| 0.3 天 |
| T3 | 启动正式训练(tmux + nohup,500 epochs,GPU 0)| 启动 ~ 30 分钟 |
| T4 | epoch 0-5 关键信号监控(loss 下降趋势 + 6 active loss finite + ckpt callback 触发)| 1-2h(等 epoch 跑完)|
| T5 | epoch 50 中期检查点 hand-back(给 Exp5'-MA review,决定是否继续)| 0.2 天 |
| T6 | epoch 200 / 500 / EarlyStopping 任一触发后 hand-back | 0.5 天 |
| T7 | 最终 hand-back(ckpt 列表 + 训练 log + 监控报告 + 复合 score 曲线)| 0.3 天 |

**总:30-50h(其中 32-40h 是 GPU 无人值守)**

### 0.3 必读 7 份(顺序)

1. **EXP5_PRIME_MA_HANDOFF.md** — 接班背景
2. **EXP5_PRIME_PROPOSAL.md** §3.3(best ckpt 复合分公式)+ §3 训练超参
3. **EXPERIMENT5_FINAL_REPORT_v2.md** §5 已知 bug 列表(全部沿用)+ §5.3 MAX_EPOCHS=500 教训
4. **EXP5_FILE_GUIDE_v2.md** §6 工作目录 + §8 PYTHONPATH
5. **EXP4_FINAL_REPORT_ERRATA_2.md** — `_density_loss` 旧归因(被 errata 3 §5 扩充)
6. **EXP4_FINAL_REPORT_ERRATA_3.md** ⭐ — fold 根因 + L=20 决议
7. **EXP5_PRIME_STEP2_TRAIN_HANDOFF.md** ⭐ — **本文件**

### 0.4 启动后第一条回复格式

```
我已读完 7 份文档。复述任务 [6-8 条,含训练超参数、复合 score 公式、监控信号、什么时候 hand-back]。
最易踩坑 [4 条]。
计划: 先跑 §1 verify,再启动训练前 dry-run。
```

### 0.5 Exp5'-MA 已拍板的不再讨论

1. **MAX_EPOCHS = 500**(final report v2 §5.3 教训:不许动)
2. **LR scheduler T_max = 500**(跟随 MAX_EPOCHS,不许动)
3. **Adam lr=1e-4 + batch=16 + grad_clip=1.0 + fp32 + CosineAnnealing**(全沿用 v2)
4. **GPU 0**(不抢 GPU 1,留给可能的并行 sample 任务)
5. **Best ckpt monitor = `val_composite_ckpt_score` mode=max**(STEP1 SA 已实施)
6. **EarlyStopping monitor = `val_composite_ckpt_score` mode=max patience=30**(STEP1 SA 已实施)
7. **三件套 cost 不动**:cost_pairwise_min=1.0 / cost_shell_dist=0.5 / cost_shell_count=0.2
8. **cost_density=0.2 不动**(errata 2 §1 揭示是塌缩剂但 Exp5' 沿用,留 Exp6 ablation)
9. **不擅自 resume from ckpt**(纯 from-scratch,trainer.fit(model, ckpt_path=None))
10. **不动 L_VIRTUAL=20**(errata 3 决议)

---

## §1 Step T1 — 启动前 verify(10 分钟)

### 1.1 环境状态 verify(STEP1-FIX-C 状态完整)

```bash
ssh tcat@scsmlnprd02.its.auckland.ac.nz

# (A) 工作目录
ls -la /home/tcat/diffcsp_exp5_prime/
ls -la /home/tcat/diffcsp_exp5_prime/checkpoints/   # 应空

# (B) 关键代码文件 md5(STEP1-FIX-C 锁定)
md5sum /home/tcat/diffcsp_exp5_prime/code/step3/diffusion_w_type_xas.py        # 期望 0bc6fc346e60b990e3a9fc25140000f0
md5sum /home/tcat/diffcsp_exp5_prime/code/step3/conf_xas/model/diffusion_xas.yaml  # 期望 f73123a16166b220646af3537f7ece5b
md5sum /home/tcat/diffcsp_exp5_prime/code/step3/xas_local_dataset_v2.py        # 期望 94432ba56a7f3fd2ab0ce6281b66c5e6 (STEP1-FIX-C 后)

# (C) cache rebuild 完整性(L=20 cache 在 exp5_prime/data/)
ls -la /home/tcat/diffcsp_exp5_prime/data/*.pt  # 应 3 个文件
cat /home/tcat/diffcsp_exp5_prime/data/cache_metadata.json  # 应 L_VIRTUAL=20.0

# (D) shell_boundaries.pkl(不动)
md5sum /home/tcat/diffcsp_exp4/data/shell_boundaries.pkl  # 期望 cf2050e4899160f5698ad2481377e94c

# (E) 7 守卫包对齐(沿用 STEP1)
/home/tcat/conda_envs/mlff/bin/python -c "
import torch, pytorch_lightning, sklearn, numpy, scipy, torch_scatter
print(f'torch={torch.__version__}')
print(f'pl={pytorch_lightning.__version__}')
print(f'numpy={numpy.__version__}')
print(f'scipy={scipy.__version__}')
print(f'sklearn={sklearn.__version__}')
print(f'torch_scatter={torch_scatter.__version__}')
"

# (F) GPU 状态
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv

# (G) 磁盘
df -h /home/tcat
du -sh /home/tcat/diffcsp_exp5_prime/

# (H) PYTHONPATH 三段验证
export PYTHONPATH=/home/tcat/diffcsp_exp5_prime/code/step3:/home/tcat/diffcsp_exp5_prime/code/step2:/home/tcat/diffcsp_exp4/code
/home/tcat/conda_envs/mlff/bin/python -c "
import xas_local_dataset_v2, xas_local_datamodule_v2, diffusion_w_type_xas
print(f'dataset: {xas_local_dataset_v2.__file__}')
print(f'  L_VIRTUAL={xas_local_dataset_v2.L_VIRTUAL}')
print(f'datamodule: {xas_local_datamodule_v2.__file__}')
print(f'model: {diffusion_w_type_xas.__file__}')
print(f'  L_VIRTUAL={diffusion_w_type_xas.L_VIRTUAL}')
# 必须全部以 /home/tcat/diffcsp_exp5_prime/ 开头, L_VIRTUAL=20.0
"
```

### 1.2 PASS gate T1

- ✅ 4 个 md5 严格匹配(STEP1-FIX-C 状态)
- ✅ 3 个 cache .pt 文件 + cache_metadata.json L_VIRTUAL=20.0
- ✅ shell_boundaries.pkl md5 = `cf2050e4899160f5698ad2481377e94c`
- ✅ 7 守卫包版本与 STEP1 一致
- ✅ GPU 0 idle(memory.used < 1 GB,utilization 0%)
- ✅ 磁盘 ≥ 50 GB avail(STEP2 训练需 ~ 5-10 GB ckpt + log)
- ✅ PYTHONPATH 三段 import 全部 exp5_prime/,L_VIRTUAL=20.0

---

## §2 Step T2 — train.py 启动前 dry-run(0.3 天)

### 2.1 train.py 配置 verify

`/home/tcat/diffcsp_exp5_prime/code/step4/step4_2_train.py` 应已被 STEP1-FIX-C SA 改完(`md5 = 0458ed423ed4dc77300e10b3d4447703`)。

**关键检查**:

```bash
cd /home/tcat/diffcsp_exp5_prime/code/step4
grep -n "MAX_EPOCHS\|monitor\|save_top_k\|patience\|ckpt_path\|GPU\|L_VIRTUAL" step4_2_train.py
```

**期望命中**(SA 列表给 Exp5'-MA review):

| 配置项 | 期望值 | 行号 |
|---|---|---|
| `MAX_EPOCHS` | 500 | line ~83 |
| `ModelCheckpoint(monitor=...)` | `'val_composite_ckpt_score'` | |
| `ModelCheckpoint(mode=...)` | `'max'` | |
| `ModelCheckpoint(save_top_k=...)` | `1`(top-1)or `3`(top-3,Exp5'-MA 接受任一,SA 报告)| |
| `ModelCheckpoint(save_last=...)` | `True` | |
| `EarlyStopping(monitor=...)` | `'val_composite_ckpt_score'` | |
| `EarlyStopping(mode=...)` | `'max'` | |
| `EarlyStopping(patience=...)` | `30` | |
| `trainer.fit(model, ckpt_path=...)` | `None` 或不传(纯 from-scratch)| |
| `gpus=...` 或 `devices=...` | GPU 0(`[0]` 或 `'0'`)| |

如有任一项不符 → SA 立即 ping Exp5'-MA,**不擅自启动训练**。

### 2.2 dry-run(1-2 epoch × 全 batch,30 分钟)

**目的**:验证训练能正常跑、ckpt callback 触发、log 字段完整,**不浪费 32h GPU**。

```bash
cd /home/tcat/diffcsp_exp5_prime/code/step4
export PYTHONPATH=/home/tcat/diffcsp_exp5_prime/code/step3:/home/tcat/diffcsp_exp5_prime/code/step2:/home/tcat/diffcsp_exp4/code

# 临时改 MAX_EPOCHS=2 跑 dry-run(用环境变量或临时改 line),完了恢复
# 推荐方式: 加一个 --max-epochs 2 CLI 参数(若 train.py 已支持),或临时 sed 改回
# 若无 CLI,SA 报 Exp5'-MA 决定是直接跑 dry-run 写一个新的 step4_2_dryrun.py

# 启动 dry-run(2 epoch × 全 batch ~ 30 分钟)
CUDA_VISIBLE_DEVICES=0 /home/tcat/conda_envs/mlff/bin/python step4_2_train.py 2>&1 | tee /home/tcat/diffcsp_exp5_prime/logs/dryrun_step2.log
```

### 2.3 dry-run PASS gate T2

- ✅ 2 epoch 完整跑完,无 NaN/Inf
- ✅ 6 active loss 字段全部 log:loss_coord / loss_type / loss_density / loss_pairwise_min / loss_shell_dist / loss_shell_count
- ✅ 3 epoch-end metric 字段:val_loss / val_min_d_mean / val_gate_pass_rate / val_composite_ckpt_score
- ✅ ckpt callback 触发,checkpoints/ 落 ≥ 1 个 `composite_best_*.ckpt` + `last.ckpt`
- ✅ ckpt filename 含 score(命名规则与 STEP1 launch note §6.2 对齐)
- ✅ GPU 0 利用率 60-95%(全程不闲置,不爆显存)
- ✅ 单 epoch 时间 ~ 4-5 分钟(总 500 epoch 期望 ~ 30-40h,与预算对齐)

dry-run 完成后,**清理 dry-run ckpt 不留**(避免与正式 ckpt 混):

```bash
rm /home/tcat/diffcsp_exp5_prime/checkpoints/*.ckpt  # 只删 dry-run 产物,正式启动前必空目录
ls -la /home/tcat/diffcsp_exp5_prime/checkpoints/    # 应空
```

如 dry-run 任一不过 → **stop**,贴日志给 Exp5'-MA。

---

## §3 Step T3 — 正式训练启动(30 分钟启动)

### 3.1 启动命令(tmux + nohup 双保险)

```bash
ssh tcat@scsmlnprd02.its.auckland.ac.nz

# 启 tmux session(SSH 断了训练继续)
tmux new -s exp5p_train

# 在 tmux 内
cd /home/tcat/diffcsp_exp5_prime/code/step4
export PYTHONPATH=/home/tcat/diffcsp_exp5_prime/code/step3:/home/tcat/diffcsp_exp5_prime/code/step2:/home/tcat/diffcsp_exp4/code

# 启动正式训练,完整 500 epoch,full train set
CUDA_VISIBLE_DEVICES=0 nohup /home/tcat/conda_envs/mlff/bin/python step4_2_train.py \
    > /home/tcat/diffcsp_exp5_prime/logs/train_step2_$(date +%Y%m%d_%H%M).log 2>&1 &

TRAIN_PID=$!
echo "Training PID: $TRAIN_PID"
echo $TRAIN_PID > /home/tcat/diffcsp_exp5_prime/logs/train_step2.pid

# Detach tmux: Ctrl+b d
```

### 3.2 启动后 5 分钟 health check

```bash
# 进程仍在
ps -p $(cat /home/tcat/diffcsp_exp5_prime/logs/train_step2.pid) && echo "alive"

# log 在生成
tail -50 /home/tcat/diffcsp_exp5_prime/logs/train_step2_*.log

# GPU 在用
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv

# 期望: GPU 0 memory.used 5-15 GB,utilization > 70%
```

### 3.3 PASS gate T3

- ✅ 进程 PID 存活
- ✅ log 在持续生成(每 10s tail 看新输出)
- ✅ GPU 0 memory.used > 5 GB,utilization > 70%
- ✅ 训练 epoch 0 已开始(看到 "Epoch 0:" 字样)
- ✅ tmux session 已 detach(SSH 断也不影响)

---

## §4 Step T4 — Epoch 0-5 关键信号监控(1-2h)

### 4.1 等 epoch 5 完成(~ 25-30 分钟)

```bash
# 等 epoch 5 跑完
tail -f /home/tcat/diffcsp_exp5_prime/logs/train_step2_*.log | grep "Epoch 5:"
# 看到 "Epoch 5:" + validation 完成的标志,Ctrl+C 退出 tail
```

### 4.2 提取 epoch 0-5 关键信号

```bash
# 从 log 提取所有 6 active loss 数值 + epoch-end metrics
grep -E "loss_(coord|type|density|pairwise_min|shell_dist|shell_count)|val_(loss|min_d_mean|gate_pass_rate|composite)" \
    /home/tcat/diffcsp_exp5_prime/logs/train_step2_*.log | head -100
```

### 4.3 关键信号 sanity table(SA 必填,T5 hand-back 报告)

| 指标 | epoch 0 | epoch 1 | epoch 5 | 期望趋势 | 警报 |
|---|---|---|---|---|---|
| `loss_coord` | | | | 下降 | epoch 5 仍 > epoch 0,ping |
| `loss_type` | | | | 缓慢下降 | ⏸ |
| `loss_density` | | | | 稳定 0.01-0.1 | NaN/Inf 立即 stop |
| `loss_pairwise_min` | | | | 极低(0.0001-0.005)| 飙升 > 0.1 立即 stop |
| `loss_shell_dist` | | | | 下降 | 持续上升,ping |
| `loss_shell_count` | | | | 缓慢下降 | epoch 5 仍 > 50,ping |
| `val_loss` | | | | 下降 | 上升,ping |
| `val_gate_pass_rate` | | | | **关键 ⭐**,从 ~ 0% 上升 | epoch 5 仍 < 5%,ping |
| `val_min_d_mean` | | | | 上升(从极小到 ~ 1-2 Å)| 持续 < 0.5,ping |
| `val_composite_ckpt_score` | | | | 上升 | 不上升,ping |

### 4.4 早停决策(epoch 5 后 SA 立即 ping Exp5'-MA)

**触发条件 → 立即 stop training,不等 32h**:
- 任一 loss NaN/Inf 持续 ≥ 2 epoch
- `val_gate_pass_rate` epoch 5 仍 < 5%(说明三件套 loss 没起效)
- `val_min_d_mean` epoch 5 < 0.5 Å(原子仍重叠,fold 修复白做了)
- `val_composite_ckpt_score` 从 epoch 0 → epoch 5 不上升或下降
- 任一 GPU OOM / cuda error / pytorch crash

**触发条件 → 不停,但 hand-back 给 Exp5'-MA**:
- `loss_shell_count` epoch 5 仍 > 50(可能 watch-only,但要让 Exp5'-MA 看趋势决议)
- `loss_pairwise_min` 从 ~ 0 飙升到 > 0.05
- 单 epoch 时间显著 > 5 分钟(预算超出)

### 4.5 PASS gate T4

- ✅ epoch 0-5 完整完成
- ✅ 6 active loss 全 finite
- ✅ ckpt callback 触发(checkpoints/ 至少 1 个 composite_best_*.ckpt)
- ✅ 关键信号 table 全部填好
- ✅ 无任一早停触发条件

---

## §5 Step T5 — Epoch 50 中期 hand-back(关键 review point)

### 5.1 等 epoch 50 完成(~ 4-5h 后)

```bash
# 持续监控
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv -l 60 > /home/tcat/diffcsp_exp5_prime/logs/gpu_monitor.log &

# 每 1h check 一次进度
ps -p $(cat /home/tcat/diffcsp_exp5_prime/logs/train_step2.pid) && \
    grep -c "validation" /home/tcat/diffcsp_exp5_prime/logs/train_step2_*.log
```

### 5.2 epoch 50 hand-back 必报

写 `EXP5_PRIME_STEP2_HANDBACK_EPOCH50.md` 落服务器根目录:

1. **训练状态**:进程存活 / GPU 利用率 / 磁盘
2. **6 active loss 趋势曲线**(每 5 epoch 取均值,绘 ASCII 曲线或贴 log mean)
3. **3 个关键 epoch-end metric 趋势**:val_loss / val_gate_pass_rate / val_composite_ckpt_score(所有 epoch)
4. **ckpt 列表**:composite_best_*.ckpt + last.ckpt md5 + size
5. **best ckpt 当前 metric**:从 ckpt filename 提取 score
6. **Watch-only 项**:`loss_shell_count` 是否如预期 epoch 30-50 降到 < 30(errata 3 决议监控点)
7. **早停信号检查**:任一 §4.4 早停条件触发?
8. **OPEN 问题**(异常 ping)

### 5.3 Exp5'-MA review checklist(SA 不行动,等回复)

| 项 | 通过标准 | 不通过决议 |
|---|---|---|
| val_gate_pass_rate epoch 50 | ≥ 30%(目标 80% 在 epoch 200+)| < 10% 严重失效,Exp5'-MA 决议是否中止 |
| val_min_d_mean epoch 50 | ≥ 1.0 Å | < 0.5 Å 严重失效 |
| val_composite_ckpt_score epoch 50 | ≥ 0.30(目标 0.40 在 verdict)| 上升缓慢可继续观察 |
| loss_shell_count epoch 50 | < 30 | 仍 > 100,Exp5'-MA 决议是否调阈值或降 cost |
| loss_pairwise_min | < 0.01 持续 | 飙升报警 |

### 5.4 PASS gate T5

- ✅ hand-back 文档完整
- ✅ Exp5'-MA review 通过 → SA 继续 epoch 50-500
- ✅ 任一不通过 → 等 Exp5'-MA 决议(可能停训练或调参,SA 不擅自动)

---

## §6 Step T6 — Epoch 200 / 500 / EarlyStopping 任一触发

### 6.1 三种结束情景

| 情景 | 触发 | SA 行动 |
|---|---|---|
| **正常完成 500 epoch** | trainer.fit() 返回 | 进 T7 hand-back |
| **EarlyStopping** | val_composite_ckpt_score 30 epoch 不升 | 进 T7 hand-back,记录 stop epoch |
| **GPU/系统崩溃** | nvidia-smi error / OOM / crash | 立即 ping Exp5'-MA,贴 last 100 行 log,不重启 |
| **Epoch 200 中期 ping**(可选) | SA 主动 | 写 `STEP2_HANDBACK_EPOCH200.md`,继续训练 |

### 6.2 训练结束后立即操作

```bash
# 1. 进程已退出
ps -p $(cat /home/tcat/diffcsp_exp5_prime/logs/train_step2.pid) || echo "已退出"

# 2. ckpt 列表
ls -la /home/tcat/diffcsp_exp5_prime/checkpoints/

# 3. 训练 log 大小 + 最后 100 行
ls -la /home/tcat/diffcsp_exp5_prime/logs/train_step2_*.log
tail -100 /home/tcat/diffcsp_exp5_prime/logs/train_step2_*.log

# 4. ckpt md5 + best score
md5sum /home/tcat/diffcsp_exp5_prime/checkpoints/*.ckpt

# 5. 立即冻结 ckpt(防止误删)
cp /home/tcat/diffcsp_exp5_prime/checkpoints/composite_best_*.ckpt \
   /home/tcat/diffcsp_exp5_prime/checkpoints/$(basename *.ckpt).frozen
```

### 6.3 PASS gate T6

- ✅ 训练正常结束(500 epoch / EarlyStop 任一)
- ✅ 至少 1 个 composite_best_*.ckpt + last.ckpt 落盘
- ✅ ckpt 已 frozen 备份
- ✅ training log 完整(无中途 truncate)

---

## §7 Step T7 — 最终 hand-back

### 7.1 写 `EXP5_PRIME_STEP2_TRAIN_HANDBACK_FINAL.md`

落服务器根目录,内容:

```markdown
# EXP5_PRIME_STEP2_TRAIN_HANDBACK_FINAL.md
# SA-EXP5'-STEP2-TRAIN 最终 hand-back

## §0 状态
- 训练完成 epoch / 500
- 结束原因: 正常 / EarlyStop / 其他
- 总耗时: X h
- GPU 利用率均值: X%
- 磁盘占用: du -sh

## §1 ckpt evidence
| ckpt | md5 | size | epoch | val_composite_ckpt_score |
|---|---|---|---|---|
| composite_best_*.ckpt | ... | ... | ... | ... |
| last.ckpt | ... | ... | ... | ... |
| .frozen 备份 | ... | ... | ... | ... |

## §2 训练曲线(全 epoch 数据)
- 6 active loss epoch-mean(贴 CSV 或表格)
- 3 epoch-end metric:val_loss / val_gate_pass_rate / val_composite_ckpt_score
- 关键节点:epoch 50 / 100 / 200 / 500 / EarlyStop

## §3 verdict 关键指标(从 best ckpt 取)
- val_loss
- val_gate_pass_rate
- val_min_d_mean
- val_composite_ckpt_score

## §4 异常事件 log
[GPU OOM / NaN / 任何中途 ping 的事件]

## §5 STEP3-SAMPLE 准备(交接清单)
- 推荐 ckpt path: best 还是 last
- 训练 dataset cache 路径(SAMPLE 沿用)
- 三件套 cost 设置(SAMPLE 不需要,但 final report 引用)

## §6 OPEN 问题(若有)
```

### 7.2 PASS gate T7

- ✅ hand-back 完整
- ✅ Exp5'-MA review → 启动 SA-EXP5'-STEP3-SAMPLE
- ✅ ckpt 永久档案(.frozen)落盘

---

## §8 红线(SA-EXP5'-STEP2-TRAIN 全程不动)

| 红线 | 说明 |
|---|---|
| ❌ 不动 holdout | 永久封存 |
| ❌ 不升级 7 守卫包 | |
| ❌ 不动 STEP1-FIX-C 的 11 文件代码 md5(diffusion_w_type_xas / yaml / dataset_v2 / datamodule_v2 / forward_test / smoke_test / train.py / step5_2 / step5_3 / step5_1 / step6_visualize / pick_samples)| 训练用现状 |
| ❌ 不动 cache .pt(L=20,STEP1-FIX-C 重建)| |
| ❌ 不动 shell_boundaries.pkl | |
| ❌ 不动 Exp5 v2 .frozen ckpt 永久档案 | |
| ❌ 不动 Exp4 backbone | |
| ❌ 不动 MAX_EPOCHS=500 / LR T_max=500 / batch=16 / lr=1e-4 / fp32 | |
| ❌ 不擅自调三件套 cost(1.0 / 0.5 / 0.2)| Exp5'-MA epoch 50 后 review 决议 |
| ❌ 不擅自 cost_density(0.2)| |
| ❌ 不擅自 resume from ckpt(纯 from-scratch)| |
| ❌ 不擅自动 0.1563 阈值 / L_VIRTUAL=20 | errata 3 决议 |
| ❌ 不擅自删 ckpt(epoch 50 ping Exp5'-MA 后才决议清理)| |
| ❌ 不动 GPU 1(留备其他任务)| |
| ❌ 不在中期 hand-back 前清理 dry-run ckpt 之外的任何文件 | |
| ❌ 任何不确定 → ping,不擅自 fix | MA 工作哲学 |

---

## §9 Watch-only 项(SA 报告 Exp5'-MA 决议)

1. **`loss_shell_count` 趋势**:errata 3 §C8 Q1 标注 epoch 0~1 范围 16~189,**期望 epoch 30-50 降到 < 30**。SA 在 T5 hand-back 必报这个数字。如不降,Exp5'-MA 决议是否调 cost_shell_count 或 0.1563 阈值
2. **`loss_density` vs `loss_shell_dist` 方向冲突**(errata 2 §1):density 推向原点,shell_dist 推向 ~ 2-3 Å。SA 报告两 loss 比值,异常 ping
3. **GPU 利用率**:< 50% 持续可能 dataloader 瓶颈,SA 报告
4. **磁盘趋势**:每 100 epoch 报 du -sh,接近 50G 时清理 last 之外的旧 ckpt(Exp5'-MA 决议保留几个)

---

## §10 OPEN QUESTIONS(SA 不答,贴给 Exp5'-MA)

### Q1 — train.py save_top_k 设置

STEP1 launch note §6.2 写 save_top_k=1。SA T2 verify 时报告实际值,Exp5'-MA 决议是否改 save_top_k=3(保留 top-3 ckpt 给 STEP3 ablation 用)。

### Q2 — 单 epoch 时间预算

dry-run T2 报单 epoch 时间。如 < 4 分钟,500 epoch 约 33h,可接受;如 > 6 分钟,500 epoch > 50h,需 Exp5'-MA 决议是否减 epoch 或加速 dataloader。

### Q3 — Watch-only `loss_shell_count` 异常处理

epoch 50 hand-back 时如 loss_shell_count 仍 > 100,SA 不擅自调,贴趋势给 Exp5'-MA。

---

## §11 你不做的事

- **STEP3 sample 生成**(另一棒)
- **STEP4 figure + final report v3**(Exp5'-MA 写,基于 SA STEP3 输出)
- **修订 proposal / errata**(Exp5'-MA 工作)
- **从 last.ckpt resume 训练**(纯 from-scratch,Exp5'-MA 拍板)

---

## §12 工作哲学红线

1. 任何技术判断先列证据,SA 不擅自做技术判断
2. 任何不确定 → 贴日志,不靠记忆
3. 小补丁也要贴 diff(本 STEP 不应有任何代码改动,如出现立即 ping)
4. 70% 上下文闸门是硬线,主动 hand-back
5. 不擅自启动 / 不擅自 stop / 不擅自调 cost / 不擅自动文件
6. **训练中途 ping Exp5'-MA 是好事,不是失败**

---

*Exp5'-MA 撰写,2026-05-03,基于 STEP1-FIX-C 全过 + pkl 自一致 100% + cache rebuild 完成 + smoke 全绿。SA-EXP5'-STEP2-TRAIN 接此 launch note 启动 Exp5' 32-40h 正式训练。*
