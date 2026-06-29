# Exp4 Step 4 训练优化工作报告

**报告时间**：2026-04-26
**优化结论**：训练速度提升约 4×（单 epoch 19 min → ~5 min）
**精度验证**：缓存路径与原路径**逐位一致**（10 样本 bit-exact match）
**当前状态**：训练正在后台运行，无需再次调整

---

## 0. 快速导航

| 你需要 | 看哪一节 |
|---|---|
| 当前正在跑的训练状态 | §1 |
| 我们一共做了什么 | §2 |
| 所有文件的位置 + 备份 | §3 |
| 怎么监控训练进度 | §4 |
| 训练结束后怎么办 | §5 |
| 出问题怎么回滚 | §6 |

---

## 1. 当前训练状态

**进程**
- PID 文件：`/home/tcat/diffcsp_exp4/logs/step4_train.pid`
- 启动时间：2026-04-26 07:19
- 检查命令：`ps -p $(cat /home/tcat/diffcsp_exp4/logs/step4_train.pid) -o pid,etime,rss,cmd`

**最新成果（截至 T+7min）**
- epoch 0: val_loss = 1.0342（来自原 19 分钟训练）
- epoch 1: val_loss = 0.98457（优化后 5 分钟训练完）
- 当前 step ≈ 8949
- best ckpt: `/home/tcat/diffcsp_exp4/checkpoints/best-epoch001-val0.9846.ckpt`

**速度对比**

| 指标 | 优化前 | 优化后 | 提升 |
|---|---|---|---|
| 数据加载延迟（每样本） | 16.23 ms | 0.15 ms | 111× |
| 单 epoch 时间 | ~19 min | ~5 min | ~4× |
| 总训练时间预估 | 几天 | 8-17 小时 | ~90% 缩短 |

---

## 2. 我们做了什么（按时间顺序）

### 2.1 诊断阶段
- 确认进程存活、metrics.csv 在写、stderr 无错
- 定位瓶颈：GPU util 0–10%，瓶颈在 **dataset `__getitem__` 里的 POSCAR 解析 + SpacegroupAnalyzer**（每样本 11–23 ms）
- 确认 ckpt 已保存（epoch 0），可安全重启

### 2.2 优化阶段（三个独立步骤）

**步骤 1：预计算结构缓存**
- 新建脚本：`/home/tcat/diffcsp_exp4/code/step4_exp4/precompute_structure_cache.py`
- 一次性把所有样本的 `frac_coords / atom_types / feff_scaled / valid_mask` 算好存盘
- 输出：`{train,val,test}_structure_cache.pt` 共 53.5 MB
- 总耗时：13.1 分钟（一次性）
- 数据完整性：train 99.99%、val 99.96%、test 100%

**步骤 2：让 Dataset 用上缓存**
- 修改 `/home/tcat/diffcsp_exp4/code/step3/xas_local_dataset_v2.py`
- 添加 `use_cache` 参数和 `EXP4_USE_CACHE` 环境变量开关
- 添加 fast-path：缓存存在时跳过 POSCAR/SGA/get_neighbors/scaler.transform
- 添加 sample_order 一致性校验（防止使用过期缓存）
- 验证：10 样本与原路径 bit-exact match，speedup 111.9×

**步骤 3：优化 DataLoader 参数**
- 修改 `/home/tcat/diffcsp_exp4/code/step3/xas_local_datamodule_v2.py`
  - 添加 `pin_memory=True`（CPU→GPU 传输加速）
  - 添加 `persistent_workers=(num_workers > 0)`（worker 不在 epoch 间销毁）
- 修改 `/home/tcat/diffcsp_exp4/code/step4_exp4/step4_2_train.py`
  - `NUM_WORKERS = 0` → `NUM_WORKERS = 8`

### 2.3 不变的部分（重要！数学等价性保证）

以下任何东西**都没动**，所以训练动力学与原版完全一致：

- 模型架构（CSPNet, decoder, time_embedding, spectrum_encoder）
- 优化器（Adam, lr=1e-4, betas=[0.9, 0.999], weight_decay=0）
- LR scheduler（CosineAnnealingLR, T_max=500, eta_min=1e-6）
- batch_size（16）
- precision（fp32）
- gradient_clip_val（1.0）
- max_epochs（500）
- early_stop patience（30）
- 所有 hparams.yaml 字段
- 随机种子（保持原行为）
- collate_fn 行为（None 样本过滤）

---

## 3. 文件清单与备份

### 3.1 训练核心文件（修改过的）

| 文件 | 用途 | 修改 |
|---|---|---|
| `/home/tcat/diffcsp_exp4/code/step4_exp4/step4_2_train.py` | 主训练脚本 | NUM_WORKERS: 0 → 8 |
| `/home/tcat/diffcsp_exp4/code/step3/xas_local_dataset_v2.py` | Dataset 类 | 添加缓存 fast-path |
| `/home/tcat/diffcsp_exp4/code/step3/xas_local_datamodule_v2.py` | DataModule 类 | 添加 pin_memory, persistent_workers |

### 3.2 新增文件

| 文件 | 用途 |
|---|---|
| `/home/tcat/diffcsp_exp4/code/step4_exp4/precompute_structure_cache.py` | 离线预计算脚本，可重复运行 |
| `/home/tcat/diffcsp_exp4/code/step4_exp4/monitor.py` | 实时进度条（可选） |
| `/home/tcat/diffcsp_exp4/data/train_structure_cache.pt` | 训练集缓存（44.6 MB） |
| `/home/tcat/diffcsp_exp4/data/val_structure_cache.pt` | 验证集缓存（5.6 MB） |
| `/home/tcat/diffcsp_exp4/data/test_structure_cache.pt` | 测试集缓存（3.3 MB） |

### 3.3 备份文件清单（按时间）

**A. ckpt 备份**
| 文件 | 时间 | 内容 |
|---|---|---|
| `/home/tcat/diffcsp_exp4/checkpoints.bak_20260426_0653/` | 06:53 | kill 旧训练前的 epoch 0 ckpt 备份 |

**B. 代码备份**
| 文件 | 时间 | 备份原因 |
|---|---|---|
| `/home/tcat/diffcsp_exp4/code/step3/xas_local_dataset_v2.py.bak_phase46` | 4-26 06:13 | 你之前的 phase4.6 备份（与本次优化无关） |
| `/home/tcat/diffcsp_exp4/code/step3/xas_local_dataset_v2.py.bak_before_cache_20260426_0714` | 07:14 | 加缓存逻辑前的版本 |
| `/home/tcat/diffcsp_exp4/code/step3/xas_local_datamodule_v2.py.bak_phase46` | 4-26 06:14 | 你之前的 phase4.6 备份（与本次优化无关） |
| `/home/tcat/diffcsp_exp4/code/step3/xas_local_datamodule_v2.py.bak_2026XXXX_XXXX` | 07:1X | 加 pin_memory/persistent_workers 前的版本 |
| `/home/tcat/diffcsp_exp4/code/step4_exp4/step4_2_train.py.bak_2026XXXX_XXXX` | 07:1X | 改 NUM_WORKERS 前的版本 |

> 注：`.bak_2026XXXX_XXXX` 的精确时间戳需要 `ls -la` 查看实际文件名。

**C. 日志备份**
| 文件 | 内容 |
|---|---|
| `/home/tcat/diffcsp_exp4/logs/step4_train_stdout.log.bak_20260426_0719` | 旧训练（19 min/epoch）的 stdout |
| `/home/tcat/diffcsp_exp4/logs/step4_train_stderr.log.bak_20260426_0719` | 旧训练 stderr |
| `/home/tcat/diffcsp_exp4/logs/step4_train.pid.bak_20260426_0719` | 旧训练 PID |

### 3.4 当前活跃日志

| 文件 | 内容 |
|---|---|
| `/home/tcat/diffcsp_exp4/logs/step4_train_stdout.log` | 优化后训练的 stdout |
| `/home/tcat/diffcsp_exp4/logs/step4_train_stderr.log` | 优化后训练的 stderr（含 epoch progress） |
| `/home/tcat/diffcsp_exp4/logs/step4_train.pid` | 优化后训练的 PID（3285027） |
| `/home/tcat/diffcsp_exp4/logs/csv/step4_train/version_2/metrics.csv` | Lightning CSVLogger 输出（每 50 step 一行） |

### 3.5 当前活跃 ckpt

| 文件 | 内容 |
|---|---|
| `/home/tcat/diffcsp_exp4/checkpoints/last.ckpt` | 最新进度（断电恢复用） |
| `/home/tcat/diffcsp_exp4/checkpoints/best-epoch001-val0.9846.ckpt` | epoch 1 的最佳 ckpt（val_loss=0.98457） |
| `/home/tcat/diffcsp_exp4/checkpoints/best-epoch000-val1.0342.ckpt` | epoch 0 的最佳 ckpt（旧训练遗留，会在更好的 ckpt 出现时被自动删除） |

---

## 4. 监控训练进度

### 4.1 一次性 snapshot 检查

```bash
# 进程 + GPU + metrics 一键检查
ps -p $(cat /home/tcat/diffcsp_exp4/logs/step4_train.pid) -o pid,etime,rss,cmd
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
LATEST=$(ls -td /home/tcat/diffcsp_exp4/logs/csv/step4_train/version_* | head -1)
tail -n 2 "$LATEST/metrics.csv"
ls -la /home/tcat/diffcsp_exp4/checkpoints/best-*.ckpt | tail -3
```

### 4.2 Epoch 进度（最直观）

```bash
grep "global step.*val_loss.*reached" /home/tcat/diffcsp_exp4/logs/step4_train_stderr.log | tail -10
```

每行一个 epoch，能看到 val_loss 的改善曲线。

### 4.3 实时进度条（需要新开 SSH 终端）

```bash
python /home/tcat/diffcsp_exp4/code/step4_exp4/monitor.py
```

注意：进度条显示的总进度按 max_epochs=500 计算，但因为 EarlyStopping patience=30，实际可能 100-200 epoch 就早停了。所以进度条到 30%-40% 训练就可能结束。

---

## 5. 训练结束后的工作

训练结束的标志：
- `ps` 进程消失
- stderr 出现 `Trainer 训练完成` / `EarlyStopping` 信息
- `/home/tcat/diffcsp_exp4/best_checkpoint_path.txt` 文件被写入（脚本最后一步）

训练结束后：
1. **找到最佳 ckpt**：`cat /home/tcat/diffcsp_exp4/best_checkpoint_path.txt`
2. **清理**（可选）：旧的 ckpt 备份目录 `checkpoints.bak_20260426_0653/` 可以删除
3. **保留**：`*_structure_cache.pt` 三个文件如果之后还要再训练就保留；如果不用了可以删

---

## 6. 出问题怎么回滚

### 6.1 完全回到优化前

```bash
# 停止当前训练
kill $(cat /home/tcat/diffcsp_exp4/logs/step4_train.pid)
sleep 15

# 恢复代码（用 .bak_before_cache_* 文件，注意替换实际时间戳）
cp /home/tcat/diffcsp_exp4/code/step3/xas_local_dataset_v2.py.bak_before_cache_20260426_0714 \
   /home/tcat/diffcsp_exp4/code/step3/xas_local_dataset_v2.py
# (类似恢复 datamodule 和 train script，看 ls -la 找实际备份文件名)

# 恢复 ckpt（如果当前 ckpt 出问题）
rm -rf /home/tcat/diffcsp_exp4/checkpoints
cp -r /home/tcat/diffcsp_exp4/checkpoints.bak_20260426_0653 /home/tcat/diffcsp_exp4/checkpoints
```

### 6.2 仅禁用缓存（不需要恢复代码）

```bash
# 重启训练时不传 EXP4_USE_CACHE，或显式传 0：
EXP4_USE_CACHE=0 nohup python -u step4_2_train.py > ... &
```

或者直接删除缓存文件：
```bash
mv /home/tcat/diffcsp_exp4/data/*_structure_cache.pt /tmp/  # 移走，不删
```

dataset 代码会检测缓存不存在，**自动 fallback 到原 POSCAR+SGA 路径**（这是 patch 时设计的兜底逻辑）。

---

## 7. 设计决策记录（供将来参考）

为什么这么做：

| 决策 | 理由 |
|---|---|
| 缓存改 dataset 而不是 datamodule | dataset 是数据真值的源头，缓存与运行时数据生成强耦合 |
| 通过 `ds[idx]` 调用而不是复制逻辑做缓存 | 保证数学等价：缓存数据完全来自原 `__getitem__`，没有重新实现 |
| 用 `EXP4_USE_CACHE` 环境变量做开关 | 不修改训练脚本就能禁用缓存（比如做对照实验） |
| 校验 sample_order 一致性 | 防止 csv 改了但缓存没更新导致样本对不上 |
| `clone()` 缓存张量 | 避免下游 in-place 操作污染缓存 |
| 不调 batch_size / precision | 你的脚本注释明确"immutable / 锁定"，与 Exp2 baseline 严格对齐 |
| 不设置 `set_float32_matmul_precision('high')` | 你的脚本注释明确"删除以与 Phase 6.5 fp32 baseline 一致" |

---

## 附录 A：所有相关命令一键参考

**查看训练状态**
```bash
ps -p $(cat /home/tcat/diffcsp_exp4/logs/step4_train.pid) -o pid,etime,rss,cmd
```

**查看最近的 epoch 进度**
```bash
grep "global step.*val_loss" /home/tcat/diffcsp_exp4/logs/step4_train_stderr.log | tail -10
```

**查看 GPU 使用**
```bash
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
```

**实时进度条**
```bash
python /home/tcat/diffcsp_exp4/code/step4_exp4/monitor.py
```

**查看最新 ckpt**
```bash
ls -la /home/tcat/diffcsp_exp4/checkpoints/
```

**完成后查看最佳 ckpt 路径**
```bash
cat /home/tcat/diffcsp_exp4/best_checkpoint_path.txt
```

---

**END OF REPORT**
