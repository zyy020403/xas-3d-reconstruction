# EXP5_PRIME_STEP1_HANDOFF.md
# SA-EXP5'-STEP1 任务 launch note(Exp5'-MA → SA-EXP5'-STEP1)

> **From**: Exp5'-MA(继 MA5 接班,Exp5 系列第 3 任 Main Agent)
> **To**: SA-EXP5'-STEP1(Exp5' 第一棒 Sub-Agent)
> **日期**: 2026-05-01
> **任务范围**: Exp5' Step 1 — 服务器目录 + 三件套物理 loss 实现 + shell_boundaries inject + smoke / forward_test 全 PASS(~ 1-2 天工程)
> **预期 hand-back**: 中期报告(forward_test PASS + smoke PASS + 关键日志)→ Exp5'-MA review → 启动 SA-EXP5'-train
> **本文档定位**: 给你的精确技术规格,所有 pseudocode 段都是 spec,实施时按本文 + proposal §2 双 source 对照

---

## §0 一屏掌握

### 0.1 你是谁,做什么

你是 **SA-EXP5'-STEP1**。任务 8 步:

| 步 | 任务 | 工程量 |
|---|---|---|
| 1.0 | 服务器 mkdir + cp Exp5 v2 code 树 + symlink data | 10 分钟 |
| 1.1 | dataset_v2.py 加 shell_boundaries inject + sample_name sanity check | 0.5 天 |
| 1.2 | datamodule_v2.py collate 加 5 字段 verify | 0.3 天 |
| 1.3 | diffusion_w_type_xas.py 加 3 loss 函数 + forward 调用 + 5 output 字段 + isfinite guard | 0.5 天 |
| 1.4 | yaml 加 3 cost_* 字段 | 5 分钟 |
| 1.5 | train.py from-scratch + best ckpt 用 PL ModelCheckpoint(monitor=val_composite_ckpt_score) | 0.3 天 |
| 1.6 | forward_test.py 加 Phase 6.7(三新 loss + yaml cost 加载) | 0.3 天 |
| 1.7 | smoke test 改 6 active loss 字段验证 + 跑 2 epoch × 10 batch PASS | 0.3 天 |
| 1.8 | 中期 hand-back 给 Exp5'-MA → review → 启动 SA-EXP5'-train | 0.2 天 |

**你不启动正式训练**(SA-EXP5'-train 是另一棒)。**你不动 holdout / 7 守卫包 / Exp4 backbone / Exp5 v2 ckpt / Phase 6.5 fp32**(详 §10 红线)。

### 0.2 必读 4 份(顺序)

1. **EXP5_PRIME_PROPOSAL.md** §2(三件套 loss 精确公式)+ §4(实施步骤 4.1-4.6)
2. **EXP5_FILE_GUIDE_v2.md** §6(目录新建命令)+ §8(PYTHONPATH)+ §9(verify 块)+ §11(风险点)
3. **EXP4_FINAL_REPORT_ERRATA_2.md** §1(`_density_loss` 是塌缩剂,Exp5' 沿用 cost=0.2 不动,你要在 smoke 阶段监控其与新 shell_dist_loss 的方向冲突信号)
4. **EXPERIMENT5_FINAL_REPORT_v2.md** §5(已知 bug / 工程债务 6 条全部沿用),§4.3(完全沿用清单)

### 0.3 启动后第一条回复格式

```
我已读完 4 份必读文档。复述任务要点 [6-8 条]。
最易踩坑 4 个: [...]。
计划: 第 1 步 ssh 跑 §1 mkdir 命令。
```

### 0.4 Exp5'-MA 已拍板的 5 条不再讨论(避免 SA 自由发挥)

1. **best ckpt selection 用 PL 原生 `ModelCheckpoint(monitor='val_composite_ckpt_score', mode='max', save_top_k=1, save_last=True)`**。在 LightningModule 的 `on_validation_epoch_end` 里 `self.log('val_composite_ckpt_score', score, prog_bar=True)` 。**禁止**自定义 `CompositeBestCkptCallback`(避免与 PL ModelCheckpoint 双轨)。这覆盖 handoff §3.6 的方案。
2. **三个新 loss 函数末尾必加 `isfinite` guard**(详 §4.5)。proposal §2.5 警告"epoch 0-10 ill-defined" 但没强制 guard,本 launch note 强制。
3. **PYTHONPATH 必须 `exp5_prime/code/step3:exp5_prime/code/step2:exp4/code` 三段,不放 `exp5/`**。smoke test 启动前 SA 必跑 import path verify(详 §8.2)。
4. **shell_boundaries `sample_name` schema sanity check 强制**:dataset `__init__` 加 hit_rate ≥ 95/100 assert,fail 立即 raise(详 §2.3)。
5. **MAX_EPOCHS = 500 在 train.py line 83 写死不在 yaml**(沿用 final report v2 §5.2)。

---

## §1 Step 1.0 — 服务器目录建立(10 分钟)

### 1.1 命令(完全照抄 EXP5_FILE_GUIDE_v2.md §6)

```bash
ssh tcat@scsmlnprd02.its.auckland.ac.nz

# 1. mkdir
mkdir -p /home/tcat/diffcsp_exp5_prime/{checkpoints,logs}

# 2. cp Exp5 v2 code 树
cp -r /home/tcat/diffcsp_exp5/code /home/tcat/diffcsp_exp5_prime/

# 3. data symlink
ln -s /home/tcat/diffcsp_exp4/data /home/tcat/diffcsp_exp5_prime/data

# 4. verify 目录结构
ls -la /home/tcat/diffcsp_exp5_prime/
ls -la /home/tcat/diffcsp_exp5_prime/code/
ls -la /home/tcat/diffcsp_exp5_prime/code/step3/
ls -la /home/tcat/diffcsp_exp5_prime/data/  # 应显示 → 链接

# 5. verify shell_boundaries 可访问 + md5
ls -la /home/tcat/diffcsp_exp5_prime/data/shell_boundaries.pkl
md5sum /home/tcat/diffcsp_exp5_prime/data/shell_boundaries.pkl
# 期望 md5 = cf2050e4899160f5698ad2481377e94c

# 6. checkpoints 空(from-scratch)
ls -la /home/tcat/diffcsp_exp5_prime/checkpoints/  # 空

# 7. 磁盘 sanity(目前 65G avail,STEP1 ~ 1G,够;但 hand-back 时报 du -sh)
df -h /
du -sh /home/tcat/diffcsp_exp5_prime/
```

### 1.2 PASS gate

- ✅ `code/{step2,step3,step4,step5}` 各有 .py 文件(见 EXP5_FILE_GUIDE_v2.md §2.1-2.4)
- ✅ `data` → 软链接到 `exp4/data/`
- ✅ shell_boundaries.pkl md5 = `cf2050e4899160f5698ad2481377e94c`
- ✅ `checkpoints/` 空目录(无 .ckpt)

---

## §2 Step 1.1 — dataset_v2.py inject shell_boundaries(0.5 天)

### 2.1 改动文件

`/home/tcat/diffcsp_exp5_prime/code/step3/xas_local_dataset_v2.py`

**先 cp 锚点**(改前):
```bash
cp /home/tcat/diffcsp_exp5_prime/code/step3/xas_local_dataset_v2.py \
   /home/tcat/diffcsp_exp5_prime/code/step3/xas_local_dataset_v2.py.bak_pre_exp5_prime
```

### 2.2 改动 pseudocode

```python
# 文件顶部 imports 区加:
import pickle
import logging

logger = logging.getLogger(__name__)

SHELL_BOUNDARIES_PATH = "/home/tcat/diffcsp_exp4/data/shell_boundaries.pkl"


class XasLocalDatasetV2:
    def __init__(self, ..., shell_boundaries_path=SHELL_BOUNDARIES_PATH):
        # ...(已有逻辑保留)...

        # ⭐ Exp5' inject: load shell_boundaries.pkl 一次到内存(387 MB)
        logger.info(f"[Exp5' inject] loading shell_boundaries.pkl from {shell_boundaries_path}")
        with open(shell_boundaries_path, 'rb') as f:
            self._shell_boundaries = pickle.load(f)
        logger.info(f"[Exp5' inject] loaded {len(self._shell_boundaries)} samples")

        # ⭐ sample_name schema sanity check(强制,fail 立即 raise)
        # 取前 100 个 indices 对应的 sample_name,在 self._shell_boundaries 里 lookup
        n_check = min(100, len(self.indices))
        sample_names_check = [self._get_sample_name(self.indices[i]) for i in range(n_check)]
        hits = sum(1 for sn in sample_names_check if sn in self._shell_boundaries)
        if hits < 95:
            # 列出前 5 个 miss + 前 5 个 pkl key 帮调试
            misses = [sn for sn in sample_names_check if sn not in self._shell_boundaries][:5]
            pkl_sample = list(self._shell_boundaries.keys())[:5]
            raise RuntimeError(
                f"[Exp5' inject] sample_name schema mismatch: {hits}/100 hits.\n"
                f"  dataset misses (first 5): {misses}\n"
                f"  pkl keys      (first 5): {pkl_sample}\n"
                f"  expected schema: 'mp-XXXXX__mp-XXXXX-EXAFS-{{element}}-K'"
            )
        logger.info(f"[Exp5' inject] sample_name schema sanity OK: {hits}/100 hits")

    def __getitem__(self, idx):
        # ...(已有逻辑到 data 对象构建好为止)...

        sample_name = self._get_sample_name(self.indices[idx])  # 沿用现有 helper
        sb_i = self._shell_boundaries.get(sample_name, None)
        if sb_i is None:
            return None  # 接 Phase 4.6 silent_drop 逻辑(已有)

        # 提取 5 字段(用 shell_of_atom + distances + shell_n_atoms,不碰 shell_starts/ends)
        shell_n        = sb_i['shell_n_atoms']      # (n_shells,) int32
        shell_of_atom  = sb_i['shell_of_atom']      # (n_neighbors,) int32, 全邻居
        distances      = sb_i['distances']          # (n_neighbors,) float32, 全邻居距中心

        # shell-1: shell_of_atom == 0
        shell1_mask = (shell_of_atom == 0)
        shell1_distances = distances[shell1_mask]
        true_shell1_d_mean = float(shell1_distances.mean()) if len(shell1_distances) > 0 else 0.0
        true_shell1_n      = int(shell_n[0]) if len(shell_n) > 0 else 0

        # shell-2: shell_of_atom == 1(可能不存在)
        if len(shell_n) > 1 and int(shell_n[1]) > 0:
            shell2_mask = (shell_of_atom == 1)
            shell2_distances = distances[shell2_mask]
            true_shell2_d_mean = float(shell2_distances.mean())
            true_shell2_n      = int(shell_n[1])
            has_shell2         = True
        else:
            true_shell2_d_mean = 0.0
            true_shell2_n      = 0
            has_shell2         = False

        # 塞进 Data 对象(per-sample 标量,collate 自动拼成 (B,) 张量)
        data.true_shell1_d_mean = torch.tensor(true_shell1_d_mean, dtype=torch.float32)
        data.true_shell2_d_mean = torch.tensor(true_shell2_d_mean, dtype=torch.float32)
        data.has_shell2         = torch.tensor(has_shell2, dtype=torch.bool)
        data.true_shell1_n      = torch.tensor(true_shell1_n, dtype=torch.long)
        data.true_shell2_n      = torch.tensor(true_shell2_n, dtype=torch.long)

        return data
```

### 2.3 关键注意事项

1. **`_get_sample_name(self, raw_idx)` helper 必须存在**或与现有 dataset 内部生成 sample_name 的方式一致。verify 输出已确认 schema 是 `'mp-555067__mp-555067-EXAFS-As-K'`(double underscore + 'EXAFS' tag + 元素 + edge)。如现有 dataset 内部用的不是这个 schema → 立即 raise(sanity check 会 catch)。**SA 不要擅自改 sample_name 生成逻辑去对齐 pkl**,先贴日志给 Exp5'-MA。
2. **`shell_starts / shell_ends` 是 `float32` 不是 int**(verify 输出确认,distance 边界不是 index 边界)。proposal §2.2 不用这两字段,**SA 也不要碰**。
3. **shell_n_atoms shape (5,)** 表明最多 5 壳。Exp5' 只用 index 0(shell-1)和 1(shell-2)。
4. **distances 长度 ~ 200+**(verify sample 是 221)是 cutoff(9.984 Å)内**全邻居**,不是 N=20 截断。这意味着 `true_shell1_d_mean / true_shell2_d_mean` 是基于全邻居的真值,而 _shell_distance_loss 里 pred 用的是 N=20 截断的 frac_coords。**已知 inconsistency,proposal §2.2 接受这个设计**(SA 不改公式),你只需在 hand-back 时报告 smoke 2 epoch 的 shell_dist_loss 数量级,异常飙升(> 100)报警。

### 2.4 PASS gate

- ✅ dataset `__init__` 不报 sanity 错(hit_rate ≥ 95/100)
- ✅ `__getitem__(0)` 输出 Data 对象包含 5 个新字段,dtype 严格符合上面 spec(float32 / float32 / bool / long / long)
- ✅ silent_drop 行为不变(Phase 4.6 沿用)
- ✅ `xas_local_dataset_v2.py.bak_pre_exp5_prime` 锚点存在

---

## §3 Step 1.2 — datamodule collate verify(0.3 天)

### 3.1 改动文件

`/home/tcat/diffcsp_exp5_prime/code/step3/xas_local_datamodule_v2.py`

**预期不需大改**:PyG `Batch.from_data_list` 自动处理 per-sample 标量字段,5 个新字段(全是标量 tensor)会自动 stack 成 `(B,)`。

但需要 **verify**:

### 3.2 Verify pseudocode

加一个 sanity print 在 `setup()` 末尾(临时,smoke 通过后可保留):

```python
def setup(self, stage=None):
    # ...(已有逻辑)...

    # ⭐ Exp5' verify: 第 1 个 batch 5 字段 shape
    if stage == 'fit' or stage is None:
        loader = self.train_dataloader()
        first_batch = next(iter(loader))
        for field in ['true_shell1_d_mean', 'true_shell2_d_mean', 'has_shell2',
                      'true_shell1_n', 'true_shell2_n']:
            assert hasattr(first_batch, field), f"missing field: {field}"
            v = getattr(first_batch, field)
            assert v.shape[0] == self.train_ds.batch_size or v.shape[0] <= self.train_ds.batch_size, \
                f"{field} shape {v.shape} != (B,)"
            logger.info(f"[Exp5' verify] {field}: shape={tuple(v.shape)}, dtype={v.dtype}")
```

### 3.3 PASS gate

- ✅ collate 第 1 个 batch 5 字段全部存在,shape `(B,)` 或 `(<=B,)`(silent_drop 后可能比 batch_size 小)
- ✅ dtype 一致:float32 / float32 / bool / long / long
- ✅ datamodule API 沿用 `.train_ds`(不是 `.train_dataset`,见 final report v2 §5.6)
- ✅ `.bak_pre_exp5_prime` 锚点存在

---

## §4 Step 1.3 — diffusion_w_type_xas.py 加 3 loss(0.5 天,核心)

### 4.1 改动文件

`/home/tcat/diffcsp_exp5_prime/code/step3/diffusion_w_type_xas.py`

**先 cp 锚点**:
```bash
cp /home/tcat/diffcsp_exp5_prime/code/step3/diffusion_w_type_xas.py \
   /home/tcat/diffcsp_exp5_prime/code/step3/diffusion_w_type_xas.py.bak_pre_exp5_prime
```

### 4.2 加 `_pairwise_min_distance_penalty`(主线 1)

完全按 proposal §2.1 pseudocode,加 isfinite guard:

```python
@staticmethod
def _pairwise_min_distance_penalty(pred_frac_coords, num_atoms, L=6.0, threshold=1.5):
    """
    proposal §2.1. Penalize pairs with cartesian d < 1.5 Å, min-image folded.
    ReLU(threshold - d)^2 mean per sample, then mean across batch.
    """
    total_loss = torch.tensor(0.0, device=pred_frac_coords.device)
    n_samples = 0
    start = 0
    for ni in num_atoms:
        ni = int(ni)
        if ni < 2:
            start += ni
            continue
        frac_i = pred_frac_coords[start:start+ni]  # (ni, 3)
        diff_frac = frac_i.unsqueeze(0) - frac_i.unsqueeze(1)   # (ni, ni, 3)
        diff_frac = diff_frac - diff_frac.round()               # min-image to [-0.5, 0.5]
        diff_cart = diff_frac * L                                # (ni, ni, 3)
        d = diff_cart.norm(dim=-1)                               # (ni, ni)
        mask = torch.triu(torch.ones_like(d), diagonal=1).bool()
        d_pairs = d[mask]                                        # (ni*(ni-1)/2,)
        violation = torch.relu(threshold - d_pairs)
        total_loss = total_loss + (violation ** 2).mean()
        n_samples += 1
        start += ni

    loss = total_loss / max(n_samples, 1)
    # ⭐ isfinite guard
    if not torch.isfinite(loss):
        loss = torch.tensor(0.0, device=pred_frac_coords.device, requires_grad=True)
    return loss
```

### 4.3 加 `_shell_distance_loss`(主线 2)

完全按 proposal §2.2,加 isfinite guard,signature 接受 batch 5 字段:

```python
def _shell_distance_loss(self, pred_frac_coords, num_atoms,
                         true_shell1_d_mean, true_shell2_d_mean, has_shell2,
                         L=6.0, threshold_gap=0.1563):
    """
    proposal §2.2. Gap-based shell split on pred → mean radial distance →
    MSE vs Step 2.5 ground truth.
    """
    total_loss = torch.tensor(0.0, device=pred_frac_coords.device)
    n_active = 0
    start = 0
    for i, ni in enumerate(num_atoms):
        ni = int(ni)
        if ni < 2:
            start += ni
            continue
        coords_i = pred_frac_coords[start:start+ni] * L     # (ni, 3) cartesian
        radial = coords_i.norm(dim=1)                        # (ni,) center-to-atom dist
        sorted_d, _ = radial.sort()

        gaps = sorted_d[1:] - sorted_d[:-1]
        boundaries = (gaps > threshold_gap).nonzero(as_tuple=True)[0]
        if len(boundaries) >= 1:
            shell1_end = int(boundaries[0].item()) + 1
            pred_s1_d_mean = sorted_d[:shell1_end].mean()
            total_loss = total_loss + (pred_s1_d_mean - true_shell1_d_mean[i]) ** 2

            if len(boundaries) >= 2 and bool(has_shell2[i]):
                shell2_end = int(boundaries[1].item()) + 1
                pred_s2_d_mean = sorted_d[shell1_end:shell2_end].mean()
                total_loss = total_loss + (pred_s2_d_mean - true_shell2_d_mean[i]) ** 2

            n_active += 1
        start += ni

    loss = total_loss / max(n_active, 1)
    if not torch.isfinite(loss):
        loss = torch.tensor(0.0, device=pred_frac_coords.device, requires_grad=True)
    return loss
```

### 4.4 加 `_shell_count_loss`(辅助)

完全按 proposal §2.3,同样加 isfinite guard:

```python
def _shell_count_loss(self, pred_frac_coords, num_atoms,
                      true_shell1_n, true_shell2_n, has_shell2,
                      L=6.0, threshold_gap=0.1563):
    """proposal §2.3. Float MSE on gap-derived shell counts."""
    total_loss = torch.tensor(0.0, device=pred_frac_coords.device)
    n_active = 0
    start = 0
    for i, ni in enumerate(num_atoms):
        ni = int(ni)
        if ni < 2:
            start += ni
            continue
        coords_i = pred_frac_coords[start:start+ni] * L
        radial = coords_i.norm(dim=1)
        sorted_d, _ = radial.sort()

        gaps = sorted_d[1:] - sorted_d[:-1]
        boundaries = (gaps > threshold_gap).nonzero(as_tuple=True)[0]
        if len(boundaries) >= 1:
            pred_s1_n = float(int(boundaries[0].item()) + 1)
            total_loss = total_loss + (pred_s1_n - float(true_shell1_n[i])) ** 2

            if len(boundaries) >= 2 and bool(has_shell2[i]):
                pred_s2_n = float(int(boundaries[1].item()) - int(boundaries[0].item()))
                total_loss = total_loss + (pred_s2_n - float(true_shell2_n[i])) ** 2

            n_active += 1
        start += ni

    loss = total_loss / max(n_active, 1)
    if not torch.isfinite(loss):
        loss = torch.tensor(0.0, device=pred_frac_coords.device, requires_grad=True)
    return loss
```

### 4.5 forward() 内调用 + total_loss

```python
def forward(self, batch):
    # ...(已有 coord/type/density 三 loss 计算)...

    # ⭐ Exp5' 三件套
    loss_pairwise_min = self._pairwise_min_distance_penalty(
        pred_frac_coords, batch.num_atoms, L=self.L, threshold=1.5
    )
    loss_shell_dist = self._shell_distance_loss(
        pred_frac_coords, batch.num_atoms,
        batch.true_shell1_d_mean, batch.true_shell2_d_mean, batch.has_shell2,
        L=self.L, threshold_gap=0.1563
    )
    loss_shell_count = self._shell_count_loss(
        pred_frac_coords, batch.num_atoms,
        batch.true_shell1_n, batch.true_shell2_n, batch.has_shell2,
        L=self.L, threshold_gap=0.1563
    )

    # ⭐ total_loss(7 项,proposal §2.4)
    total_loss = (self.cost_lattice * loss_lattice          # 0.0
                + self.cost_coord   * loss_coord            # 1.0
                + self.cost_type    * loss_type             # 1.0
                + self.cost_density * loss_density          # 0.2 沿用
                + self.cost_pairwise_min * loss_pairwise_min  # 1.0 起步,新
                + self.cost_shell_dist   * loss_shell_dist    # 0.5 起步,新
                + self.cost_shell_count  * loss_shell_count)  # 0.2 起步,新

    return {
        'loss': total_loss,
        'loss_lattice': loss_lattice,
        'loss_coord':   loss_coord,
        'loss_type':    loss_type,
        'loss_density': loss_density,
        # ⭐ Exp5' 新加 5 字段
        'loss_pairwise_min': loss_pairwise_min,
        'loss_shell_dist':   loss_shell_dist,
        'loss_shell_count':  loss_shell_count,
        # 简化 monitoring metric(epoch-level 在 validation_step / on_validation_epoch_end 聚合)
        # 这两个由 validation_step 单独算,不在 forward 算:
        # 'val_min_d_mean':       (per-sample min pairwise d, mean)
        # 'val_gate_pass_rate':   (fraction of samples with min_d ≥ 1.5 Å)
    }
```

### 4.6 validation_step / on_validation_epoch_end 加 monitoring

LightningModule(若是 self;否则在 step3 入口的 LightningModule wrapper)加:

```python
def validation_step(self, batch, batch_idx):
    out = self.forward(batch)
    # ...(已有 log)...

    # ⭐ Exp5' simplified physical monitor(不跑 Hungarian,只用 pred 内部)
    with torch.no_grad():
        # per-sample min pairwise distance
        per_sample_min_d = []
        start = 0
        L = self.L
        for ni in batch.num_atoms:
            ni = int(ni)
            if ni < 2:
                start += ni
                continue
            frac_i = batch.pred_frac_coords[start:start+ni]  # 用 forward 输出的 pred
            diff = frac_i.unsqueeze(0) - frac_i.unsqueeze(1)
            diff = diff - diff.round()
            d = (diff * L).norm(dim=-1)
            mask = torch.triu(torch.ones_like(d), diagonal=1).bool()
            min_d = d[mask].min().item()
            per_sample_min_d.append(min_d)
            start += ni

    self.validation_step_outputs.append({
        'val_loss':              out['loss'].detach(),
        'val_loss_pairwise_min': out['loss_pairwise_min'].detach(),
        'val_loss_shell_dist':   out['loss_shell_dist'].detach(),
        'val_loss_shell_count':  out['loss_shell_count'].detach(),
        'min_d_per_sample':      per_sample_min_d,
    })
    return out['loss']

def on_validation_epoch_end(self):
    outputs = self.validation_step_outputs
    val_loss = torch.stack([o['val_loss'] for o in outputs]).mean()
    val_loss_pairwise_min = torch.stack([o['val_loss_pairwise_min'] for o in outputs]).mean()
    val_loss_shell_dist = torch.stack([o['val_loss_shell_dist'] for o in outputs]).mean()
    val_loss_shell_count = torch.stack([o['val_loss_shell_count'] for o in outputs]).mean()

    all_min_d = [m for o in outputs for m in o['min_d_per_sample']]
    val_min_d_mean = float(np.mean(all_min_d)) if len(all_min_d) > 0 else 0.0
    val_gate_pass_rate = float(np.mean([1.0 if m >= 1.5 else 0.0 for m in all_min_d])) \
                         if len(all_min_d) > 0 else 0.0

    # ⭐ 复合 ckpt score(Exp5'-MA 拍板,proposal §3.3)
    val_composite_ckpt_score = (
        0.2 * (1.0 - min(float(val_loss), 1.0))
      + 0.5 * val_gate_pass_rate
      + 0.3 * (1.0 - min(float(val_loss_pairwise_min), 1.0))
    )

    # log 全部(prog_bar=True 让 ModelCheckpoint 看到 + 命令行可见)
    self.log('val_loss',                     val_loss,                  prog_bar=True)
    self.log('val_loss_pairwise_min',        val_loss_pairwise_min)
    self.log('val_loss_shell_dist',          val_loss_shell_dist)
    self.log('val_loss_shell_count',         val_loss_shell_count)
    self.log('val_min_d_mean',               val_min_d_mean,            prog_bar=True)
    self.log('val_gate_pass_rate',           val_gate_pass_rate,        prog_bar=True)
    self.log('val_composite_ckpt_score',     val_composite_ckpt_score,  prog_bar=True)

    self.validation_step_outputs.clear()
```

**注意**:`pred_frac_coords` 怎么从 forward 输出传到 validation_step,要看现有 step3 的 forward 接口。如 forward 已 return 了 pred_frac_coords,直接用;否则在 forward output dict 里加 `'pred_frac_coords': pred_frac_coords`。SA 实施时按现有接口适配。

### 4.7 PASS gate

- ✅ 三 loss 函数存在 + 都有 isfinite guard
- ✅ forward() return dict 含 8 个 loss 字段(原 4 + 新 3 + total)
- ✅ validation_step 累积 per-sample min_d
- ✅ on_validation_epoch_end log 7 个 metric(含 `val_composite_ckpt_score`)
- ✅ `.bak_pre_exp5_prime` 锚点存在

---

## §5 Step 1.4 — yaml 加 3 cost 字段(5 分钟)

### 5.1 改动文件

`/home/tcat/diffcsp_exp5_prime/code/step3/conf_xas/model/diffusion_xas.yaml`

**末尾 append**:

```yaml
# ⭐ Exp5' 三件套物理 loss(proposal §2.1-2.3)
cost_pairwise_min: 1.0   # Exp5' 起步,Exp5'-MA 监控 epoch 0-50 调度
cost_shell_dist:   0.5   # Exp5' 起步
cost_shell_count:  0.2   # Exp5' 起步
```

确认 `cost_density: 0.2` 已存在(Exp5 v2 SA1' 已设),不动。

### 5.2 PASS gate

- ✅ yaml load 后 model 实例 `self.cost_pairwise_min == 1.0`,`self.cost_shell_dist == 0.5`,`self.cost_shell_count == 0.2`
- ✅ `self.cost_density == 0.2` 不变(verify 沿用)

---

## §6 Step 1.5 — train.py from-scratch + ckpt callback(0.3 天)

### 6.1 改动文件

`/home/tcat/diffcsp_exp5_prime/code/step4/step4_2_train.py`

**先 cp 锚点**:
```bash
cp /home/tcat/diffcsp_exp5_prime/code/step4/step4_2_train.py \
   /home/tcat/diffcsp_exp5_prime/code/step4/step4_2_train.py.bak_pre_exp5_prime
```

### 6.2 改动概要

(a) **删除 SA2'' 的 `last_ckpt = ...epoch=484-...` 硬编码**,改为 `ckpt_path = None`(from-scratch)

(b) **MAX_EPOCHS = 500**(line 83 沿用,LR scheduler T_max=500 自动跟随)

(c) **替换 ModelCheckpoint 配置**:

```python
# ❌ 删除 v2 时代的 monitor='val_loss'
# checkpoint_callback = ModelCheckpoint(monitor='val_loss', ...)

# ✅ Exp5' 用复合 score
checkpoint_callback = ModelCheckpoint(
    dirpath='/home/tcat/diffcsp_exp5_prime/checkpoints',
    filename='composite_best_epoch{epoch:03d}_score{val_composite_ckpt_score:.4f}',
    monitor='val_composite_ckpt_score',
    mode='max',           # ⭐ 大者优,与 v2 时代 monitor='val_loss' mode='min' 反向
    save_top_k=1,
    save_last=True,       # 保 last.ckpt
    auto_insert_metric_name=False,  # filename 自含 metric,避免重复
)
```

(d) **trainer.callbacks 包含**:
- ModelCheckpoint(上面)
- EarlyStopping(monitor='val_composite_ckpt_score', mode='max', patience=30)
- LearningRateMonitor

⚠️ early stop monitor 也切到 composite score,不是 val_loss(否则 val_loss 早收敛 ckpt 会被 patience 抢走)。

(e) **trainer.fit(model, ckpt_path=None)**(明确 from-scratch)

### 6.3 PASS gate

- ✅ `last_ckpt` 硬编码已删除
- ✅ ModelCheckpoint monitor='val_composite_ckpt_score' mode='max'
- ✅ EarlyStopping monitor='val_composite_ckpt_score' mode='max'
- ✅ MAX_EPOCHS = 500
- ✅ trainer.fit() 不传 ckpt_path
- ✅ `.bak_pre_exp5_prime` 锚点存在

---

## §7 Step 1.6 — forward_test Phase 6.7(0.3 天)

### 7.1 改动文件

`/home/tcat/diffcsp_exp5_prime/code/step3/forward_test.py`

**先 cp 锚点**(改前)。

### 7.2 加 Phase 6.7

```python
def phase_67(batch_cpu):
    log("Phase 6.7 — Exp5' three new physical loss functions")

    model = _instantiate_model()

    # 6.7.a: 三函数存在
    assert hasattr(model, '_pairwise_min_distance_penalty'), "missing _pairwise_min_distance_penalty"
    assert hasattr(model, '_shell_distance_loss'),           "missing _shell_distance_loss"
    assert hasattr(model, '_shell_count_loss'),              "missing _shell_count_loss"
    log("[6.7.a PASS] three loss methods exist")

    # 6.7.b: yaml cost 字段加载到 model
    assert abs(model.cost_pairwise_min - 1.0) < 1e-6, f"cost_pairwise_min={model.cost_pairwise_min}"
    assert abs(model.cost_shell_dist   - 0.5) < 1e-6, f"cost_shell_dist={model.cost_shell_dist}"
    assert abs(model.cost_shell_count  - 0.2) < 1e-6, f"cost_shell_count={model.cost_shell_count}"
    assert abs(model.cost_density      - 0.2) < 1e-6, f"cost_density={model.cost_density}"
    log("[6.7.b PASS] yaml cost_* loaded correctly")

    # 6.7.c: pairwise_min on dummy collapsed batch (min_d=0.5 Å) → loss > 0
    L = 6.0
    # 20 atoms all at origin → frac=0, min_d=0 → strong violation
    coords_collapse = torch.zeros(20, 3)
    num_atoms_collapse = torch.tensor([20])
    loss_collapse = model._pairwise_min_distance_penalty(coords_collapse, num_atoms_collapse, L=L, threshold=1.5)
    assert loss_collapse.item() > 1.0, f"expected high loss for collapse, got {loss_collapse.item()}"
    log(f"[6.7.c PASS] collapse batch loss = {loss_collapse.item():.4f}")

    # 6.7.d: pairwise_min on dummy spread batch (random in [-0.45, 0.45] frac → min_d ~ 1-2 Å)
    # → 检查 isfinite + finite-positive
    torch.manual_seed(42)
    coords_spread = (torch.rand(20, 3) - 0.5) * 0.9  # random in [-0.45, 0.45] frac
    loss_spread = model._pairwise_min_distance_penalty(coords_spread, num_atoms_collapse, L=L, threshold=1.5)
    assert torch.isfinite(loss_spread), f"loss not finite: {loss_spread}"
    assert loss_spread.item() < loss_collapse.item(), \
        f"spread should < collapse: spread={loss_spread.item()}, collapse={loss_collapse.item()}"
    log(f"[6.7.d PASS] spread batch loss = {loss_spread.item():.4f}, < collapse OK")

    # 6.7.e: shell_dist_loss on dummy → finite (即使 gap 切壳 garbage)
    true_s1 = torch.tensor([2.0])
    true_s2 = torch.tensor([3.5])
    has_s2  = torch.tensor([True])
    loss_sd = model._shell_distance_loss(coords_spread, num_atoms_collapse,
                                         true_s1, true_s2, has_s2, L=L, threshold_gap=0.1563)
    assert torch.isfinite(loss_sd), f"shell_dist_loss not finite: {loss_sd}"
    log(f"[6.7.e PASS] shell_dist_loss on dummy = {loss_sd.item():.4f}")

    # 6.7.f: shell_count_loss on dummy → finite
    true_n1 = torch.tensor([6])
    true_n2 = torch.tensor([12])
    loss_sc = model._shell_count_loss(coords_spread, num_atoms_collapse,
                                      true_n1, true_n2, has_s2, L=L, threshold_gap=0.1563)
    assert torch.isfinite(loss_sc), f"shell_count_loss not finite: {loss_sc}"
    log(f"[6.7.f PASS] shell_count_loss on dummy = {loss_sc.item():.4f}")

    # 6.7.g: isfinite guard 主动测(NaN injection)
    coords_nan = torch.full((20, 3), float('nan'))
    loss_nan = model._pairwise_min_distance_penalty(coords_nan, num_atoms_collapse, L=L, threshold=1.5)
    assert torch.isfinite(loss_nan), f"isfinite guard failed: loss = {loss_nan}"
    assert loss_nan.item() == 0.0, f"isfinite guard should return 0.0, got {loss_nan.item()}"
    log("[6.7.g PASS] isfinite guard works on NaN input")

    log("[Phase 6.7 PASS]")
```

### 7.3 PASS gate

- ✅ Phase 6.7 a-g 全 PASS
- ✅ 原有 Phase 6.1-6.6 不变(注意 6.5 仍然 SKIPPED-by-design)
- ✅ 总计 5/5 PASS + 1 SKIPPED + Phase 6.7 PASS

---

## §8 Step 1.7 — smoke test 改 + 跑(0.3 天)

### 8.1 改动文件

`/home/tcat/diffcsp_exp5_prime/code/step4/step4_1_smoke_test.py`

加 6 active loss 字段验证(原 v2 是 4 active loss):

```python
# v2 时代验证 4 字段
# v5' 验证 6 字段(原 4 + 新 3 - lattice = 6)
EXPECTED_LOSS_FIELDS = [
    'loss_coord', 'loss_type', 'loss_density',
    'loss_pairwise_min', 'loss_shell_dist', 'loss_shell_count',
]
for field in EXPECTED_LOSS_FIELDS:
    assert field in out, f"missing field {field}"
    assert torch.isfinite(out[field]), f"{field} not finite: {out[field]}"
# loss_lattice 应为 0(cost_lattice=0)— OK if exists but 0
```

### 8.2 PYTHONPATH + smoke 启动 verify(强制)

```bash
# ⭐ 启动 smoke 前必跑(SA-EXP5'-STEP1 在 hand-back log 里报告这段输出)
export PYTHONPATH=/home/tcat/diffcsp_exp5_prime/code/step3:/home/tcat/diffcsp_exp5_prime/code/step2:/home/tcat/diffcsp_exp4/code

# Verify import 来自 exp5_prime/(不是 exp5/)
/home/tcat/conda_envs/mlff/bin/python -c "
import xas_local_dataset_v2
import xas_local_datamodule_v2
import diffusion_w_type_xas
print(f'xas_local_dataset_v2:    {xas_local_dataset_v2.__file__}')
print(f'xas_local_datamodule_v2: {xas_local_datamodule_v2.__file__}')
print(f'diffusion_w_type_xas:    {diffusion_w_type_xas.__file__}')
# 必须全部以 /home/tcat/diffcsp_exp5_prime/ 开头
"
```

如任一返回 `/home/tcat/diffcsp_exp5/` 路径 → 立即 stop,贴日志给 Exp5'-MA。

### 8.3 smoke 跑

```bash
cd /home/tcat/diffcsp_exp5_prime/code/step4
/home/tcat/conda_envs/mlff/bin/python step4_1_smoke_test.py 2>&1 | tee /home/tcat/diffcsp_exp5_prime/logs/smoke_step1.log
```

### 8.4 PASS gate

- ✅ import path verify 全部以 `/home/tcat/diffcsp_exp5_prime/` 开头
- ✅ 2 epoch × 10 batch 完整跑完
- ✅ 6 active loss 字段全 finite
- ✅ best ckpt callback 触发(ckpt 文件落盘 + 命名含 score)
- ✅ on_validation_epoch_end log 7 个 metric

---

## §9 Step 1.8 — 中期 hand-back(给 Exp5'-MA review)

### 9.1 hand-back 必报字段(写进 OUTPUT.md)

1. **§1.0 PASS evidence**: ls -la 服务器目录;md5sum shell_boundaries.pkl;du -sh exp5_prime/
2. **§1.1-1.7 各 PASS gate 通过日志**(逐项贴 evidence)
3. **forward_test 完整 log**(5 + 1 SKIPPED + Phase 6.7 a-g PASS)
4. **smoke test 2 epoch 全部 6 active loss 表**:

| field | epoch 0 mean | epoch 1 mean | finite? |
|---|---|---|---|
| loss_coord | ... | ... | ✅ |
| loss_type | ... | ... | ✅ |
| loss_density | ... | ... | ✅ |
| loss_pairwise_min | ... | ... | ✅ |
| loss_shell_dist | ... | ... | ✅ |
| loss_shell_count | ... | ... | ✅ |

5. **on_validation_epoch_end 7 个 metric 数值**(单 epoch dummy validation):val_loss / val_loss_pairwise_min / val_loss_shell_dist / val_loss_shell_count / val_min_d_mean / val_gate_pass_rate / val_composite_ckpt_score
6. **best ckpt 触发 evidence**: ls -la exp5_prime/checkpoints/(应见 1 个 composite_best_*.ckpt + 1 个 last.ckpt)
7. **import path verify 输出**(§8.2 命令的 stdout)
8. **磁盘占用**: du -sh /home/tcat/diffcsp_exp5_prime/(预期 ~ 200-500 MB,smoke ckpt + log)
9. **OPEN 问题或异常**: 任何不确定的事,贴日志,**不擅自 fix**

### 9.2 Exp5'-MA review checklist

| 项 | 通过 | 说明 |
|---|---|---|
| §1-§8 全部 PASS gate | | 逐项 |
| smoke 6 loss 全 finite + 数量级合理(loss_pairwise_min 远 > loss_shell_dist 是 OK,初期 pred 重合)| | |
| `loss_density` 与 `loss_shell_dist` 数量级对比合理(防 cost 0.2 仍 dominate) | | watch-only |
| best ckpt 文件落盘 + filename 含 score | | |
| import path 全 exp5_prime | | |
| 磁盘 / GPU 状态 | | |

如全过 → Exp5'-MA 写 SA-EXP5'-train launch note,启动正式 ~ 32-40h 训练。
如任一不过 → SA 修复后再 hand-back,不擅自启动 train。

---

## §10 红线汇总(SA-EXP5'-STEP1 全程不动)

| 红线 | 说明 |
|---|---|
| ❌ 不动 holdout | 永久封存 |
| ❌ 不升级 7 守卫包 | 沿用 v2 锁定版本 |
| ❌ 不动 Exp5 v2 ckpt(`/home/tcat/diffcsp_exp5/checkpoints/`)| 永久档案 |
| ❌ 不动 Exp4 backbone(`/home/tcat/diffcsp_exp4/code/diffcsp/`)| Exp5' PYTHONPATH 末尾 import |
| ❌ 不修 Phase 6.5 hardcoded fp32 | 永久 SKIPPED-by-design |
| ❌ 不动 step5_3_composite_score.py | 沿用,Exp5' 主指标 |
| ❌ 不启动正式训练 | SA-EXP5'-train 是另一棒 |
| ❌ 不擅自调三件套 λ | proposal §2.1-2.3 起步值锁定: 1.0 / 0.5 / 0.2 |
| ❌ 不擅自调 cost_density(0.2 沿用)| Exp5'-MA 监控期决议,不是 SA |
| ❌ 不动 sample_name 生成逻辑去对齐 pkl | sanity check fail 立即 raise + 贴日志 |
| ❌ 不碰 shell_starts / shell_ends 字段(float32 边界,Exp5' 不用)| |
| ❌ 任何不确定的事 → 贴日志,不靠记忆 | |

---

## §11 Watch-only 项(Exp5'-MA 监控,SA 报告即可不行动)

1. **`loss_density` vs `loss_shell_dist` 方向冲突**(errata 2 §1):cost_density=0.2 减弱后,density 仍把 pred 拉向原点,但 shell_dist 拉向 ~2-3 Å。SA hand-back 报这两 loss 数值 + 对比 Exp5'-MA 看趋势。
2. **shell_dist_loss 真值/预测 inconsistency**:真值用全邻居(~ 200+ atoms 在 cutoff 9.984 内)算 mean,预测用 N=20 截断算 mean。已知设计,SA 不改公式,但 hand-back 报 shell_dist_loss 数量级,异常飙升(> 100 持续)报警。
3. **磁盘空间趋势**:65G avail,STEP1 ~ 1G,但 SA-EXP5'-train(~ 30-40 个 ckpt × 44MB + 大 log)需 ~ 5-10G。SA hand-back 时报 du -sh,Exp5'-MA 在写 SA-EXP5'-train launch note 前评估是否清理 v2 历史 log。

---

## §12 OPEN QUESTIONS(SA 不答,贴给 Exp5'-MA)

### Q1 — `pred_frac_coords` 在 forward 输出 dict 里是否已存在?

需要从 forward 输出传到 validation_step 算 min_d。如已存在,直接用;若不存在,在 forward output dict 加 `'pred_frac_coords': pred_frac_coords`。SA 实施时按 v2 现有接口适配,在 hand-back 报告改了哪种。

### Q2 — `_get_sample_name(self, raw_idx)` helper 是否已存在?

dataset_v2 内部如何生成 sample_name 决定 sanity check 是否能通过。SA 先看 v2 现有逻辑(很可能已有,SA1' 时代加 `center_element_Z` 字段时摸过),报告现状 + sanity 结果。

### Q3 — datamodule `pin_memory` / `num_workers=0` 与 387 MB pkl 的兼容性

`num_workers=0`(沿用 v2 pymatgen worker safety)意味着 pkl 在主进程一次 load,~ 387 MB 内存常驻,不复制到 worker。这是预期。SA hand-back 报告 process RSS 增量(`ps -o rss` smoke 前后对比)。

### Q4 — LightningModule 是否有 `self.validation_step_outputs` 属性

PL 2.5 已弃用 `validation_epoch_end(outputs)` 旧 hook,改为 `on_validation_epoch_end()` + 在 LightningModule 维护 `self.validation_step_outputs` list。SA 检查 v2 现有用法,沿用同一 pattern;若 v2 还在用 `validation_epoch_end(outputs)` 旧 API,SA 报告(可能要 surgery)。

---

## §13 你不做的事(SA-EXP5'-train 才做)

- 启动 ~ 32-40h 正式训练
- 调三件套 λ(epoch-level 监控决议是 Exp5'-MA 工作)
- sample 生成
- 7 项复合分评估

---

## §14 OUTPUT.md 模板

写 `EXP5_STEP1_PRIME_OUTPUT.md` 落服务器 `/home/tcat/diffcsp_exp5_prime/` 根目录:

```
# EXP5_STEP1_PRIME_OUTPUT.md
# SA-EXP5'-STEP1 中期 hand-back

## §0 状态
- Step 1.0-1.7 全部完成 / 部分完成 / 卡在 X
- forward_test: X/X PASS + Phase 6.7 PASS
- smoke: 2 epoch 完成

## §1-§7 各 PASS gate evidence
[逐 step 贴 log + 命令输出]

## §8 §9.1 必报字段(详 launch note §9.1 1-9 项)

## §9 OPEN 问题(贴给 Exp5'-MA,不擅自 fix)
```

---

## §15 工作哲学红线(沿用 v1 MA + MA5)

1. **任何技术判断先 conversation_search + 列证据**(SA 没有这个能力,但 SA 不擅自做技术判断,贴问题给 Exp5'-MA)
2. **任何不确定的事 → 贴日志,不靠记忆**
3. **小补丁也要贴 diff**(任何文件改动,贴 git diff 风格的前后对比给 Exp5'-MA review)
4. **70% 上下文闸门是硬线**(SA 接近时主动 hand-back,Exp5'-MA 决议)
5. **不擅自启动正式训练 / 不擅自调 λ / 不擅自动 cost_density**

---

*Exp5'-MA 撰写,2026-05-01,基于 EXP5_PRIME_PROPOSAL.md / EXP5_FILE_GUIDE_v2.md §9 verify 全过 / EXP4_FINAL_REPORT_ERRATA_2.md / EXPERIMENT5_FINAL_REPORT_v2.md。SA-EXP5'-STEP1 接此 launch note 启动 Exp5' 第一棒。*
