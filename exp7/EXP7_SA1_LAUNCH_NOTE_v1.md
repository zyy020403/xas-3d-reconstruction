# EXP7_SA1_LAUNCH_NOTE_v1.md
# Exp7 SA1 — Launch Note from MA1

> **撰写者**: Exp7-MA1
> **日期**: 2026-05-12
> **任务**: Exp7 GAN Phase 1 — Day 0 verify + setup + smoke test
> **GPU**: cuda:0 (RTX 4090, 24GB) — Exp6 占用 cuda:1,两者互不干扰
> **SA1 工作范围**: Phase 1 全部 (step1.0 → step1.4) + Phase 2 sanity (10 epoch)
> **本文档结构**: §1 你需要做什么 → §2 不变量红线 → §3 逐步指令 → §4 验收标准 → §5 raise 清单

---

## §1 你的任务:一句话

**在 GPU cuda:0 上,为 Exp7 WGAN-GP 实验完成所有 Day 0 verify、repo setup、和 smoke test,产出可以开始全量训练的环境。**

完成后 MA1 会根据你的 hand-back 决定是否进 Phase 3(全量训练)。

---

## §2 不变量红线(任何情况下不许违反,不许擅自决定)

### 2.1 绝对禁止

| 禁止项 | 来源 |
|---|---|
| 引入 `_shell_distance_loss` / `_shell_count_loss` / `_density_loss` 任一 | Exp5 系列三阶段证伪 + errata 2/5/6 |
| 引入 TypeClassifier head | Exp3 双重 + Exp5 三重证伪 |
| 改动 `CURRICULUM_FRACTIONS` / `CURRICULUM_EPOCH_BOUNDARIES` 数值 | 用户决议锁定 |
| 改动 `lambda_gp = 10.0` | Gulrajani 2017 标值,不许动 |
| 改动 `n_critic = 5` | 同上,只有风险 1 触发才允许 MA1 决议 |
| touch holdout 数据(除 Phase 4 build cache 外) | 全系列永久封存红线 |
| Phase 0-2(epoch < 150)选 best ckpt 或让 EarlyStop 计 patience | Curriculum 设计原则 |
| Option B 改动 `xas_local_dataset_v2.py` | SOP 12 "dataset 是用户决议红线" |
| 自行"擅自 patch" dataset contract 异常 | 出乎预期必须 RAISE |
| 跳过用户物理 sanity | must-do §13.3 |

### 2.2 必须 RAISE 给 MA1(不许擅自决定)

- 任何 md5 verify 失败
- Dataset contract V1-V5 任一项出乎预期(见 §3.3)
- Exp6 文件与预期不一致(CPS 公式 / vocab / MIN_PDIST 数值)
- Smoke test 中 G 或 D forward 报错
- L4 评估 active 不通过(`n_pred_shells > 0` 比例 < 80%)
- 双套评估差 > 30% relative

---

## §3 逐步指令

**执行顺序严格按序,任一步 FAIL 必须 raise,不许跳步。**

---

### Step 1.0 — 数据完整性 verify + symlink setup

**目标产出**: `experiment7/data/data_integrity.json`

**没有此文件,不许进 Step 1.1。**

#### 1.0.1 创建实验目录

```bash
mkdir -p /home/tcat/experiment7/{data,shared,step1,step2,step3,step4,_vendor}
cd /home/tcat/experiment7
```

#### 1.0.2 五项强制 verify(任一 FAIL → RAISE MA1)

```bash
# (1) shell_boundaries.pkl md5
echo "=== V1: shell_boundaries.pkl ==="
md5sum /home/tcat/diffcsp_exp4/data/shell_boundaries.pkl
# Expected: cf2050e4899160f5698ad2481377e94c

# (2) Exp5' best ckpt md5
echo "=== V2: Exp5' ckpt ==="
md5sum /home/tcat/diffcsp_exp5_prime/checkpoints/composite_epoch169_score0.5881.ckpt
# Expected: 127afa44a850d8f7e4fcdae17e2761a1

# (3) cache_metadata L=20
echo "=== V3: cache_metadata ==="
python3 -c "
import json
m = json.load(open('/home/tcat/diffcsp_exp5_prime/data/cache_metadata.json'))
assert m.get('L_VIRTUAL') == 20.0, f'FAIL: L_VIRTUAL={m.get(\"L_VIRTUAL\")}, expected 20.0'
print('OK: L_VIRTUAL =', m['L_VIRTUAL'])
"

# (4) Exp6 step1 五文件存在
echo "=== V4: Exp6 step1 产出 ==="
for f in min_pdist_calibration.json shell_integrity_report.json exp6_element_vocab.json baseline_cps.json composite_score.py; do
    test -f /home/tcat/experiment6_v7/shared/$f \
        && echo "OK: $f" \
        || echo "MISSING: $f  ← RAISE MA1"
done

# (5) Exp6 MIN_PDIST 数值
echo "=== V5: MIN_PDIST ==="
python3 -c "
import json
c = json.load(open('/home/tcat/experiment6_v7/shared/min_pdist_calibration.json'))
val = c['min_pdist']
frozen = c['frozen']
assert frozen == True, f'FAIL: frozen={frozen}'
assert abs(val - 1.5075718402862548) < 1e-10, f'FAIL: MIN_PDIST={val}'
print(f'OK: MIN_PDIST={val}, frozen={frozen}')
"
```

#### 1.0.3 symlink 所有数据文件

```bash
cd /home/tcat/experiment7/data

# Structure cache (L=20)
ln -s /home/tcat/diffcsp_exp5_prime/data/train_structure_cache.pt .
ln -s /home/tcat/diffcsp_exp5_prime/data/val_structure_cache.pt .
ln -s /home/tcat/diffcsp_exp5_prime/data/test_structure_cache.pt .
ln -s /home/tcat/diffcsp_exp5_prime/data/cache_metadata.json .

# shell boundaries (387 MB, symlink 不 cp)
ln -s /home/tcat/diffcsp_exp4/data/shell_boundaries.pkl .

# Spectrum + FEFF
ln -s /home/tcat/diffcsp_exp5_prime/data/spectra_train.pkl .
ln -s /home/tcat/diffcsp_exp5_prime/data/spectra_val.pkl .
ln -s /home/tcat/diffcsp_exp5_prime/data/spectra_test.pkl .
ln -s /home/tcat/diffcsp_exp5_prime/data/spectra_holdout.pkl .
ln -s /home/tcat/diffcsp_exp5_prime/data/feff_features_imputed.pkl .
ln -s /home/tcat/diffcsp_exp5_prime/data/feff_feature_scaler.pkl .

# Split CSV
ln -s /home/tcat/diffcsp_exp5_prime/data/train_samples_v2.csv .
ln -s /home/tcat/diffcsp_exp5_prime/data/val_samples_v2.csv .
ln -s /home/tcat/diffcsp_exp5_prime/data/test_samples_v2.csv .
ln -s /home/tcat/diffcsp_exp5_prime/data/holdout_samples_v2.csv .
# holdout_structure_cache.pt 不存在,Phase 4 前再建,现在不动
```

#### 1.0.4 写入 data_integrity.json

```python
# step1/step1.0_cache_setup.py — 在完成上述 verify 后执行此脚本写 json
import json, subprocess, datetime

def md5(path):
    r = subprocess.run(['md5sum', path], capture_output=True, text=True)
    return r.stdout.split()[0]

result = {
    "audit_date": datetime.datetime.now().isoformat(),
    "shell_boundaries_md5": md5('/home/tcat/diffcsp_exp4/data/shell_boundaries.pkl'),
    "shell_boundaries_expected": "cf2050e4899160f5698ad2481377e94c",
    "shell_boundaries_ok": md5('/home/tcat/diffcsp_exp4/data/shell_boundaries.pkl') == "cf2050e4899160f5698ad2481377e94c",
    "exp5_prime_ckpt_md5": md5('/home/tcat/diffcsp_exp5_prime/checkpoints/composite_epoch169_score0.5881.ckpt'),
    "exp5_prime_ckpt_expected": "127afa44a850d8f7e4fcdae17e2761a1",
    "exp5_prime_ckpt_ok": md5('/home/tcat/diffcsp_exp5_prime/checkpoints/composite_epoch169_score0.5881.ckpt') == "127afa44a850d8f7e4fcdae17e2761a1",
    "L_VIRTUAL": 20.0,
    "MIN_PDIST": 1.5075718402862548,
    "exp6_files_present": True,  # SA1 手动确认
    "all_ok": True,              # SA1 手动确认所有项都通过后才写 True
}

with open('/home/tcat/experiment7/data/data_integrity.json', 'w') as f:
    json.dump(result, f, indent=2)
print("data_integrity.json written.")
```

---

### Step 1.1 — cp shared 文件 + 两项必做 diff

**目标产出**: `experiment7/shared/` 下所有文件就位。

#### 1.1.1 从 Exp5' cp

```bash
cd /home/tcat/experiment7/shared

# Dataset + datamodule (注意是 _v2 不是 _v3)
cp /home/tcat/diffcsp_exp5_prime/code/step3/xas_local_dataset_v2.py .
cp /home/tcat/diffcsp_exp5_prime/code/step3/xas_local_datamodule_v2.py .

# Spectrum encoder (先 cp,然后必须 diff)
cp /home/tcat/diffcsp_exp5_prime/code/step2/spectrum_encoder.py .

# Precompute cache 脚本 (Phase 4 用,先放 step3)
cp /home/tcat/diffcsp_exp5_prime/code/step3/precompute_structure_cache_exp5_prime.py ../step3/
```

#### 1.1.2 从 Exp6 cp

```bash
# step1 已完成产出
cp /home/tcat/experiment6_v7/shared/min_pdist_calibration.json .
cp /home/tcat/experiment6_v7/shared/shell_integrity_report.json .
cp /home/tcat/experiment6_v7/shared/exp6_element_vocab.json exp7_element_vocab.json
cp /home/tcat/experiment6_v7/shared/baseline_cps.json .
cp /home/tcat/experiment6_v7/shared/composite_score.py eval_cps.py
cp /home/tcat/experiment6_v7/shared/min_pdist_rdf_hist.png .
cp /home/tcat/experiment6_v7/shared/shell_n_atoms_hist.png .

# eval_cps.py: 加 license 注释,调整 import path
# 在文件开头加:
# # Imported from /home/tcat/experiment6_v7/shared/composite_score.py @ 2026-05-12
```

#### 1.1.3 ⚠️ 必做 Diff 1: step5_3 两版本

```bash
diff /home/tcat/diffcsp_exp5_prime/code/step5/step5_3_composite_score.py \
     /home/tcat/diffcsp_exp5_prime/code/step5/step5_3_composite_score_exp5_prime.py
```

**判断规则**:
- 如果两文件相同 → cp 任一,命名 `eval_step5_3.py`
- 如果有差异 → 默认选 `_exp5_prime` 后缀版(推测是 Exp5'-MA 修订版,与 final report 数字 composite_val=0.0801 对应)
- **无论哪种情况**,在 hand-back 中汇报 diff 结果摘要(有无差异,差在哪)
- 结果出乎预期(如两者都非 0.0801 对应版)→ **RAISE MA1**

#### 1.1.4 ⚠️ 必做 Diff 2: spectrum_encoder vs spectrum_tokenizer

```bash
diff /home/tcat/diffcsp_exp5_prime/code/step2/spectrum_encoder.py \
     /home/tcat/experiment6_v7/shared/spectrum_tokenizer.py
```

**判断规则**:
- 如果实质相同(只差注释/命名)→ 沿用已 cp 的 `spectrum_encoder.py`,不动
- 如果有实质差异 → 汇报差异内容,**RAISE MA1** 决议选哪个
- Exp7 命名锁定为 `spectrum_encoder.py`(spectrum 是 condition,不是 token)

#### 1.1.5 clone GitHub 起点仓库

```bash
cd /home/tcat/experiment7/_vendor

git clone https://github.com/eriklindernoren/PyTorch-GAN eriklindernoren_PyTorch-GAN
git clone https://github.com/gcucurull/cond-wgan-gp gcucurull_cond-wgan-gp
git clone https://github.com/christiancosgrove/pytorch-spectral-normalization-gan christiancosgrove_pytorch-spectral-normalization-gan

# Spectral normalization 直接 cp (零改动)
cp christiancosgrove_pytorch-spectral-normalization-gan/spectral_normalization.py \
   /home/tcat/experiment7/shared/

# GAN base 文件零改动 cp
cp eriklindernoren_PyTorch-GAN/implementations/cgan/cgan.py \
   /home/tcat/experiment7/shared/cgan_base.py
cp eriklindernoren_PyTorch-GAN/implementations/wgan_gp/wgan_gp.py \
   /home/tcat/experiment7/shared/wgan_gp_base.py
```

---

### Step 1.2 — Dataset Contract Audit(V1-V5 verify)

**目标产出**: `experiment7/data/dataset_contract_audit.json`

**没有此文件,不许进 Step 1.3。**

用 `xas_local_dataset_v2.py` 取 5 个 train sample,逐项 verify:

```python
# step1/step1.2_dataset_contract_audit.py

import sys, json, torch
sys.path.insert(0, '/home/tcat/experiment7/shared')
from xas_local_dataset_v2 import XASLocalDataset  # 实际 class 名 SA1 自行查

# 加载 5 个样本
ds = XASLocalDataset(
    split='train',
    cache_path='/home/tcat/experiment7/data/train_structure_cache.pt',
    # ... 其他参数按文件实际 API 填
)
samples = [ds[i] for i in range(5)]

# ---- V1: atom_types 是否按距离升序 ----
# 取 frac_coords 和 lengths,算到中心距离,check sort
results = {}
for i, s in enumerate(samples):
    fc = s['frac_coords']      # 预期 (20, 3)
    types = s['atom_types']    # 预期 (20,)
    lengths = s.get('lengths', None)  # 可能是 (3,) 或不存在
    
    # V3: frac_coords range
    results[f'sample_{i}_frac_range'] = f'[{fc.min().item():.3f}, {fc.max().item():.3f}]'
    
    # V2: padding method (找 types 中 <= 0 或 > 88 的值)
    results[f'sample_{i}_types_unique'] = types.unique().tolist()
    
    # V4: center_element 是否独立字段
    results[f'sample_{i}_has_center_element'] = 'center_element' in s or 'center_z' in s
    
    # V5: lengths scope
    results[f'sample_{i}_lengths'] = lengths.tolist() if lengths is not None else 'NOT_FOUND'

# V1 summary: check sort
v1_sorted_count = 0
for i, s in enumerate(samples):
    fc = torch.tensor(s['frac_coords']) if not isinstance(s['frac_coords'], torch.Tensor) else s['frac_coords']
    # 用 global L=20 近似(或用 per-sample lengths 若存在)
    dists = fc.norm(dim=-1)  # 简化:用 L2 norm of frac (等价于 Å if scale by 20)
    is_sorted = (dists[:-1] <= dists[1:]).all().item()
    results[f'sample_{i}_V1_sorted'] = is_sorted
    if is_sorted:
        v1_sorted_count += 1

audit = {
    "audit_date": __import__('datetime').datetime.now().isoformat(),
    "dataset_source": "/home/tcat/experiment7/shared/xas_local_dataset_v2.py",
    "V1_atom_types_sorted": v1_sorted_count == 5,  # True if all 5 sorted
    "V1_detail": {f"sample_{i}": results[f'sample_{i}_V1_sorted'] for i in range(5)},
    "V2_padding_method": "SA1_FILL_IN",  # 看 types 中特殊值,填 NO_OBJECT_IDX / mask / other
    "V3_frac_range": results,  # SA1 从上面结果判断
    "V4_center_element_separate": any(results[f'sample_{i}_has_center_element'] for i in range(5)),
    "V5_lengths_scope": "SA1_FILL_IN",  # per_sample / global / NOT_FOUND
    "decision": {
        "distance_matrix_compute_location": "training_loop_on_the_fly",
        "atom_types_resort_needed": not (v1_sorted_count == 5),
        "resort_implementation": "not_needed" if v1_sorted_count == 5 else "collate_fn",
    }
}

with open('/home/tcat/experiment7/data/dataset_contract_audit.json', 'w') as f:
    json.dump(audit, f, indent=2)
print(json.dumps(audit, indent=2))
```

**SA1 必须判断并填入**:
- V2: `atom_types` 中 padding 用什么值(在 dataset 代码里查 `NO_OBJECT_IDX` 或等价常量)
- V3: `frac_coords` 范围(`[-0.5, 0.5]` 或 `[0, 1]` 或其他)
- V4: center element 是单独字段还是包含在 `atom_types[0]` 里
- V5: `lengths` 是 per-sample `(3,)` 还是全局常量 `tensor([20., 20., 20.])`

**如果 V3 不是 `[-0.5, 0.5]` 也不是 `[0, 1]` → RAISE MA1,不许 patch。**

---

### Step 1.3 — Cartesian Sanity + Exp5' Dual Baseline

#### 1.3.1 Cartesian sanity 100/100

```python
# step1/step1.3a_cartesian_sanity.py
import torch, random
cache = torch.load('/home/tcat/experiment7/data/train_structure_cache.pt')
samples = random.sample(list(cache.values()), 100)
fails = []
for i, s in enumerate(samples):
    frac = torch.tensor(s['frac_coords']) if not isinstance(s['frac_coords'], torch.Tensor) else s['frac_coords']
    cart = frac * 20.0  # L=20
    diff = cart[:, None] - cart[None, :]        # (N, N, 3)
    d = diff.norm(dim=-1)                        # (N, N)
    mask = ~torch.eye(len(d), dtype=torch.bool)
    min_d = d[mask].min().item()
    if min_d < 0.7:
        fails.append((i, min_d))

if fails:
    print(f"FAIL: {len(fails)} samples with bond < 0.7 Å: {fails[:3]}")
    raise AssertionError("Cartesian sanity FAILED — RAISE MA1")
else:
    print("Cartesian sanity 100/100 PASS (min bond ≥ 0.7 Å)")
```

#### 1.3.2 Exp5' dual baseline (step1.3_dual_baseline_exp5_prime.py)

用 Exp5' best ckpt 跑双套评估,产出 `baseline_dual.json`。

```python
# step1/step1.3b_dual_baseline.py
# 加载 Exp5' ckpt,在 val split 上运行 sample,调用 eval_cps.py + eval_step5_3.py 各自独立评估
# 两套结果写入 baseline_dual.json

# 框架 (SA1 按 Exp5' 模型的实际 API 填):
import torch, json

# 1. 加载 Exp5' 模型
# exp5_model = ...load from composite_epoch169_score0.5881.ckpt...

# 2. sample val set (或先用 100 sample 验证双套一致性)
# preds = exp5_model.sample(val_loader)  # (N, 20, 3) + (N, 20) types

# 3. CPS (主套) — 独立调用
# from shared.eval_cps import compute_cps_dataset
# cps_result = compute_cps_dataset(preds, val_gt, shell_boundaries)

# 4. step5_3 (副套) — 独立调用
# from shared.eval_step5_3 import compute_step5_3_dataset
# step5_3_result = compute_step5_3_dataset(preds, val_gt, shell_boundaries)

# 5. 写 baseline_dual.json
baseline = {
    "exp5_prime": {
        "ckpt": "composite_epoch169_score0.5881.ckpt",
        "md5": "127afa44a850d8f7e4fcdae17e2761a1",
        "CPS_val": "SA1_FILL",
        "step5_3_composite_val": 0.0801,   # 已知真值,SA1 verify 实测是否一致
        "step5_3_gate_val": 0.640,
        "step5_3_collapse_val": 0.0,
        "dual_delta_relative": "SA1_FILL",  # |CPS - step5_3| / max
    },
    "exp6": None,   # 训完后回填
    "exp7": None,   # 训完后回填
}

with open('/home/tcat/experiment7/shared/baseline_dual.json', 'w') as f:
    json.dump(baseline, f, indent=2)
print("baseline_dual.json written.")
```

**注意**: baseline_dual.json 未完成前,不许定 acceptance_thresholds.json → 不许开始训练。

#### 1.3.3 设定 acceptance_thresholds.json

基于 baseline_dual.json 实测结果,设定:

```json
{
  "CPS": {
    "GREEN":  "val_cps >= Exp5_prime_CPS + 0.05",
    "AMBER":  "val_cps >= Exp5_prime_CPS",
    "RED":    "val_cps < Exp5_prime_CPS"
  },
  "step5_3": {
    "GREEN":  "val_step5_3_composite >= 0.10",
    "AMBER":  "val_step5_3_composite >= 0.080",
    "RED":    "val_step5_3_composite < 0.080"
  },
  "mode_diversity": {
    "fail":   "mode_diversity_per_spectrum == 0 for > 50% spectra"
  },
  "dual_delta": {
    "raise_threshold": 0.30
  }
}
```

SA1 将 Exp5' 实测 CPS 数值回填后,交 MA1 最终 confirm 才算锁定。

---

### Step 1.4 — GAN Architecture Build + Smoke Test

**这是 Phase 1 的主要工程任务。按顺序:**

#### 1.4.1 合并 cgan_base + wgan_gp_base → cond_wgan_gp.py

参考 `_vendor/gcucurull_cond-wgan-gp/` 的合并方式:
- cGAN 提供:Generator/Discriminator 接受 condition 拼接的框架
- WGAN-GP 提供:critic loss、gradient penalty、n_critic 训练循环
- 合并重点:condition 从 class label(one-hot)改为 `spectrum_cond (B, 256)` 连续向量

**合并后必须保留(不许删)**:
- `n_critic = 5` 训练循环
- `lambda_gp = 10.0` gradient penalty
- `beta1 = 0.0, beta2 = 0.9` (WGAN-GP Adam)
- TTUR: G lr = 1e-4,D lr = 4e-4

#### 1.4.2 Generator architecture(改造 cgan_base.py)

将 image ConvTranspose2d decoder 替换为:

```python
class LocalStructureGenerator(nn.Module):
    """
    条件输入: spectrum_cond (B, 256) + center_z_embed (B, 16) + z_noise (B, 128)
    输出: pred_frac_coords (B, 20, 3) + pred_type_logits (B, 20, K+1)
    
    K = len(neighbor_vocab),约 88。+1 for NO_OBJECT.
    """
    def __init__(self, noise_dim=128, spectrum_dim=256, center_embed_dim=16,
                 n_atoms=20, n_types=89):  # 89 = 88 + NO_OBJECT
        super().__init__()
        cond_dim = noise_dim + spectrum_dim + center_embed_dim  # 400
        
        # 主 MLP backbone
        self.backbone = nn.Sequential(
            nn.Linear(cond_dim, 512), nn.LayerNorm(512), nn.LeakyReLU(0.2),
            nn.Linear(512, 512), nn.LayerNorm(512), nn.LeakyReLU(0.2),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.LeakyReLU(0.2),
        )
        
        # 输出 head
        self.pos_head  = nn.Linear(256, n_atoms * 3)    # frac coords
        self.type_head = nn.Linear(256, n_atoms * n_types)  # type logits
        self.n_atoms = n_atoms
        self.n_types = n_types
    
    def forward(self, z, spectrum_cond, center_embed):
        x = torch.cat([z, spectrum_cond, center_embed], dim=-1)  # (B, 400)
        h = self.backbone(x)
        
        # coords: tanh → [-1, 1] → * 0.5 → [-0.5, 0.5] frac range
        coords = torch.tanh(self.pos_head(h)).view(-1, self.n_atoms, 3) * 0.5
        
        # types: logits,CE 在外面算
        types = self.type_head(h).view(-1, self.n_atoms, self.n_types)
        
        return coords, types
```

#### 1.4.3 Discriminator architecture(改造 wgan_gp_base.py)

```python
class LocalStructureDiscriminator(nn.Module):
    """
    输入: distance_matrix (B, 20, 20) + atom_dist_to_center (B, 20) + atom_types_onehot (B, 20, K)
          + spectrum_cond (B, 256) [condition]
    输出: critic_score (B, 1)  ← WGAN,不是概率
    """
    def __init__(self, n_atoms=20, n_types=88, spectrum_dim=256):
        super().__init__()
        
        # Distance matrix branch (2D → 1D flatten)
        dist_feat_dim = n_atoms * n_atoms  # 400
        
        # Struct input dim: dist_matrix + dist_to_center + types_onehot
        struct_dim = dist_feat_dim + n_atoms + n_atoms * n_types  # 400+20+1760=2180
        
        cond_dim = spectrum_dim  # 256
        in_dim = struct_dim + cond_dim  # 2436
        
        self.net = nn.Sequential(
            SpectralNorm(nn.Linear(in_dim, 512)), nn.LeakyReLU(0.2),
            SpectralNorm(nn.Linear(512, 256)),   nn.LeakyReLU(0.2),
            SpectralNorm(nn.Linear(256, 128)),   nn.LeakyReLU(0.2),
            SpectralNorm(nn.Linear(128, 1)),
        )
    
    def forward(self, struct_dict, spectrum_cond):
        dist_mat = struct_dict['dist_matrix']          # (B, 20, 20)
        dist_center = struct_dict['dist_to_center']    # (B, 20)
        types_oh = struct_dict['types_onehot']         # (B, 20, K)
        
        flat = torch.cat([
            dist_mat.flatten(1),    # (B, 400)
            dist_center,            # (B, 20)
            types_oh.flatten(1),    # (B, 1760)
            spectrum_cond,          # (B, 256)
        ], dim=-1)                  # (B, 2436)
        
        return self.net(flat)       # (B, 1)
```

**SpectralNorm** 从 `shared/spectral_normalization.py` import。Discriminator 参数量目标 ≈ G × (0.5-1.0),开训前打印两者参数量做检查。

#### 1.4.4 Curriculum callbacks

按 Proposal §6.2.4 精确实现 `CurriculumCkptFilter` + `CurriculumEarlyStopFilter`,写入 `shared/curriculum_callbacks.py`。

关键常量(不许改):
```python
CURRICULUM_EPOCH_BOUNDARIES = [50, 100, 150]
CURRICULUM_FRACTIONS = [0.33, 0.53, 0.73, 1.00]
```

#### 1.4.5 Smoke test — L3 + L4 双层 active verify

```python
# step1/step1.4_smoke_test.py

# 1. 取 5 个 batch
# 2. G forward: z ~ N(0,1), spectrum_cond, center_embed → pred_coords, pred_types
# 3. D forward: real struct + fake struct → critic scores
# 4. 四项 loss 计算: G_adv, pairwise_min(curriculum epoch=0), type_ce, GP
# 5. backward() on all losses — 检查梯度非 None, 非 NaN
# 6. 打印参数量: G / D

# --- L3 check ---
print("=== L3: Training active ===")
# 所有 loss 值 finite + 梯度非零

# --- L4 check (最重要) ---
print("=== L4: Evaluation active ===")
# 对 5 sample 的 G 输出跑 step5_3 片段
# 检查: n_pred_shells_per_sample > 0 的比例
from shared.eval_step5_3 import count_pred_shells  # SA1 找实际函数名
n_shells = [count_pred_shells(pred_coords[i], shell_boundaries[sample_id]) 
            for i, sample_id in enumerate(sample_ids)]
zero_ratio = sum(1 for n in n_shells if n == 0) / len(n_shells)
assert zero_ratio < 0.20, f"L4 FAIL: {zero_ratio:.0%} samples have 0 pred shells — RAISE MA1"
print(f"L4 PASS: n_pred_shells>0 比例 = {1-zero_ratio:.0%}")

# --- Curriculum verify ---
print("=== Curriculum schedule verify ===")
for ep in [0, 49, 50, 99, 100, 149, 150, 151]:
    pdist = get_curriculum_min_pdist(ep, MIN_PDIST)
    print(f"  epoch={ep}: min_pdist={pdist:.4f}")
# 预期:
# epoch=0:   ~0.4975  (0.33 × 1.5076)
# epoch=50:  ~0.7990  (0.53 × 1.5076)
# epoch=100: ~1.1006  (0.73 × 1.5076)
# epoch=150: ~1.5076  (1.00 × 1.5076)
```

**Smoke test 全部通过后**,在 hand-back 中汇报各 loss 数值 + L3/L4 结论 + curriculum 数值表。

---

### Step 1.5 — Sanity Training(10 epoch)

**仅在 step1.4 smoke test 全通过后启动。**

```yaml
# sanity run config
max_epochs: 10
batch_size: 32
n_critic: 5
G_lr: 1e-4
D_lr: 4e-4
# curriculum 从 epoch 0 开始,10 epoch 全在 Phase 0 (min_pdist = 0.33 × cal)
```

**10 epoch 期间必须 monitor(每 epoch 打印)**:

| Metric | 健康区间 |
|---|---|
| `G_loss_adversarial` | 有限值,不 NaN |
| `D_loss_critic` | < 0(D 更新后 fake - real 应趋向负值) |
| `D_gp_loss` | [0.01, 5.0](初期可能偏大) |
| `G_pmin_loss` | 有限值,epoch 5 后应下降 |
| `G_type_ce_loss` | 有限值,初期可能 > 4(log(89) ≈ 4.5 是随机基线) |
| `mode_diversity` | > 0(不是全零输出) |
| `train_curriculum_min_pdist` | epoch 0-9 应全是 0.33 × 1.5076 ≈ 0.4975 |

**如果 10 epoch 内出现 NaN → 立即 raise MA1,不许尝试自行修复。**

10 epoch 正常完成后 hand-back 给 MA1,附完整训练 log。

---

## §4 Hand-back 验收清单

SA1 向 MA1 hand-back 时,必须提供以下所有项:

```
[ ] data_integrity.json 已写入,所有 md5 verify = True
[ ] dataset_contract_audit.json 已写入(V1-V5 全部填写,无 FILL_IN)
[ ] diff 结果 1: step5_3 两版本 — 汇报是否有差异,选哪个
[ ] diff 结果 2: spectrum_encoder vs spectrum_tokenizer — 汇报是否有差异
[ ] baseline_dual.json 已写入,Exp5' CPS val 已实测
[ ] acceptance_thresholds.json 草稿(等 MA1 final confirm)
[ ] smoke test 通过报告:
    - G/D 参数量
    - 4 项 loss 数值
    - L3 通过 ✓
    - L4 通过 ✓ (n_pred_shells > 0 比例 ≥ 80%)
    - Curriculum 数值表
[ ] 10 epoch sanity 训练 log(完整 stdout 或 tensorboard 截图)
[ ] 任何 RAISE 项的完整诊断(若有)
```

---

## §5 RAISE 触发条件速查

下列任一情况立即停止并 raise MA1:

1. 任意 md5 verify 失败
2. V1/V2/V3/V4/V5 任一项 verify 出乎预期范围
3. step5_3 两版本 diff 结果出乎预期
4. spectrum_encoder vs spectrum_tokenizer 有实质差异
5. G 或 D forward 报错(shape mismatch, NaN, etc.)
6. L4: n_pred_shells > 0 比例 < 80%
7. 双套 baseline 差 > 30% relative
8. 10 epoch 内出现 NaN loss
9. 任何不在本文档覆盖范围的异常情况

**Raise 格式**:

```
[RAISE] Step X.Y — <问题描述>
实测值: <贴具体数字/错误信息>
预期值: <预期>
建议: <SA1 对可能原因的判断>
```

---

*MA1 撰写,2026-05-12*
*基于 EXP7_GAN_PROPOSAL_v6.md + EXP7_MA1_ONBOARDING_MANIFEST.md + EXPERIMENT5_SERIES_FINAL_REPORT.md*
