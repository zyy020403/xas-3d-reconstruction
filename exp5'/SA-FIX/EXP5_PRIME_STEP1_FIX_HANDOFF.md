# EXP5_PRIME_STEP1_FIX_HANDOFF.md
# SA-EXP5'-STEP1-FIX 任务 launch note(Exp5'-MA → SA-EXP5'-STEP1-FIX)

> **From**: Exp5'-MA(Exp5 系列第 3 任 Main Agent)
> **To**: SA-EXP5'-STEP1-FIX(新一棒 Sub-Agent,起自干净窗口)
> **日期**: 2026-05-02
> **任务范围**: 修复 errata 3 揭示的 L=6 fold artifact,改 L_VIRTUAL=6→20,dataset cache 重建,smoke 重跑(~ 1-2 天)
> **预期 hand-back**: 中期报告(cartesian sanity PASS + smoke 6 active loss PASS)→ Exp5'-MA review → 启动 STEP1-续(原 §1.5/§1.6/§1.7)
> **本文档定位**: 给你的精确技术规格,errata 3 §9 是 surgery 清单源头,本 launch note 加 PASS gate + 红线 + hand-back 模板

---

## §0 一屏掌握

### 0.1 你是谁,在做什么

你是 **SA-EXP5'-STEP1-FIX**,新一棒 SA。前一棒 SA-EXP5'-STEP1 在 §1.4 完成后自查 dataset 输出 frac_coords 物理性,发现 fold artifact 灾难。SA-EXP5'-STEP1-AUDIT 完成根因诊断 + 出 errata 3 final。**你接 errata 3 §9 surgery 清单,改代码 + cache rebuild + smoke 重跑**。

**你的任务 6 步**:

| 步 | 任务 | 工程量 |
|---|---|---|
| F1 | 解锁 4 个 md5 + cp `.bak_pre_step1_fix` 锚点 | 5 分钟 |
| F2 | 改 8 个文件的 `L_VIRTUAL=6→20`(errata 3 §9 清单) | 0.3 天 |
| F3 | dataset `__init__` 加 cartesian sanity check(errata 3 §7.1)| 0.3 天 |
| F4 | dataset cache rebuild(预计 2-4h 机器跑)+ 重跑 cartesian sanity 验证 fold 消除 | 半天 |
| F5 | forward_test Phase 6.7 重跑(三件套 loss 在新 L=20 下 dummy 行为)+ smoke 2 epoch × 10 batch 重跑 | 0.3 天 |
| F6 | 中期 hand-back 给 Exp5'-MA → review → 启动 STEP1-续(原 §1.5/§1.6/§1.7) | 0.2 天 |

### 0.2 必读 6 份(按顺序)

1. **EXP5_PRIME_MA_HANDOFF.md** — Exp5' 接班背景(知道你在哪个实验序列)
2. **EXP5_PRIME_PROPOSAL.md** §2(三件套 loss 公式)+ §1.2(架构 / L 沿用)
3. **EXPERIMENT5_FINAL_REPORT_v2.md** §0 verdict + §4 数据处理沿用清单
4. **EXP5_FILE_GUIDE_v2.md** §6 工作目录 + §8 PYTHONPATH 三段
5. **EXP4_FINAL_REPORT_ERRATA_2.md** — `_density_loss` 旧归因(注意:errata 3 §5.2 已扩充为三层,你以 errata 3 为准)
6. **EXP4_FINAL_REPORT_ERRATA_3.md** ⭐ — **核心**:fold 根因 + 路径 B 决议(L=6→20)+ §9 待改 8 个文件清单 + §7.1 MIN_BOND_LENGTH=0.7 Å sanity 阈值

### 0.3 启动后第一条回复格式

```
我已读完 6 份必读文档。复述任务 [6-8 条,含为什么 L=6 是 bug、为什么选 L=20、§9 8 个文件改什么]。
最易踩坑 [4 条]。
计划: 第 1 步 ssh 跑 verify [...] 确认前 SA 4 个 md5 锁定文件状态完整。
```

### 0.4 Exp5'-MA 已拍板的 5 条不再讨论

1. **L_VIRTUAL = 20.0**(不是 12 也不是 21,errata 3 §8 决议)。`L/2=10 ≥ CUTOFF_R=10`,等号成立但完全消除 fold(无浮点边界 fold,因为 fold 触发条件是严格 `>L/2`)。
2. **不做 element-aware 阈值**(errata 3 §8.2 决议)。`_pairwise_min_distance_penalty` threshold 保持 1.5 Å 全局。
3. **shell_boundaries.pkl 不动,不重建**(errata 3 §3 已确认 cartesian Å 干净)。
4. **`_density_loss` 不动**。它用 `% 1.0` 周期归一,不依赖 L 数值;errata 2 §1 揭示它是塌缩剂但 Exp5' 沿用 cost=0.2 不调(留 Exp5''/Exp6 ablation)。
5. **MIN_BOND_LENGTH = 0.7 Å**(H-H 物理下限 ~0.74 Å)用于 dataset sanity check,**不是 1.5 Å**(1.5 是 pairwise loss 阈值,两件事别混,errata 3 §7.1 强制)。

---

## §1 Step F1 — 解锁 4 个 md5 + 锚点(5 分钟)

### 1.1 前 SA 锁定的 4 个 md5(STEP1 hand-back 留下)

| 文件 | 锁定 md5(STEP1 hand-back) |
|---|---|
| `step3/diffusion_w_type_xas.py` | `f6a65ea0e0f2d37d09194e8ef1b45c28` |
| `step3/conf_xas/model/diffusion_xas.yaml` | `f73123a16166b220646af3537f7ece5b` |
| `step3/xas_local_datamodule_v2.py` | `a129dca8e4083e82d0dd2fbb11f3f917` |
| `step3/xas_local_dataset_v2.py` | `68b5d24fed8e7ee48080597fd2b26ecf` |

### 1.2 启动 verify

```bash
ssh tcat@scsmlnprd02.its.auckland.ac.nz
cd /home/tcat/diffcsp_exp5_prime/code/step3

# 验证 4 个 md5 与 STEP1 hand-back 一致
md5sum diffusion_w_type_xas.py conf_xas/model/diffusion_xas.yaml \
       xas_local_datamodule_v2.py xas_local_dataset_v2.py

# 期望 4 行严格匹配上面 §1.1 表
```

如任一 md5 不匹配 → **stop 立即贴日志给 Exp5'-MA**(说明 STEP1-AUDIT 期间有人改过文件,违背 audit-only 红线)。

### 1.3 锚点 cp(STEP1-FIX 改前)

```bash
cd /home/tcat/diffcsp_exp5_prime/code/step3
for f in diffusion_w_type_xas.py xas_local_dataset_v2.py xas_local_datamodule_v2.py conf_xas/model/diffusion_xas.yaml; do
    cp "$f" "${f}.bak_pre_step1_fix"
done

# step5 / step6 涉及文件也加锚点(errata 3 §9 清单)
cd /home/tcat/diffcsp_exp5_prime/code/step5
cp step5_2_compute_metrics.py step5_2_compute_metrics.py.bak_pre_step1_fix
cp step5_3_smoke_test.py step5_3_smoke_test.py.bak_pre_step1_fix 2>/dev/null || true

cd /home/tcat/diffcsp_exp5_prime/code/step6 2>/dev/null
cp step6_visualize_v2.py step6_visualize_v2.py.bak_pre_step1_fix 2>/dev/null || true
cp pick_samples_for_feff.py pick_samples_for_feff.py.bak_pre_step1_fix 2>/dev/null || true
```

### 1.4 PASS gate F1

- ✅ 4 个 STEP1 md5 严格匹配
- ✅ 8 个 `.bak_pre_step1_fix` 锚点存在(允许部分 step5/step6 文件不在,SA 报告即可)

---

## §2 Step F2 — 改 8 个文件 L_VIRTUAL=6→20(0.3 天)

### 2.1 errata 3 §9 surgery 清单

| 文件 | 位置 | 改动 |
|---|---|---|
| `step3/xas_local_dataset_v2.py` | L69 `L_VIRTUAL = 6.0` | → `L_VIRTUAL = 20.0` + 加 §7.1 cartesian sanity check(F3 加,F2 暂不加) |
| `step3/xas_local_datamodule_v2.py` | L56 `L_VIRTUAL = 6.0` | → 20.0 |
| `step3/diffusion_w_type_xas.py` | L99 `L_VIRTUAL = 6.0` | → 20.0 |
| `step3/conf_xas/model/diffusion_xas.yaml` | 无 L_VIRTUAL hardcode | **不改** |
| `step5/step5_2_compute_metrics.py` | `L=6.0` default args | → 20.0 |
| `step5/step5_3_smoke_test.py` | `L = 6.0` | → 20.0 |
| `step6/step6_visualize_v2.py`(若存在)| `L = 6.0` | → 20.0 |
| `step6/pick_samples_for_feff.py`(若存在)| `L = 6.0` | → 20.0 |

### 2.2 改动方法 — grep 验证 + sed 改 + grep 再验证

```bash
# (a) 改前再 grep 一次,确认 errata 3 §9 清单完整(避免遗漏)
cd /home/tcat/diffcsp_exp5_prime/code
grep -rn "L_VIRTUAL\s*=\s*6" --include="*.py"
grep -rn "L\s*=\s*6\.0" --include="*.py"

# 期望命中数 ≈ errata 3 §9 表行数。如多于清单 → 先停,贴 grep 输出给 Exp5'-MA。

# (b) 用 sed 改(逐文件,不用 -i.bak 因为已经有 .bak_pre_step1_fix)
sed -i 's/L_VIRTUAL\s*=\s*6\.0/L_VIRTUAL = 20.0/g' \
    step3/xas_local_dataset_v2.py \
    step3/xas_local_datamodule_v2.py \
    step3/diffusion_w_type_xas.py

# step5/step6 的 `L = 6.0` 句法可能不同,逐文件 view + str_replace 改,不用 sed 通配
# (因为 `L = 6.0` 也可能匹配到注释或其他变量)

# (c) 改后再 grep 验证
grep -rn "L_VIRTUAL\s*=\s*6" --include="*.py"   # 期望: 0 hit(除 .bak_* 文件)
grep -rn "L_VIRTUAL\s*=\s*20" --include="*.py"  # 期望: ≥3 hit
```

### 2.3 关键注意事项

1. **`% 1.0` / `.round()` / `min-image` 操作不动**。这些是 frac space 的周期归一,与 L 数值无关。errata 3 §6 说"`_density_loss` `% 1.0` 周期归一,与 fold 无关",同理三件套 loss 内的 `diff_frac.round()` 也是 frac 上的周期归一,**不变**。
2. **CUTOFF_R = 10.0 不动**(errata 3 §8 排除选项 D 的理由)。
3. **N_NEIGHBORS = 20 不动**(L 改了不影响邻居数量上界)。
4. **yaml 不改**(errata 3 §9 表确认 yaml 无 L_VIRTUAL hardcode)。
5. **三件套 loss 函数签名 `L=L_VIRTUAL`**(STEP1 SA 已写)— 这里 `L_VIRTUAL` 是 module-level 常量,改了 module-level 数值后,函数 default arg 自动跟随。**SA 不要改函数签名 default 写死成 20**(这会失去常量解耦)。
6. **Phase 6.5 仍 SKIPPED-by-design**(final report v2 §5.1)。

### 2.4 PASS gate F2

- ✅ grep `L_VIRTUAL = 6` 在 .py 文件(不含 .bak_*)0 hit
- ✅ grep `L_VIRTUAL = 20` 在 .py 文件 ≥ 3 hit(step3 三个文件)
- ✅ step5/step6 的 `L = 6.0` 也已改 20.0(逐文件 view 确认)
- ✅ yaml / CUTOFF_R / N_NEIGHBORS 未动

---

## §3 Step F3 — dataset cartesian sanity check(0.3 天)

### 3.1 改动文件

`/home/tcat/diffcsp_exp5_prime/code/step3/xas_local_dataset_v2.py`

### 3.2 改动 pseudocode

在 `__init__` 末尾(STEP1 SA 已加的 100-sample sanity check 之后),**加** cartesian distance sanity check:

```python
# 在 STEP1 已有的 sample_name schema sanity check 之后,加:

# ⭐ STEP1-FIX: cartesian distance sanity check (errata 3 §7.1 ExpN 不变量级 SOP)
# 验证 dataset 输出的 frac_coords 在 cartesian Å 下两两距离 ≥ MIN_BOND_LENGTH
import torch
MIN_BOND_LENGTH = 0.7  # Å, H-H 物理下限,不是 pairwise loss 的 1.5 Å 阈值

logger.info(f"[Exp5' STEP1-FIX] running cartesian sanity check (L={L_VIRTUAL}, MIN_BOND={MIN_BOND_LENGTH} Å)")
n_check = min(100, len(self.indices))
n_pass = 0
n_fail = 0
fail_examples = []
for i in range(n_check):
    sample = self.__getitem__(i)
    if sample is None:
        continue
    frac = sample.frac_coords if hasattr(sample, 'frac_coords') else sample['frac_coords']
    cart = frac * L_VIRTUAL  # (N, 3) Å
    # pairwise cart distances WITHOUT min-image (raw cartesian, fold-free)
    diff = cart.unsqueeze(0) - cart.unsqueeze(1)  # (N, N, 3)
    d = diff.norm(dim=-1)                          # (N, N)
    # mask self-distance + upper triangle
    n = cart.shape[0]
    mask = torch.triu(torch.ones(n, n), diagonal=1).bool()
    d_pairs = d[mask]
    min_d = float(d_pairs.min())
    if min_d >= MIN_BOND_LENGTH:
        n_pass += 1
    else:
        n_fail += 1
        if len(fail_examples) < 5:
            fail_examples.append((i, min_d))

logger.info(f"[Exp5' STEP1-FIX] cartesian sanity: {n_pass}/{n_check} PASS, {n_fail} FAIL (threshold {MIN_BOND_LENGTH} Å)")
if n_fail > n_check * 0.05:  # > 5% 失败认为有系统问题
    raise RuntimeError(
        f"[Exp5' STEP1-FIX] cartesian sanity FAIL: {n_fail}/{n_check} samples have min_d < {MIN_BOND_LENGTH} Å. "
        f"Examples (idx, min_d): {fail_examples}\n"
        f"This indicates fold artifact NOT fully fixed by L={L_VIRTUAL}, or dataset has true overlapping sites."
    )
```

### 3.3 关键注意事项

1. **min-image 不算**:cartesian sanity 是验证**真实物理距离**,不是 pairwise loss 视角。所以 `diff = cart - cart` 不做 `.round()`,直接算 norm。
2. **轻元素短键容差 ≤ 5%**:errata 3 §3 报 0.06% < 1.5 Å 真短键。0.7 Å 阈值下 fail 率应 << 1%。设 5% 是宽松上限,真出现 5% 以上 fail 必定是 fold 没修干净或别的 bug。
3. **fail 立即 raise**:不允许"warning 跑下去",errata 3 §7.1 强制 SOP。
4. **F3 在 cache rebuild 之前还会跑一次**(用旧 cache),会 fail(因为旧 cache 是 L=6 fold-distorted)。这是预期。SA F3 完成后立即进 F4 rebuild cache,rebuild 后再跑 sanity 必须 PASS。

### 3.4 PASS gate F3

- ✅ sanity check 代码加进 `__init__`
- ✅ 在**新建测试脚本**上跑(不要在生产 dataset cache 上跑,见 F4)— 先 mock 一个 L=20 dataset 实例验证代码逻辑无错(import + 函数能跑通)
- ✅ MIN_BOND_LENGTH 写在文件顶部 module-level 常量(便于后续 ExpN 调)

---

## §4 Step F4 — dataset cache rebuild + 验证(半天)

### 4.1 cache 位置 + rebuild 触发

```bash
# 找 cache 位置(STEP1 SA 改过 dataset,cache 路径可能在 hand-back 或 dataset_v2.py 里)
cd /home/tcat/diffcsp_exp5_prime
grep -rn "cache" code/step3/xas_local_dataset_v2.py | head -20
ls -la /home/tcat/diffcsp_exp5_prime/data/  # 看是否有 cache pkl
ls -la /home/tcat/diffcsp_exp4/data/ | grep -i cache
```

### 4.2 rebuild 命令 — SA 不擅自删

**关键**:cache 是 Exp4 沿用,**Exp5' / Exp4 / Exp5 v2 共享 data 目录(symlink)**。直接删 Exp4 cache 会污染历史档案。

**正确做法**:在 `/home/tcat/diffcsp_exp5_prime/` 下建独立 cache(若 dataset 现在是写到共享 data 目录,SA 改 cache 路径写到 exp5_prime/cache/)。

**SA F4 第 1 件事:贴 cache 现状给 Exp5'-MA**:

```bash
# 报告:
# 1. cache 路径 grep 结果
# 2. cache pkl 文件位置(共享还是独立?)
# 3. cache 大小
# 4. 是否需要改 cache 路径到 exp5_prime/cache/
```

**Exp5'-MA 看完决议**:
- 若 cache 是 Exp5 独立:直接删 cache + reload 重建
- 若 cache 是 Exp4 共享:改 dataset_v2.py 让 cache 写到 `/home/tcat/diffcsp_exp5_prime/cache/`,old cache 不动

### 4.3 rebuild 后跑 cartesian sanity(预期 PASS)

cache rebuild 完成后(机器 2-4h),跑:

```bash
cd /home/tcat/diffcsp_exp5_prime/code
export PYTHONPATH=/home/tcat/diffcsp_exp5_prime/code/step3:/home/tcat/diffcsp_exp5_prime/code/step2:/home/tcat/diffcsp_exp4/code

/home/tcat/conda_envs/mlff/bin/python -c "
from xas_local_dataset_v2 import XasLocalDatasetV2
ds = XasLocalDatasetV2(...)  # 实例化,触发 __init__ sanity check
print('Cartesian sanity PASS' if True else 'check log')
" 2>&1 | tee /home/tcat/diffcsp_exp5_prime/logs/cartesian_sanity_post_rebuild.log
```

### 4.4 PASS gate F4

- ✅ cache 路径决议落地(独立或共享 SA 报告 + Exp5'-MA ack)
- ✅ cache rebuild 完成,新 cache pkl 大小合理(~ 几 GB,与原 L=6 cache 量级一致)
- ✅ cartesian sanity ≥ 95/100 PASS(MIN_BOND_LENGTH=0.7 Å)
- ✅ 关键样本验证:errata 3 §2.2 那个 `[+3.2, 0, 0]` / `[-3.2, 0, 0]` 案例在新 L=20 下 fold 不再触发 — SA 取一个有 fold artifact 的 sample(STEP1 找过的),手动算新 dataset 输出的 frac_coords,验证两两 cart > 1.5 Å

---

## §5 Step F5 — forward_test Phase 6.7 + smoke 重跑(0.3 天)

### 5.1 forward_test Phase 6.7 重跑

STEP1 SA 已加 Phase 6.7 a-g 7 项 sanity test。本 STEP1-FIX 重跑,**关注两件**:

1. **Phase 6.7.c collapse batch loss 数值变化**:STEP1 SA 跑出 `loss_pairwise_min = 2.2457`,与理论 `relu(1.5)² ≈ 2.25` 误差 < 0.005。新 L=20 下:
   - `coords_collapse = torch.zeros(20, 3)` 仍是全原点,frac=0
   - `frac diff = 0 - 0 = 0` → fold 后还是 0 → cart d = 0 × 20 = 0
   - violation = ReLU(1.5 - 0) = 1.5 → loss = 1.5² = 2.25(单 sample)
   - **理论值不变**(因为 collapse 是退化 case,L 改了也是 0)
2. **Phase 6.7.d spread batch loss 数值变化**:`coords_spread = (rand - 0.5) * 0.9` 是 frac in [-0.45, 0.45]。
   - 旧 L=6:cart in [-2.7, 2.7] Å,pairwise cart 多数 > 1.5 Å,loss 接近 0
   - 新 L=20:cart in [-9.0, 9.0] Å,pairwise cart 大,loss 完全 0
   - **新 L=20 下 spread loss 应 < 旧 L=6 spread loss**,但 collapse loss 仍 2.25。比较关系不变。

### 5.2 smoke test 重跑

```bash
# import path verify(STEP1 launch note §8.2 强制)
export PYTHONPATH=/home/tcat/diffcsp_exp5_prime/code/step3:/home/tcat/diffcsp_exp5_prime/code/step2:/home/tcat/diffcsp_exp4/code
/home/tcat/conda_envs/mlff/bin/python -c "
import xas_local_dataset_v2, xas_local_datamodule_v2, diffusion_w_type_xas
print(f'dataset: {xas_local_dataset_v2.__file__}')
print(f'datamodule: {xas_local_datamodule_v2.__file__}')
print(f'model: {diffusion_w_type_xas.__file__}')
print(f'L_VIRTUAL (dataset): {xas_local_dataset_v2.L_VIRTUAL}')      # 期望 20.0
print(f'L_VIRTUAL (model):   {diffusion_w_type_xas.L_VIRTUAL}')       # 期望 20.0
"

# smoke 跑
cd /home/tcat/diffcsp_exp5_prime/code/step4
/home/tcat/conda_envs/mlff/bin/python step4_1_smoke_test.py 2>&1 | tee /home/tcat/diffcsp_exp5_prime/logs/smoke_step1_fix.log
```

### 5.3 关键监控:三件套 loss 数值在新 L=20 下的变化

| Loss | 旧 L=6 STEP1 smoke | 新 L=20 STEP1-FIX smoke | 解读 |
|---|---|---|---|
| `loss_pairwise_min` | epoch 0 ~ 高(64% 虚假违反 + 真违反)| 应**显著下降**(只剩真违反)| ⭐ 关键信号 |
| `loss_shell_dist` | epoch 0 大(fold 后 shell 切错)| 应**变化**(L 改了,gap 切壳分布变了)| 关注趋势 |
| `loss_shell_count` | epoch 0 大 | 应**变化**(同上)| 关注趋势 |
| `loss_coord` | 中等 | 应**显著上升**(原本 fold 把目标压缩,现在目标分布更宽)| 预期 |
| `loss_density` | 中等 | 应**上升**(同上)| 预期 |
| `loss_type` | 中等 | 应**接近持平**(type 与 L 无关)| 预期 |

**SA 不调任何 cost / λ**(launch note §10 红线)。**报告趋势,不行动**。

### 5.4 PASS gate F5

- ✅ import path 全部以 `/home/tcat/diffcsp_exp5_prime/` 开头
- ✅ `L_VIRTUAL` 在 dataset / model 两处 print = 20.0
- ✅ Phase 6.7 a-g 全 PASS
- ✅ smoke 2 epoch × 10 batch 完成,6 active loss 全 finite
- ✅ best ckpt callback 触发(ckpt 文件落盘,STEP1 已实施,本步只验证)

---

## §6 Step F6 — 中期 hand-back

### 6.1 hand-back 必报字段(写进 OUTPUT.md)

落服务器 `/home/tcat/diffcsp_exp5_prime/EXP5_PRIME_STEP1_FIX_HANDBACK.md`:

1. **F1 PASS evidence**: 4 个 STEP1 md5 verify + 8 个 `.bak_pre_step1_fix` 锚点 ls
2. **F2 PASS evidence**: grep `L_VIRTUAL = 6` = 0 hit + grep `L_VIRTUAL = 20` ≥ 3 hit + 改动文件新 md5
3. **F3 PASS evidence**: dataset 加 sanity check 代码 diff 段
4. **F4 PASS evidence**:
   - cache 路径决议(独立 / 共享)
   - cache rebuild 完成时间 + 大小
   - **cartesian sanity post-rebuild ≥ 95/100 PASS log**(关键)
   - errata 3 §2.2 fold 案例样本手动验证(新 L=20 下两两 cart 数值)
5. **F5 PASS evidence**:
   - import path verify stdout(必须 `exp5_prime/` 开头)
   - `L_VIRTUAL` print = 20.0
   - Phase 6.7 a-g 全 PASS log
   - smoke 6 active loss 表(epoch 0 / epoch 1 mean,与 STEP1 旧 smoke 对比)
6. **磁盘 + RSS 增量**: du -sh exp5_prime/ + ps RSS smoke 前后(cache rebuild 后 RSS 应 ~ 387 MB shell_boundaries + 几 GB cache)
7. **OPEN 问题**: 任何不确定贴日志,**不擅自 fix**

### 6.2 Exp5'-MA review checklist

| 项 | 通过 | 说明 |
|---|---|---|
| F1-F5 全 PASS gate | | 逐项 |
| cartesian sanity post-rebuild ≥ 95/100 | | ⭐ 核心,fail 立即停 |
| errata 3 fold 案例手动验证(新 L=20 不再 fold)| | ⭐ 硬证 |
| smoke 6 loss 数量级与 §5.3 表预期一致 | | watch-only,异常贴日志 |
| 磁盘 ≥ 50G avail(SA-EXP5'-train 32-40h 训练前提)| | |

如全过 → Exp5'-MA 写 STEP1-续 launch note,启动 §1.5(train.py from-scratch + composite ckpt callback)→ §1.6(forward_test 6.7 已在 F5 跑过)→ §1.7(完整 smoke,本 F5 已部分覆盖)→ STEP2-train。

如 cartesian sanity fail → 立即停,贴日志,Exp5'-MA 决议(可能要 L=21 留余量,或重审 fold 触发条件)。

---

## §7 红线汇总(SA-EXP5'-STEP1-FIX 全程不动)

| 红线 | 说明 |
|---|---|
| ❌ 不动 holdout | 永久封存 |
| ❌ 不升级 7 守卫包 | |
| ❌ 不动 Exp5 v2 ckpt(`/home/tcat/diffcsp_exp5/checkpoints/`)| 永久档案 |
| ❌ 不动 Exp4 backbone(`/home/tcat/diffcsp_exp4/code/diffcsp/`)| |
| ❌ 不动 shell_boundaries.pkl(errata 3 §3 干净)| |
| ❌ 不动 `_density_loss`(L 改了它的 `% 1.0` 周期归一不变)| |
| ❌ 不动 yaml(无 L_VIRTUAL hardcode,errata 3 §9 确认)| |
| ❌ 不动 CUTOFF_R / N_NEIGHBORS | errata 3 §8 排除选项 D 理由 |
| ❌ 不动 Phase 6.5 SKIPPED-by-design | |
| ❌ 不擅自调三件套 cost(1.0 / 0.5 / 0.2 沿用)| Exp5'-MA 决议,SA 不动 |
| ❌ 不擅自 cost_density(0.2 沿用)| |
| ❌ 不动 element-aware 阈值 | errata 3 §8.2 决议 |
| ❌ 不动 Exp4 共享 cache(若是共享,改 cache 路径到 exp5_prime/cache/)| |
| ❌ 不擅自启动正式训练 | STEP2 是另一棒 |
| ❌ 不擅自删 STEP1 加的功能(三件套 loss / sanity check / Phase 6.7)| 只改 L 数值 |
| ❌ 任何不确定 → 贴日志,不靠记忆 | MA 工作哲学 |

---

## §8 Watch-only 项(SA 报告即可不行动)

1. **smoke 6 loss 数值变化**(§5.3 表):报趋势对比 STEP1 旧 smoke,异常 Exp5'-MA 决议
2. **`loss_shell_dist` 在新 L=20 下的 gap 切壳行为**:proposal §2.2 用 `threshold_gap=0.1563`,这个值是 Exp4 Step 2.5 拍板的 cartesian Å 阈值,不依赖 L,**不变**。但 sorted_d 分布变了(L 改了),可能 gap 不再合理,SA 报告 epoch 0/1 shell_dist_loss 数量级,异常飙升(> 100 持续)报警
3. **磁盘趋势**:cache rebuild 后 du -sh,verify 时 65G avail 减 cache 大小后剩多少,STEP2 训练需 ~ 5-10G,SA hand-back 报告

---

## §9 OPEN QUESTIONS(SA 不答,贴给 Exp5'-MA)

### Q1 — cache 路径决议(F4)

cache 是 Exp5_prime 独立还是 Exp4 共享?SA F4 第 1 件事 grep + ls 报告,Exp5'-MA 决议。

### Q2 — Phase 6.7 数值预期变化

Phase 6.7.c collapse loss 理论值不变(§5.1),但 6.7.d spread loss 数值会变。SA 报告新数值,Exp5'-MA 看是否在合理范围。

### Q3 — `_shell_distance_loss` 在 L=20 下的稳定性

proposal §2.2 `threshold_gap=0.1563` 是 cartesian Å 阈值,sorted_d 在 L=20 下分布更宽,gap 切壳是否仍合理?SA 不改阈值,但报告 smoke 期间 shell_dist_loss / shell_count_loss 数值,Exp5'-MA 监控。

---

## §10 你不做的事(STEP1-续 / STEP2 / STEP3 才做)

- 启动正式 ~ 32-40h 训练
- 调三件套 λ
- sample 生成
- 7 项复合分评估
- 修订 Exp5' proposal §2(若三件套 loss 阈值物理意义在 L=20 下需重新论证,Exp5'-MA 写,不是 SA)
- 写 errata 4(若有新发现,Exp5'-MA 决议是否开新 errata)

---

## §11 OUTPUT.md 模板

写 `EXP5_PRIME_STEP1_FIX_HANDBACK.md` 落服务器 `/home/tcat/diffcsp_exp5_prime/` 根目录:

```
# EXP5_PRIME_STEP1_FIX_HANDBACK.md
# SA-EXP5'-STEP1-FIX 中期 hand-back

## §0 状态
- F1-F5 全部完成 / 部分完成 / 卡在 X
- cartesian sanity post-rebuild: X/100 PASS
- Phase 6.7 a-g 全 PASS
- smoke 2 epoch 完成

## §1-§6 各 PASS gate evidence
[逐 step 贴 log + 命令输出]

## §7 必报字段(详 launch note §6.1 1-7 项)

## §8 OPEN 问题(贴给 Exp5'-MA,不擅自 fix)

## §9 改动文件新 md5(STEP1 → STEP1-FIX 对比表)
| 文件 | STEP1 md5 | STEP1-FIX md5 |
|---|---|---|
| ... | ... | ... |
```

---

## §12 工作哲学红线(沿用)

1. 任何技术判断先列证据,SA 不擅自做技术判断,贴问题给 Exp5'-MA
2. 任何不确定的事 → 贴日志,不靠记忆
3. 小补丁也要贴 diff
4. 70% 上下文闸门是硬线,主动 hand-back
5. 不擅自启动训练 / 不擅自调 λ / 不擅自动 cost / 不擅自删 STEP1 功能

---

*Exp5'-MA 撰写,2026-05-02,基于 EXP4_FINAL_REPORT_ERRATA_3.md §9 surgery 清单 + §8 路径 B 决议 + §7.1 ExpN 不变量 SOP。SA-EXP5'-STEP1-FIX 接此 launch note 启动 Exp5' fold 修复一棒。*
