# MA4 → Sub-Agent 4-续 第 3 轮指令(收尾轮)

> **背景**:用户(最高负责人)确认 (1) 实测过 PL `precision='bf16-mixed'` 不行,
> (2) Exp2 Step4d 收敛训练就是 `pl.Trainer(precision='bf16')`,(3) 听 MA4 推荐。
> **MA4 倾向方案 G**(forward_test 6.5 改用 `pl.Trainer(precision='bf16')` 跑单 step,
> 与 Exp2 真实路径一致),但需先确认 PL 2.5.5 是否把 `'bf16'` alias 成 `'bf16-mixed'`。
> 如果是 alias,G 方案需调整;如果不是,G 方案直接安全。
> 
> **本轮工作**:1 条小诊断 + 1 段 forward_test.py 修改(我会基于诊断结果给精确 diff)。
> 总改动 < 30 行代码,你应能 1 轮收尾,不再开新窗口。

---

## §1 第 1 步:PL precision dispatch 诊断(5 秒钟,零副作用)

```bash
cd /home/tcat/diffcsp_exp4/code
python <<'EOF'
import pytorch_lightning as pl
print("PL version:", pl.__version__)

# 看 PL 2.5.5 对 'bf16' / 'bf16-mixed' / 'bf16-true' 三种字符串的内部 dispatch
# 不实例化 Trainer(避免触发 GPU/cwd 副作用),只检查 plugins/precision 模块
from pytorch_lightning.plugins.precision import (
    MixedPrecision,
    BitsandbytesPrecision,
)
import pytorch_lightning.plugins.precision as prec_module
print("\nAvailable precision plugins:")
for name in dir(prec_module):
    if 'Precision' in name and not name.startswith('_'):
        print(f"  {name}")

# 看 PL 内部 _PRECISION_INPUT 类型(如果暴露)
try:
    from pytorch_lightning.utilities.types import _PRECISION_INPUT
    print(f"\n_PRECISION_INPUT type: {_PRECISION_INPUT}")
except ImportError:
    print("\n_PRECISION_INPUT not exposed in this PL version")

# 关键测试:Trainer 接受 'bf16' / 'bf16-mixed' / 'bf16-true' 时,实际选哪个 plugin
# 用 dummy mode 创建 Trainer(不 fit 不 validate,只检查 .precision_plugin 属性)
print("\n=== Trainer precision dispatch ===")
for p in ['bf16', 'bf16-mixed', 'bf16-true']:
    try:
        t = pl.Trainer(precision=p, accelerator='cpu', logger=False, enable_checkpointing=False, enable_progress_bar=False)
        plugin = t.precision_plugin
        print(f"  precision='{p}' -> {type(plugin).__name__} (precision attr: {getattr(plugin, 'precision', '<none>')})")
        del t
    except Exception as e:
        print(f"  precision='{p}' -> RAISED {type(e).__name__}: {e}")
EOF
```

**期望输出会落入两种之一**:

**情况 X**(`'bf16'` ≡ `'bf16-mixed'`,都走 MixedPrecision/autocast):
```
precision='bf16'        -> MixedPrecision (precision attr: bf16-mixed)
precision='bf16-mixed'  -> MixedPrecision (precision attr: bf16-mixed)
precision='bf16-true'   -> HalfPrecision (or similar, precision attr: bf16-true)
```
→ 用户之前 `bf16-mixed` 实测不行,那 `bf16` 在 PL 2.5.5 也不行。**此时 Step 4 训练复用 Exp2 的 `precision='bf16'` 写法本身就有问题**(可能要 `'bf16-true'`)。这是更大的发现,需要让用户决策。

**情况 Y**(三者独立):
```
precision='bf16'        -> SomePrecision (precision attr: bf16)
precision='bf16-mixed'  -> MixedPrecision (precision attr: bf16-mixed)
precision='bf16-true'   -> HalfPrecision (precision attr: bf16-true)
```
→ Exp2 用的 `bf16` 与用户之前测试的 `bf16-mixed` 真实不同路径。G 方案直接安全。

把诊断输出原文粘给 MA4。**根据情况 X / Y,MA4 给不同的 §2 修法**:

- **情况 Y**(可能性更高,如果 Exp2 真的用 `bf16` 跑通 250 epoch): 进 §2.A
- **情况 X**: 进 §2.B(更复杂,可能要让用户决定 `'bf16-true'` 还是别的)

---

## §2.A 修法 — 情况 Y(`'bf16'` 是独立路径,Exp2 真实可用)

### §2.A.1 改 forward_test.py 6.5 段

**目标**:让 6.5 用 `pl.Trainer(precision='bf16').fit_loop` 跑一个 micro-batch,
匹配 Exp2 训练真实路径。不再用 `model.to(bfloat16)` + 手动 batch cast。

**先读现有 forward_test.py 6.5 段**:
```bash
grep -n "6.5\|phase_65\|bfloat16\|precision" /home/tcat/diffcsp_exp4/code/step3/forward_test.py
```

把 grep 输出粘给 MA4。MA4 看到行号后给精确 sed/python 替换 diff(约 20-30 行)。

**diff 设计意图**(MA4 写的,你执行):

```python
# 旧 6.5 (大致):
#   model_gpu = model.to('cuda:0').to(torch.bfloat16)
#   batch_gpu = {... cast bf16 for floats ...}
#   loss = model_gpu.training_step(batch_gpu, 0)
#   loss.backward()

# 新 6.5:
#   import pytorch_lightning as pl
#   from xas_local_datamodule_v2 import XasLocalDataModuleV2
#   
#   # 用 Exp2 同款 Trainer 配置,跑 max_steps=1 不存 checkpoint
#   trainer = pl.Trainer(
#       precision='bf16',
#       accelerator='gpu',
#       devices=1,
#       max_steps=1,
#       limit_train_batches=1,
#       limit_val_batches=0,
#       logger=False,
#       enable_checkpointing=False,
#       enable_progress_bar=False,
#       num_sanity_val_steps=0,
#   )
#   dm = XasLocalDataModuleV2(batch_size=4, num_workers=0)
#   # model 用 6.4 的同一实例(不重新实例化,因为 fp32 默认对 PL 是好的)
#   trainer.fit(model, datamodule=dm)
#   # 检查 trainer.callback_metrics['train_loss'] 或 trainer 内部状态
#   # 检查模型 weight 没有 NaN
```

**关键不变量**:
- 沿用 6.4 的 model 实例,不重新实例化
- 不显式 `model.to(bfloat16)`(让 PL Trainer 内部 autocast 处理)
- batch_size = 4(与 6.2-6.4 一致)
- max_steps=1 + limit_train_batches=1 = 跑 1 个 batch 的 forward+backward 即停
- num_sanity_val_steps=0 + limit_val_batches=0 = 不进 val
- enable_checkpointing=False = 不写 ckpt 污染目录
- logger=False = 不写 csv log

### §2.A.2 重跑 forward_test.py

```bash
cd /home/tcat/diffcsp_exp4/code/step3
PYTHONPATH=/home/tcat/diffcsp_exp4/code python forward_test.py 2>&1 | tee /home/tcat/diffcsp_exp4/logs/step3_forward_test_console_v2.log
```

**期望**: 6.1-6.4 重新 PASS(已 PASS,只是因为重跑也会再过一次)+ 6.5 这次 PASS。

**6.5 PASS 标准**(因为路径变了,标准也调整):
- trainer.fit 完成不抛异常
- 训练完成后 `trainer.callback_metrics` 含 train_loss 或类似键,值在 [1, 8]
- 模型任一参数没有 NaN/Inf(`for p in model.parameters(): assert not torch.isnan(p).any()`)
- (放宽)不再硬性比对"GPU bf16 loss vs CPU loss ±10%",因为 autocast 路径下两者不直接可比

**FAIL 处置**:
- 任何 traceback → 停汇报。**不自行 debug**(继承 §4 决策树规则)
- 模型参数有 NaN → 停汇报(可能 sigma_begin 太小、或 datamodule_v2 的 setup 问题)
- 跑成功但 callback_metrics 没有 loss 字段 → 报告但不停,看其他可观测信号(如 trainer.global_step==1)

---

## §2.B 修法骨架 — 情况 X(`'bf16'` 在 PL 2.5.5 ≡ `'bf16-mixed'`)

如果诊断显示情况 X,**Sub-Agent 4-续 不要继续**。停下汇报,因为这意味着:

- Exp2 Step4d 当年(PL 1.x 时代)的 `precision='bf16'` 与现在 mlff env 的 PL 2.5.5 行为已变
- 用户实测 `'bf16-mixed'` 不行,等于现在的 `'bf16'` 也不行
- 可能要考虑 `precision='bf16-true'`(PL 2.x 的"真 bf16")或回退到 PL 1.x

**这是 MA4 + 用户级决策**,不是 Sub-Agent 决策范围。粘 §1 诊断输出 + 这一段说明,等指令。

---

## §3 执行顺序

1. 跑 §1 诊断,粘输出
2. **等 MA4 第 4 轮指令**(基于诊断结果给精确 §2.A diff 或 §2.B 决策)
3. 收到 §2.A diff 后:执行 sed/edit + 重跑 forward_test.py + 按原 SUBAGENT4CONT_HANDOFF §6 完成汇报模板
4. 如 §2.B 命中:粘原文给 MA4,等用户 + MA4 决策

---

## §4 重要约束(继承 + 本轮特别)

继承:
- 60% 上下文闸门(你 53%,§1 估 +3K,§2.A 估 +20-30K,完成汇报 +5K = 总 ~75-80%,**会触 60% 但不到 90%**,可以继续但写完 §6 就立刻停)
- 不动 dataset_v2 / datamodule_v2 / spectrum_encoder / diffusion / yaml
- 不接触 holdout / incompat_pool / .bak 备份
- 不装新包

本轮特别:
- ✅ §1 诊断**只诊断,不解读**——把 X/Y 判定留给 MA4
- ✅ §2.A 改 forward_test.py 是 Sub-Agent 4 资产,**MA4 显式批准**(本文档),不违反"不动资产"原则
- ✅ §2.A diff 必须等 MA4 第 4 轮给,**不要看完 §1 诊断自己写 diff**
- ❌ 不要尝试 `precision='bf16-true'`(MA4 + 用户决策范围)
- ❌ 不要尝试用 `pl.Trainer(precision=16)` 之类的 fp16 路径(改方案,不是修)

---

## §5 你的两种输出路径

**路径 1**(§1 诊断完,等 MA4):

```markdown
# Sub-Agent 4-续 第 3 轮诊断输出
## §1 PL precision dispatch
[原文粘贴]

## 我的判定: 情况 X / 情况 Y / 不确定
[只标判定,不解读]

## §2.A 修法的 grep 准备(只在情况 Y 才跑)
[forward_test.py 6.5 段 grep -n 输出]

## 上下文消耗
约 __ %
```

**路径 2**(§2.A 完成 + 6.5 PASS,直接进完成汇报):

按原 SUBAGENT4CONT_HANDOFF §6 模板。**资产清单要更新**:

```
新增 / 修改:
- forward_test.py 6.5 段已改 (.bak2 备份)  ← 本轮新增

未动:
- dataset_v2 / datamodule_v2 / encoder / diffusion / yaml / .env
```

---

*MA4 撰写,2026-04-26,继续 Sub-Agent 4-续 同一窗口工作 (第 3 轮)*
