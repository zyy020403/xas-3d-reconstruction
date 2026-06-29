# MA4 → Sub-Agent 4-续 第 2 轮决策回复 + 诊断指令

> **MA4 关键修正**: 用户(最高负责人)反馈,**Exp2 Step4d 训练用的是裸 bf16,不是 bf16-mixed**。
> 这把你 §"诚实倾向" A 路径的前提推翻 — A 不是"推迟到 Trainer 解决",
> 而是"把炸弹推到 Step 4 第一个 batch"。MA4 否决 A。
> 同样推翻 B/C/D/E (B/E 用 autocast 与裸 bf16 路径不一致;C 跳 6.5 等于 A;D 污染框架)。
> 新方案 **F**:在确认根因后,做最小、可逆的修法。
> 
> **本轮你的工作**:不改任何代码,只做 2 件诊断,把事实给 MA4。MA4 看完事实再给修法指令。

---

## §1 MA4 对你 §2.X.6 汇报的认可与修正

**认可**:
- 装包阶段 7/7 守卫包零变,严格守约
- 6.5 第 0 轮 debug 就停,严格守约
- 给的 5 个方案 A-E 分析逻辑完整(虽然 A 的前提刚被推翻)
- 50% 上下文消耗,留 buffer 充足

**修正你 A 路径的前提**:

你写"Step 4 训练用 PyTorch Lightning Trainer,precision='bf16-mixed' 自动 autocast" — 
**这是错的**(我之前 handoff 也没明示,所以不全是你的责任)。用户最高负责人确认:
**Exp2 Step4d 训练用裸 bf16,model + batch 都 cast 到 bfloat16,不用 autocast/mixed。**

这意味着:
- forward_test.py 6.5 路径(`model.to(bfloat16)` + batch cast bf16)**就是 Step 4 训练真实路径**
- 6.5 暴露的 RuntimeError 不是 forward_test 边界 case,**Step 4 第一个 batch 就会撞同一个错**
- 必须现在修,不能推迟

---

## §2 你做的两件事(只读,不改代码)

### §2.1 第 1 件:看 Exp2 旧 step4d_2_train.py 怎么实际处理 bf16

```bash
# 完整看 step4d_2_train.py 的 Trainer 实例化 + model 创建那段
grep -n "Trainer\|precision\|bfloat\|bf16\|model.to\|.to(torch" \
  /home/tcat/diffcsp_exp4/code/step4d/step4d_2_train.py
```

```bash
# 看上下文 (Trainer 实例化前后 30 行,通常在文件中后部)
# 先找 Trainer 行号
grep -n "pl.Trainer\|Trainer(" /home/tcat/diffcsp_exp4/code/step4d/step4d_2_train.py | head -3
```

```bash
# 然后用上面找到的行号 N,看 N-15 到 N+30
# 假设 Trainer 在 line 200,跑:
sed -n '180,230p' /home/tcat/diffcsp_exp4/code/step4d/step4d_2_train.py
# (你按实际行号改)
```

```bash
# 同时看 datamodule / model 实例化是否在 Trainer 之前手动 .bfloat16()
grep -n "DataModule\|XASDataModule\|CSPDiffusion\|hydra\|instantiate" \
  /home/tcat/diffcsp_exp4/code/step4d/step4d_2_train.py | head -20
```

**目的**: 看 Exp2 在 train.py 里到底是
- 路径 X: `Trainer(precision='bf16')` + 不手动 cast model(让 PL Trainer 处理)
- 路径 Y: `model = model.bfloat16()` + `Trainer(precision=32)`(全裸,Trainer 不管 dtype)
- 路径 Z: `model.to(torch.bfloat16)` + Trainer 配置 bf16(混合写法)
- 路径 W: 完全别的(如全局 `torch.set_default_dtype`)

把对应代码段(20-30 行)粘给 MA4。**不要解读,粘原文**。

### §2.2 第 2 件:在 mlff env 跑 dtype 探针(不改 forward_test.py)

```bash
cd /home/tcat/diffcsp_exp4/code
PYTHONPATH=/home/tcat/diffcsp_exp4/code python <<'EOF'
import os, sys
sys.path.insert(0, '/home/tcat/diffcsp_exp4/code')
sys.path.insert(0, '/home/tcat/diffcsp_exp4/code/step3')

import torch
import hydra
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

# 1. 实例化 model 与 forward_test.py 完全相同的方式(只看,不跑 forward)
GlobalHydra.instance().clear()
initialize_config_dir(
    config_dir="/home/tcat/diffcsp_exp4/code/step3/conf_xas",
    version_base="1.1"
)
cfg = compose(config_name="model/diffusion_xas")
model = hydra.utils.instantiate(cfg, _recursive_=False)

print("=== Step A: model created (default fp32) ===")
print(f"model class: {type(model).__name__}")
# 找出所有 nn.Embedding 模块及其 weight dtype
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Embedding):
        print(f"  Embedding [{name}]: weight.dtype = {module.weight.dtype}, shape = {tuple(module.weight.shape)}")

# 2. 走 6.5 完全相同的 cast 路径: model.to(torch.bfloat16) 然后 .to('cuda:0')
print("\n=== Step B: model.to(torch.bfloat16) ===")
model_bf16 = model.to(torch.bfloat16)
for name, module in model_bf16.named_modules():
    if isinstance(module, torch.nn.Embedding):
        print(f"  Embedding [{name}]: weight.dtype = {module.weight.dtype}")

# 也看 cspnet 层的关键 Linear
print("\n=== Step B-2: 关键 Linear weight dtype ===")
for name, param in model_bf16.named_parameters():
    if 'node_embedding' in name or 'cspnet' in name.lower():
        print(f"  {name}: dtype = {param.dtype}, shape = {tuple(param.shape)}")
        # 只打前 5 个,避免刷屏
        if name.count('.') > 4: break

# 3. 用替代 API 试试
print("\n=== Step C: model.bfloat16() (替代 API) ===")
model_alt = hydra.utils.instantiate(cfg, _recursive_=False)
model_alt = model_alt.bfloat16()
for name, module in model_alt.named_modules():
    if isinstance(module, torch.nn.Embedding):
        print(f"  Embedding [{name}]: weight.dtype = {module.weight.dtype}")

# 4. 看模型有没有 register_buffer 注册的非 parameter tensor (sigma schedule 等)
print("\n=== Step D: registered buffers (非 parameter tensors) ===")
for name, buf in model_bf16.named_buffers():
    print(f"  buffer [{name}]: dtype = {buf.dtype}, shape = {tuple(buf.shape)}")

print("\n=== Probe done. ===")
EOF
```

**目的**: 验证三个假设
- **H1**: `.to(torch.bfloat16)` 是否真的转了所有 Embedding weight(Step B 输出)
- **H2**: 用 `model.bfloat16()` 替代 API 是否结果不同(Step C 输出)
- **H3**: 是否有 register_buffer 注册的 tensor 没跟随 cast(Step D 输出 dtype 不是 bfloat16 即异常)

**预期**(标准 PyTorch 行为):
- Step A 全 fp32
- Step B 全 bfloat16(如果不是 → H1 命中,model.to() 对 Embedding 不生效)
- Step C 与 Step B 应一致(如果 C 转了 B 没转 → H1 子情况确认,改用 .bfloat16())
- Step D buffer 全 bfloat16(如果有 fp32 残留 → H3 命中)

---

## §3 你的输出

把 §2.1 的代码段(grep + sed 结果)+ §2.2 的 probe 输出**完整粘贴**给 MA4。

**不要解读、不要给方案、不要 debug、不要改代码**。MA4 看完事实再给修法指令。

按下面格式:

```markdown
# Sub-Agent 4-续 第 2 轮诊断输出

## §2.1 Exp2 step4d_2_train.py 的 bf16 处理方式

### grep 输出
[原文粘贴]

### Trainer 实例化前后代码 (sed -n 'X,Yp')
[原文粘贴]

### model/datamodule 实例化方式 grep
[原文粘贴]

## §2.2 dtype probe 结果

### Step A: 默认 fp32
[输出粘贴]

### Step B: model.to(torch.bfloat16)
[输出粘贴]

### Step C: model.bfloat16()
[输出粘贴]

### Step D: registered buffers
[输出粘贴]

## 我没做的事
- 没改任何代码
- 没跑 forward_test.py
- 没尝试任何 fix
- 严格只做 2 件诊断,等 MA4 修法指令

## 上下文消耗
约 __ %
```

---

## §4 重要约束(本轮特别)

- ❌ 不要想着"我顺便测一下 model.bfloat16() 跑 6.5 是不是就过了" — 即使过了,是改路径还是改诊断脚本本身的差异不明,污染数据
- ❌ 不要 import forward_test 模块去复用其 phase_65 函数(可能改 cwd 副作用,污染 probe)
- ❌ 不要给 MA4 第三方意见(如"我建议 H1 是答案") — 你只粘事实,MA4 解读
- ✅ probe 脚本如果某一步炸(如 hydra config 找不到),停下汇报,**别 debug probe 脚本**
- 60% 上下文闸门继续生效(你 50%,probe 估 5-10K token,跑完仍 < 60%)

---

## §5 接下来发生什么

1. 你跑 §2.1 + §2.2,粘原文给 MA4
2. MA4 根据 Exp2 Step4d 实际 bf16 处理方式 + probe dtype 结果,给具体修法
   - 可能是 1 行改 forward_test.py(`.to(torch.bfloat16)` → `.bfloat16()`)
   - 或是 N 行(对齐 Exp2 train.py 的 precision 策略)
   - 或是发现框架级问题需要更大改动(罕见)
3. 你按 MA4 改法执行 + 重跑 6.5 + 按 §6 完成汇报

如果 §2.1 / §2.2 跑不动了或本身炸了,按原 SUBAGENT4CONT_HANDOFF §5 模板停汇报。

---

*MA4 撰写,2026-04-26,继续 Sub-Agent 4-续 同一窗口工作 (第 2 轮)*
