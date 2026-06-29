# Exp5 SA1 Patch 1 — Phase 6.5 dtype mismatch fix

## 单文件 patch:`diffusion_w_type_xas.py`

只改了 1 行(+ 4 行注释):

```python
# 改前 (Exp4 真版,Exp5 SA1 v1 沿用):
gt_atom_types_onehot = F.one_hot(
    batch.atom_types - 1, num_classes=MAX_ATOMIC_NUM).float()

# 改后 (Exp5 SA1 v1.1):
gt_atom_types_onehot = F.one_hot(
    batch.atom_types - 1, num_classes=MAX_ATOMIC_NUM).to(c0.dtype)
```

## 根因

Phase 6.5 把 model 整体 cast 成 bf16 (含 `cspnet.node_embedding.weight`),但 forward 里 `.float()` 强制把 onehot 转 fp32。结果:
- `atom_type_probs` (fp32) 进 cspnet 第一个 `F.linear` 撞上 `weight` (bf16) → mat1/mat2 dtype mismatch

`.to(c0.dtype)` 跟着 model 当前精度走:
- CPU phase 6.4: fp32 (不变,行为同 Exp4)
- GPU phase 6.5: bf16 (修复 mismatch)

## 这个 bug 在 Exp4 真版也存在

Exp4 forward_test 当时声称"5/5 PASS"——但代码同样有 `.float()`。两种可能:
- (a) Exp4 phase 6.5 当时 fail 了,没记录到 final report;或
- (b) PyTorch 不同版本对 fp32×bf16 broadcasting 容忍度变了

**不影响 Exp4 ckpt warm-start** — 这是纯 forward-time dtype cast,跟参数 shape 无关。

## 部署

```powershell
# 在 Windows PowerShell, 仅 1 个文件 scp
cd C:\Users\T-Cat\Desktop\exp5_sa1_inbox  # (或你存 patch 的目录)
scp diffusion_w_type_xas.py tcat@scsmlnprd02.its.auckland.ac.nz:/home/tcat/diffcsp_exp5/code/step3/
```

## 重跑 forward_test

```bash
ssh tcat@scsmlnprd02.its.auckland.ac.nz
cd /home/tcat/diffcsp_exp5/code/step3
EXP4_DATA_DIR=/home/tcat/diffcsp_exp5/data \
  /home/tcat/conda_envs/mlff/bin/python forward_test.py 2>&1 \
  | tee /home/tcat/diffcsp_exp5/logs/step1_forward_test_v2.log
```

预期 6/6 PASS,把 log 贴回对话 SA1 出 final OUTPUT.md。

## 不需重跑 smoke

smoke test 在 CPU fp32 跑,和这个 patch 无关 — 之前的 ALL SMOKE PASSES 仍有效。
