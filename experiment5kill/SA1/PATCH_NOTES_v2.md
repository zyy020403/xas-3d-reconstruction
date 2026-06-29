# Exp5 SA1 Patch 2 — Phase 6.5 SKIPPED-by-design + final OUTPUT.md

## 内容

1. **`forward_test.py`**:phase 6.5 改成 skip,原代码保留为 `_phase_65_legacy`
2. **`EXP5_STEP1_OUTPUT.md`**:final 版,含 §5.6 PYTHONPATH 警告 + §5.7 phase 6.5 skip 决策

## 部署

```powershell
# 仅 1 个 .py 文件需要 scp(OUTPUT.md 是文档,无需上服务器)
cd C:\Users\T-Cat\Desktop\exp5_sa1_inbox
scp forward_test.py tcat@scsmlnprd02.its.auckland.ac.nz:/home/tcat/diffcsp_exp5/code/step3/
```

## 重跑 forward_test(预计 30s,这次会到底)

```bash
ssh tcat@scsmlnprd02.its.auckland.ac.nz
cd /home/tcat/diffcsp_exp5/code/step3
EXP4_DATA_DIR=/home/tcat/diffcsp_exp5/data \
  /home/tcat/conda_envs/mlff/bin/python forward_test.py 2>&1 \
  | tee /home/tcat/diffcsp_exp5/logs/step1_forward_test_v3.log
```

## 预期输出

```
Phase 6.1 PASS
Phase 6.2 PASS
Phase 6.3 PASS
Phase 6.4 PASS
[Phase 6.5] SKIPPED (by design): ...
Phase 6.6 PASS
========================================================================
5/5 PHASES PASS  +  1 SKIPPED-BY-DESIGN (phase 6.5)
  Phases run: 6.1 / 6.2 / 6.3 / 6.4 / 6.6   ALL PASS
  Phase 6.5 (GPU bf16): SKIPPED — see OUTPUT.md §5.7
total wall time: ~30 s
Step 1 launch gate: CLEAR (Exp5 SA1 architecture verified, fp32 production path)
========================================================================
```

把 log 贴回对话,SA1 在 OUTPUT.md §6.1 phase 6.6 那一栏回填实测数据 → final-final 版。
然后 SA2 可启动。

## smoke test 不用重跑

CPU fp32 path,不受 phase 6.5 改动影响,v1 ALL SMOKE PASSES 仍有效。
