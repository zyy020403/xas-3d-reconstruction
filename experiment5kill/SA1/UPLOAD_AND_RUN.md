# Exp5 SA1 — 服务器部署 + 测试命令清单

## A. 部署:从对话下载到本地后,scp 到服务器

把对话里我 present 的 7 个改后文件下载到 Windows 一个文件夹,比如 `C:\Users\T-Cat\Desktop\exp5_sa1_inbox\`。

然后在 Windows PowerShell:

```powershell
# 0. 服务器建工作树(只跑一次,如果已经建了跳过)
ssh tcat@scsmlnprd02.its.auckland.ac.nz "
  mkdir -p /home/tcat/diffcsp_exp5/code/step2 \
           /home/tcat/diffcsp_exp5/code/step3/conf_xas/model \
           /home/tcat/diffcsp_exp5/code/step4 \
           /home/tcat/diffcsp_exp5/checkpoints \
           /home/tcat/diffcsp_exp5/logs \
           /home/tcat/diffcsp_exp5/data
"

# 0.5 数据软链接(节省 650 MB)
ssh tcat@scsmlnprd02.its.auckland.ac.nz "
  cd /home/tcat/diffcsp_exp5/data
  for f in /home/tcat/diffcsp_exp4/data/*; do
    [[ -e \"\$f\" && ! -e \"\$(basename \"\$f\")\" ]] && ln -s \"\$f\" .
  done
  ls /home/tcat/diffcsp_exp5/data | head
"

# 1. scp 7 个文件到对应位置
cd C:\Users\T-Cat\Desktop\exp5_sa1_inbox

scp spectrum_encoder.py        tcat@scsmlnprd02.its.auckland.ac.nz:/home/tcat/diffcsp_exp5/code/step2/
scp xas_local_dataset_v2.py    tcat@scsmlnprd02.its.auckland.ac.nz:/home/tcat/diffcsp_exp5/code/step3/
scp xas_local_datamodule_v2.py tcat@scsmlnprd02.its.auckland.ac.nz:/home/tcat/diffcsp_exp5/code/step3/
scp diffusion_w_type_xas.py    tcat@scsmlnprd02.its.auckland.ac.nz:/home/tcat/diffcsp_exp5/code/step3/
scp diffusion_xas.yaml         tcat@scsmlnprd02.its.auckland.ac.nz:/home/tcat/diffcsp_exp5/code/step3/conf_xas/model/
scp forward_test.py            tcat@scsmlnprd02.its.auckland.ac.nz:/home/tcat/diffcsp_exp5/code/step3/
scp step4_1_smoke_test.py      tcat@scsmlnprd02.its.auckland.ac.nz:/home/tcat/diffcsp_exp5/code/step4/
```

## B. 在服务器跑 forward_test + smoke

```bash
ssh tcat@scsmlnprd02.its.auckland.ac.nz

# (1) forward_test.py — 预计 30s
cd /home/tcat/diffcsp_exp5/code/step3
EXP4_DATA_DIR=/home/tcat/diffcsp_exp5/data \
  /home/tcat/conda_envs/mlff/bin/python forward_test.py 2>&1 \
  | tee /home/tcat/diffcsp_exp5/logs/step1_forward_test.log

# (2) step4_1_smoke_test.py — 预计 30-60s
# 注意:smoke 必须从 step3/ cwd 跑,因为它要找 conf_xas/model/diffusion_xas.yaml
cd /home/tcat/diffcsp_exp5/code/step3
EXP4_DATA_DIR=/home/tcat/diffcsp_exp5/data \
  /home/tcat/conda_envs/mlff/bin/python /home/tcat/diffcsp_exp5/code/step4/step4_1_smoke_test.py 2>&1 \
  | tee /home/tcat/diffcsp_exp5/logs/step1_smoke.log
```

## C. 把日志贴回对话

两个日志:
- `/home/tcat/diffcsp_exp5/logs/step1_forward_test.log`
- `/home/tcat/diffcsp_exp5/logs/step1_smoke.log`

复制 stdout 到对话(或 scp 下来再上传),SA1 看完会:
- 如全 PASS → 回填 OUTPUT.md §0 闸门 1+2,生成 final 版
- 如有 FAIL → 看错误栈做 patch,重新打包给你再 scp

## D. 紧急回滚(万一)

如果什么炸了,所有 .bak_exp4 baseline 锚点在服务器 `/home/tcat/diffcsp_exp4/code/...` 原位 read-only,任何时候都可:

```bash
# 完整回滚 Exp5(Exp4 read-only 不动)
rm -rf /home/tcat/diffcsp_exp5/code/step{2,3,4}/*
# 然后从 Exp4 重新 cp baseline,从头开始
```
