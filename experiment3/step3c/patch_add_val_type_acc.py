"""
patch_add_val_type_acc.py
=========================
为 diffusion_w_type_xas_exp3.py 的 validation_step() 添加 val_type_acc 计算。

使用方式（在 experiment3/step3c/ 目录下运行，或传入模型文件路径）：
  python patch_add_val_type_acc.py
  python patch_add_val_type_acc.py --path "C:\\Users\\T-Cat\\Desktop\\DiffCSP-main\\experiment3\\step3c\\diffusion_w_type_xas_exp3.py"

补丁内容：
  将原来的 validation_step（仅记录 loss 指标）替换为：
  1. 调用 self(batch) + compute_stats 记录原有指标
  2. 额外计算 val_type_acc（latent → TypeClassifier → argmax → 原子序数 → 与 atom_types 比较）
  3. 用 self.log('val_type_acc', acc, on_step=False, on_epoch=True) 记录
"""

import argparse
import os
import shutil

# ── 默认目标文件路径 ────────────────────────────────────────────────────────────
DEFAULT_PATH = (
    r"C:\Users\T-Cat\Desktop\DiffCSP-main"
    r"\experiment3\step3c\diffusion_w_type_xas_exp3.py"
)

# ── 要被替换的原始 validation_step ──────────────────────────────────────────────
OLD_VALIDATION_STEP = '''\
    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        output_dict = self(batch)
        log_dict, loss = self.compute_stats(output_dict, prefix='val')
        self.log_dict(log_dict, on_step=False, on_epoch=True, prog_bar=True)
        return loss'''

# ── 替换后的新 validation_step（含 val_type_acc）───────────────────────────────
NEW_VALIDATION_STEP = '''\
    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        output_dict = self(batch)
        log_dict, loss = self.compute_stats(output_dict, prefix=\'val\')

        # ── Exp3 新增：计算 val_type_acc ───────────────────────────────────
        # latent → TypeClassifier → argmax → 原子序数 → 与 batch.atom_types 比较
        with torch.no_grad():
            spectrum_cond = self.spectrum_encoder(
                batch.xmu_xanes,
                batch.chi1,
                batch.feff_features,
            )
            type_logits = self.type_classifier(spectrum_cond)  # (B, 20, N_elem)
            pred_class  = type_logits.argmax(dim=-1)            # (B, 20)

            # 构建真实 class_index 标签（与 forward() 中保持相同逻辑）
            _types_flat   = batch.atom_types                    # (N_total,) 原子序数
            _labels_list  = torch.split(_types_flat, batch.num_atoms.tolist())
            _labels       = torch.stack([t for t in _labels_list])  # (B, 20)

            _ci = torch.zeros_like(_labels)
            for b in range(_labels.shape[0]):
                for n in range(_labels.shape[1]):
                    z_str      = str(_labels[b, n].item())
                    _ci[b, n]  = self.elem_vocab.get(z_str, 0)  # OOV 归 class 0

            acc = (pred_class == _ci.to(pred_class.device)).float().mean()

        log_dict[\'val_type_acc\'] = acc
        self.log_dict(log_dict, on_step=False, on_epoch=True, prog_bar=True)
        return loss'''


def apply_patch(target_path: str):
    print(f"目标文件：{target_path}")

    if not os.path.isfile(target_path):
        raise FileNotFoundError(f"找不到目标文件：{target_path}")

    with open(target_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 检查是否已经打过补丁
    if "val_type_acc" in content:
        print("⚠️  文件中已含 'val_type_acc'，补丁可能已经应用过。跳过修改。")
        return

    # 检查原始字符串是否存在
    if OLD_VALIDATION_STEP not in content:
        print("❌  未找到原始 validation_step 代码片段。")
        print("    可能是文件格式不匹配（缩进/换行符差异）。")
        print("    请手动替换 validation_step，参考下方新代码：")
        print()
        print(NEW_VALIDATION_STEP)
        return

    # 备份原文件
    backup_path = target_path + ".bak"
    shutil.copy2(target_path, backup_path)
    print(f"  备份原文件 → {backup_path}")

    # 应用替换
    new_content = content.replace(OLD_VALIDATION_STEP, NEW_VALIDATION_STEP, 1)

    with open(target_path, "w", encoding="utf-8") as f:
        f.write(new_content)

    # 验证补丁已写入
    with open(target_path, "r", encoding="utf-8") as f:
        verify = f.read()
    assert "val_type_acc" in verify, "补丁写入后验证失败！"

    print("✅  补丁应用成功：validation_step 已加入 val_type_acc 计算。")
    print()
    print("=== 补丁摘要 ===")
    print("  修改位置 : CSPDiffusion.validation_step()")
    print("  新增内容 : spectrum_encoder → type_classifier → argmax → accuracy")
    print("  日志 key : 'val_type_acc'  (on_step=False, on_epoch=True)")
    print("  OOV 处理 : 不在词表内的原子序数归入 class 0（不影响主要精度）")


def verify_patch(target_path: str):
    """可选验证：用伪数据跑一次修改后的 validation_step 逻辑（无需 GPU）"""
    import sys
    import torch
    import torch.nn.functional as F

    print()
    print("=== 逻辑验证（CPU 上的简化版本）===")
    # 模拟 elem_vocab
    vocab   = {str(z): i for i, z in enumerate([8, 26, 14, 13, 22])}
    n_elem  = len(vocab)
    B, N    = 3, 20

    # 模拟 spectrum_cond → type_logits
    latent      = torch.randn(B, 256)
    fc          = torch.nn.Linear(256, N * n_elem)
    type_logits = fc(latent).view(B, N, n_elem)
    pred_class  = type_logits.argmax(dim=-1)   # (B, 20)

    # 模拟 batch.atom_types（原子序数）
    atom_types  = torch.tensor([8, 26, 14, 13, 22] * (B * N // 5 + 1))[:B * N]
    labels_list = list(atom_types.split([N] * B))
    labels      = torch.stack(labels_list)

    ci = torch.zeros_like(labels)
    for b in range(labels.shape[0]):
        for n in range(labels.shape[1]):
            z_str    = str(labels[b, n].item())
            ci[b, n] = vocab.get(z_str, 0)

    acc = (pred_class == ci).float().mean()
    assert not torch.isnan(acc), "acc is NaN!"
    print(f"  ✅ val_type_acc 计算通过：acc={acc.item():.4f}（随机权重）")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="为 diffusion_w_type_xas_exp3.py 添加 val_type_acc 补丁")
    parser.add_argument(
        "--path", default=DEFAULT_PATH,
        help="目标模型文件的完整路径")
    parser.add_argument(
        "--verify-only", action="store_true",
        help="仅运行逻辑验证，不修改文件")
    args = parser.parse_args()

    if args.verify_only:
        verify_patch(args.path)
    else:
        apply_patch(args.path)
        verify_patch(args.path)
