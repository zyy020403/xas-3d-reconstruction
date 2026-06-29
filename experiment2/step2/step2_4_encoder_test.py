# Step 2.4 — step2_4_encoder_test.py
# SpectrumEncoder 前向测试：shape / NaN / condition(512) 拼接验证

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from spectrum_encoder import SpectrumEncoder

if __name__ == "__main__":
    encoder = SpectrumEncoder()
    encoder.eval()

    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"   模型参数量: {total_params:,}")

    # ── 基础 shape / NaN 测试 ──────────────────────────────────
    B = 4
    xmu   = torch.randn(B, 150)
    chi1  = torch.randn(B, 200)
    feats = torch.randn(B, 73)

    with torch.no_grad():
        out = encoder(xmu, chi1, feats)

    assert out.shape == (B, 256), f"❌ shape error: {out.shape}"
    assert not out.isnan().any(), "❌ NaN in output"
    print(f"✅ 输出 shape: {out.shape}")
    print(f"✅ 无 NaN")
    print(f"   数值范围: [{out.min().item():.4f}, {out.max().item():.4f}]")

    # ── condition 拼接验证（模拟 diffusion_w_type.py 行为）──────
    time_emb  = torch.randn(B, 256)
    condition = torch.cat([time_emb, out], dim=-1)
    assert condition.shape == (B, 512), f"❌ condition shape error: {condition.shape}"
    print(f"✅ condition shape: {condition.shape}  "
          f"(time_emb(256) + spectrum(256) = 512)")

    # ── batch_size=16 压测（模拟训练时的实际 batch）─────────────
    B16 = 16
    with torch.no_grad():
        out16 = encoder(
            torch.randn(B16, 150),
            torch.randn(B16, 200),
            torch.randn(B16, 73),
        )
    assert out16.shape == (B16, 256)
    assert not out16.isnan().any()
    print(f"✅ batch_size=16 压测通过，shape: {out16.shape}")

    # ── gradient flow 检查 ─────────────────────────────────────
    encoder.train()
    xmu_g   = torch.randn(4, 150, requires_grad=True)
    chi_g   = torch.randn(4, 200, requires_grad=True)
    feats_g = torch.randn(4, 73,  requires_grad=True)
    loss = encoder(xmu_g, chi_g, feats_g).sum()
    loss.backward()
    assert xmu_g.grad is not None, "❌ xmu 梯度为空"
    assert chi_g.grad is not None, "❌ chi1 梯度为空"
    assert feats_g.grad is not None, "❌ feats 梯度为空"
    print(f"✅ 梯度反传正常（xmu / chi1 / feats 均有梯度）")

    print("\n🎉 Step 2.4 全部测试通过")
