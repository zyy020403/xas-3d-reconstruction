# Step 2.3 — spectrum_encoder.py
# SpectrumEncoder：xmu_xanes(150) + chi1(200) + feff_features(73) → (B, 256)
# ★ 纯库文件，不含执行代码，Step3 直接 import
#
# 输出 (B, 256) 将在 diffusion_w_type.py 中与 time_emb(256) 拼接：
#   condition(512) = cat(time_emb(256), spectrum(256))
#   → 传入 CSPNet（CSPNet 代码零改动）

import torch
import torch.nn as nn


class SpectrumEncoder(nn.Module):
    """
    三路 XAS 谱编码器。

    分支结构
    --------
    xmu 分支 (E空间 XANES)：
        Conv1d(1→32, k=7) → SiLU → Conv1d(32→64, k=5) → SiLU
        → AdaptiveAvgPool1d(16) → Flatten → Linear(1024→256) → SiLU

    chi1 分支 (k空间 EXAFS)：
        Conv1d(1→32, k=7) → SiLU → Conv1d(32→64, k=5) → SiLU
        → AdaptiveAvgPool1d(16) → Flatten → Linear(1024→128) → SiLU

    feff 分支 (物理先验 MLP)：
        Linear(73→128) → SiLU → Linear(128→64) → SiLU

    融合层：
        cat(256, 128, 64) = 448 → Linear(448→256) → SiLU → Linear(256→256)

    Parameters
    ----------
    xmu_dim   : int, 默认 150  — xmu_xanes 输入维度
    chi_dim   : int, 默认 200  — chi1 输入维度
    feat_dim  : int, 默认 73   — feff_features 输入维度
    latent_dim: int, 默认 256  — 输出维度（= time_emb 维度）
    """

    def __init__(self, xmu_dim=150, chi_dim=200, feat_dim=73, latent_dim=256):
        super().__init__()

        # ── E 空间：XANES 分支 ──────────────────────────────────
        self.xmu_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3), nn.SiLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.SiLU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
            nn.Linear(64 * 16, 256), nn.SiLU(),
        )

        # ── k 空间：EXAFS 分支 ──────────────────────────────────
        self.chi_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3), nn.SiLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.SiLU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
            nn.Linear(64 * 16, 128), nn.SiLU(),
        )

        # ── 物理先验：feff_features MLP 分支 ────────────────────
        self.feat_encoder = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.SiLU(),
            nn.Linear(128, 64), nn.SiLU(),
        )

        # ── 融合：256 + 128 + 64 = 448 → latent_dim ────────────
        self.fusion = nn.Sequential(
            nn.Linear(448, latent_dim), nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, xmu_xanes, chi1, feff_feats):
        """
        Parameters
        ----------
        xmu_xanes  : Tensor (B, 150)
        chi1       : Tensor (B, 200)
        feff_feats : Tensor (B, 73)

        Returns
        -------
        Tensor (B, 256)
        """
        # unsqueeze 增加 channel 维度 → (B, 1, L)
        xmu_out  = self.xmu_encoder(xmu_xanes.unsqueeze(1))   # (B, 256)
        chi_out  = self.chi_encoder(chi1.unsqueeze(1))         # (B, 128)
        feat_out = self.feat_encoder(feff_feats)               # (B, 64)

        fused = torch.cat([xmu_out, chi_out, feat_out], dim=-1)  # (B, 448)
        return self.fusion(fused)                                  # (B, 256)
