# Step 2.3 — spectrum_encoder.py
# Exp4: SpectrumEncoder：xmu_xanes(150) + chi1(200) + feff_features(74) → (B, 256)
# Exp5 SA1 patch: + center_element_Z (long) → nn.Embedding(95, 16) → concat
#                 final output (B, 272) = latent (256) ⊕ center_emb (16)
# ★ 纯库文件，不含执行代码，Step3 直接 import
#
# 输出 (B, 272) 将在 diffusion_w_type.py 中与 time_emb(256) 拼接：
#   condition(528) = cat(time_emb(256), spectrum(272))
#   → 传入 CSPNet decoder（CSPNet 代码零改动；yaml decoder.latent_dim 528）
#
# Exp5 SA1 设计选择（与 EXP5_STEP1_HANDOFF §2.1 图示一致）：
#   - center embedding 嵌入 SpectrumEncoder 内部，而非外部 cat —— 让 SpectrumEncoder
#     成为单一 (xmu, chi1, feff, center_Z) → (B, 272) 的封装单元
#   - fusion 层（Linear(448→256)）不变 → Exp4 ckpt 该层可 strict=False 完整加载
#   - 唯一新增可学参数：center_emb 表 (95×16=1520) → ckpt missing 列表预期出现 center_emb.weight

import torch
import torch.nn as nn


class SpectrumEncoder(nn.Module):
    """
    三路 XAS 谱编码器 + Exp5 SA1 center-element 条件嵌入。

    分支结构
    --------
    xmu 分支 (E空间 XANES)：
        Conv1d(1→32, k=7) → SiLU → Conv1d(32→64, k=5) → SiLU
        → AdaptiveAvgPool1d(16) → Flatten → Linear(1024→256) → SiLU

    chi1 分支 (k空间 EXAFS)：
        Conv1d(1→32, k=7) → SiLU → Conv1d(32→64, k=5) → SiLU
        → AdaptiveAvgPool1d(16) → Flatten → Linear(1024→128) → SiLU

    feff 分支 (物理先验 MLP)：
        Linear(74→128) → SiLU → Linear(128→64) → SiLU

    融合层：
        cat(256, 128, 64) = 448 → Linear(448→256) → SiLU → Linear(256→256)

    Exp5 SA1 center embedding（新增）：
        center_Z(LongTensor B,) → nn.Embedding(95, 16) → (B, 16)

    最终拼接：
        cat(fused_latent (B,256), center_emb (B,16)) → (B, 272)

    Parameters
    ----------
    xmu_dim           : int, 默认 150  — xmu_xanes 输入维度
    chi_dim           : int, 默认 200  — chi1 输入维度
    feat_dim          : int, 默认 74   — feff_features 输入维度
    latent_dim        : int, 默认 256  — fusion 输出维度（= time_emb 维度）
    n_center_elements : int, 默认 95   — Embedding 表大小（max(Z)=94, slot 0 padding）
    center_emb_dim    : int, 默认 16   — center embedding 维度
    """

    def __init__(self, xmu_dim=150, chi_dim=200, feat_dim=74, latent_dim=256,
                 n_center_elements=95, center_emb_dim=16):
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
        # (Exp4 ckpt 该层可 strict=False 完整加载，shape 不变)
        self.fusion = nn.Sequential(
            nn.Linear(448, latent_dim), nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

        # ── Exp5 SA1: center-element embedding ──────────────────
        # max(Z)=94 empirically (Pu); slot 0 reserved for padding.
        # Output (B, 16) concat with fused (B, 256) → final (B, 272).
        # Random init at warm-start; SA2 phased training (Notes §8.1) handles this.
        self.center_emb = nn.Embedding(n_center_elements, center_emb_dim)
        self._latent_out_dim = latent_dim + center_emb_dim   # 256 + 16 = 272

    @property
    def output_dim(self) -> int:
        """Final SpectrumEncoder output dim (= latent_dim + center_emb_dim)."""
        return self._latent_out_dim

    def forward(self, xmu_xanes, chi1, feff_feats, center_Z):
        """
        Parameters
        ----------
        xmu_xanes  : Tensor (B, 150)
        chi1       : Tensor (B, 200)
        feff_feats : Tensor (B, 74)
        center_Z   : LongTensor (B,)   ← Exp5 SA1, atomic number of center element

        Returns
        -------
        Tensor (B, 272)  =  fused_latent (256) ⊕ center_emb (16)
        """
        # unsqueeze 增加 channel 维度 → (B, 1, L)
        xmu_out  = self.xmu_encoder(xmu_xanes.unsqueeze(1))   # (B, 256)
        chi_out  = self.chi_encoder(chi1.unsqueeze(1))         # (B, 128)
        feat_out = self.feat_encoder(feff_feats)               # (B, 64)

        fused = torch.cat([xmu_out, chi_out, feat_out], dim=-1)  # (B, 448)
        latent = self.fusion(fused)                              # (B, 256)

        # ── Exp5 SA1: center-element conditioning ──
        center_e = self.center_emb(center_Z)                     # (B, 16)
        return torch.cat([latent, center_e], dim=-1)             # (B, 272)
