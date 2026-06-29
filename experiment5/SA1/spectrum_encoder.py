# Step 2.3 — spectrum_encoder.py
# Exp5 v2: SpectrumEncoder with MV-attention fusion + center conditioning
# =============================================================================
# Exp5 v2 改动 (vs v1 SA1):
#   1. chi 分支末端 Linear: 64*16 → 128 改为 64*16 → 256
#   2. feff 分支末端 Linear: 128 → 64 改为 128 → 256
#   3. 删除 self.fusion = nn.Sequential(Linear(448, 256), SiLU, Linear(256, 256))
#   4. 加 MV-attention 组件: mv_query / mv_attn / mv_layernorm / mv_proj
#      + mv_residual_alpha (float, NOT nn.Parameter — 固定不可学)
#   5. 保留 v1 SA1: center_emb (nn.Embedding(95, 16))
#
# 输出 (B, 272) = MV-attn fused latent (256) ⊕ center_emb (16)
#
# 在 diffusion_w_type_xas.py 中与 time_emb(256) 拼接:
#   condition(528) = cat(time_emb(256), spectrum(272))
#   → 传入 CSPNet decoder (yaml decoder.latent_dim 528)

import torch
import torch.nn as nn


class SpectrumEncoder(nn.Module):
    """
    Exp5 v2 三路 XAS 谱编码器 + MV-attention fusion + center conditioning。

    分支结构 (v2: 三 view 输出统一 256d, 平衡)
    ----------------------------------------
    XANES xmu   (B, 150)  → Conv1d-Pool-Linear → (B, 256)   view_xmu
    EXAFS chi   (B, 200)  → Conv1d-Pool-Linear → (B, 256)   view_chi   ← v2: 升 128→256
    FEFF feat   (B,  74)  → MLP                → (B, 256)   view_feff  ← v2: 升  64→256

    MV-attention fusion (v2 替换 v1 cat→MLP fusion)
    ------------------------------------------------
    views = stack([view_xmu, view_chi, view_feff], dim=1)         → (B, 3, 256)
    q     = mv_query.expand(B, -1, -1)                            → (B, 1, 256)
    attn  = MultiheadAttention(num_heads=4, batch_first=True)
                (q, K=views, V=views)                             → (B, 1, 256)
    fused = attn.squeeze(1) + alpha * views.mean(dim=1)           → (B, 256) post-residual
    fused = LayerNorm(fused)
    latent= Linear(256, 256)(fused)                               → (B, 256)

    Center conditioning (v1 SA1 carry-over)
    ---------------------------------------
    center_Z (B,) → nn.Embedding(95, 16) → center_emb (B, 16)
    output = cat([latent, center_emb], dim=-1)                    → (B, 272)

    Parameters
    ----------
    xmu_dim           : int, default 150
    chi_dim           : int, default 200
    feat_dim          : int, default 74
    latent_dim        : int, default 256       — fusion 输出维度
    n_center_elements : int, default 95        — Embedding 表大小 (max(Z)=94 + slot 0)
    center_emb_dim    : int, default 16
    mv_num_heads      : int, default 4         — MV-attention head 数 (256/4 = 64 per head)
    mv_residual_alpha : float, default 0.5     — residual 系数, 固定不可学 (NOT nn.Parameter)
    """

    def __init__(self,
                 xmu_dim=150, chi_dim=200, feat_dim=74,
                 latent_dim=256,
                 n_center_elements=95, center_emb_dim=16,
                 mv_num_heads=4, mv_residual_alpha=0.5):
        super().__init__()

        # ── E 空间: XANES 分支 (v1 结构保留, 末端 256) ──
        self.xmu_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3), nn.SiLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.SiLU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
            nn.Linear(64 * 16, 256), nn.SiLU(),
        )

        # ── k 空间: EXAFS 分支 (v2: 末端 128 → 256 ★) ──
        self.chi_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3), nn.SiLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.SiLU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
            nn.Linear(64 * 16, 256), nn.SiLU(),   # ★ v2: was Linear(64*16, 128)
        )

        # ── 物理先验: feff_features MLP 分支 (v2: 末端 64 → 256 ★) ──
        self.feat_encoder = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.SiLU(),
            nn.Linear(128, 256), nn.SiLU(),       # ★ v2: was Linear(128, 64)
        )

        # ── Exp5 v2 主线 1: MV-attention fusion 组件 ──
        # learnable query, small init (防 attention 早期偏向某一 view)
        self.mv_query = nn.Parameter(torch.randn(1, 1, latent_dim) * 0.02)

        # PyTorch 标准 MHA, batch_first=True 让 Q/K/V 形如 (B, seq, dim)
        # batch_first=True 锁死, 不可改 (handoff §5)
        self.mv_attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=mv_num_heads,
            batch_first=True,
        )

        # post-residual LayerNorm + projection
        self.mv_layernorm = nn.LayerNorm(latent_dim)
        self.mv_proj = nn.Linear(latent_dim, latent_dim)

        # 固定标量, 不可学 (NOT nn.Parameter — handoff §2 红线)
        self.mv_residual_alpha = float(mv_residual_alpha)

        # ── v1 SA1 carry-over: center-element embedding ──
        self.center_emb = nn.Embedding(n_center_elements, center_emb_dim)

        # output_dim 属性
        self._latent_out_dim = latent_dim + center_emb_dim   # 256 + 16 = 272

    @property
    def output_dim(self) -> int:
        """Final SpectrumEncoder output dim (latent + center_emb)."""
        return self._latent_out_dim

    def forward(self, xmu_xanes, chi1, feff_feats, center_Z):
        """
        Parameters
        ----------
        xmu_xanes  : Tensor (B, 150)
        chi1       : Tensor (B, 200)
        feff_feats : Tensor (B, 74)
        center_Z   : LongTensor (B,)   ← center element atomic number

        Returns
        -------
        Tensor (B, 272) = MV-attn fused latent (256) ⊕ center_emb (16)
        """
        # 三 view 各 (B, 256)
        view_xmu  = self.xmu_encoder(xmu_xanes.unsqueeze(1))   # (B, 256)
        view_chi  = self.chi_encoder(chi1.unsqueeze(1))         # (B, 256)
        view_feff = self.feat_encoder(feff_feats)               # (B, 256)

        # Stack to (B, 3, 256) for MHA
        views = torch.stack([view_xmu, view_chi, view_feff], dim=1)   # (B, 3, 256)

        # Cross-attention with learnable query
        B = views.shape[0]
        q = self.mv_query.expand(B, -1, -1)                           # (B, 1, 256)
        attn_out, _ = self.mv_attn(q, views, views, need_weights=False)  # (B, 1, 256)
        attn_out = attn_out.squeeze(1)                                # (B, 256)

        # Post-residual LayerNorm + projection
        fused = attn_out + self.mv_residual_alpha * views.mean(dim=1)  # (B, 256)
        fused = self.mv_layernorm(fused)
        latent = self.mv_proj(fused)                                   # (B, 256)

        # v1 SA1 carry-over: center embedding cat 末尾
        center_e = self.center_emb(center_Z)                           # (B, 16)
        return torch.cat([latent, center_e], dim=-1)                   # (B, 272)
