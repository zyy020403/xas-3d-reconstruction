"""
Exp6 SpectrumTokenizer — adapted from Exp4 spectrum_encoder.py.

ROLE CHANGE FROM EXP4:
  Exp4: 输出 (B, 256) 与 time_emb cat 成 condition(512) 喂进 CSPNet 扩散 decoder
  Exp6: 输出 (B, 256) 当 transformer encoder 的单个 input token

ONLY ARCHITECTURAL CHANGE:
  Exp4 fusion = Linear(448→256) → SiLU → Linear(256→256)  ← 末层 Linear 输出是 raw,
                                                              用于与 time_emb 混合
  Exp6 fusion = Linear(448→256) → SiLU                     ← 去末层 Linear,
                                                              输出是 post-activation
                                                              当 token 进 transformer
                                                              (transformer 内有 LN
                                                              做归一化)

Per handoff §3.4 "唯一改动: 去掉 forward 末尾的 nn.Linear(latent_dim, latent_dim)".

Notes on Exp6 proposal vs Exp4 reality (do NOT "fix" these in Phase 1+):
  - feff dim is **74** (proposal §3.1 says 73, Exp4 reality is 74).
    Verified by Step 0.6 ε dump: train_structure_cache['feff_scaled'].shape == (60507, 74).
  - feff scaling is **RobustScaler** (proposal says z-score; functionally equivalent).
    Verified by Step 0.6 ε dump: feff_scaled values pre-cached, scaler at
    /home/tcat/diffcsp_exp4/data/feff_feature_scaler.pkl
  Both inherited unchanged from Exp4 dataset/datamodule. Source: EXP4_FILE_GUIDE.md §3.1.

Input contract (matches Exp4 dataset output, Exp6 zero-modify):
  xmu_xanes  : Tensor (B, 150) float32, raw E-space XANES window
  chi1       : Tensor (B, 200) float32, k-space EXAFS chi1
  feff_feats : Tensor (B, 74)  float32, RobustScaler-scaled physical priors

Output contract:
  Tensor (B, latent_dim=256) — to be unsqueezed to (B, 1, 256) by detr_xas.py
  for use as transformer encoder input token.
"""
import torch
import torch.nn as nn


class SpectrumTokenizer(nn.Module):
    """
    Three-branch XAS spectrum encoder, producing a single 256-d token per sample.

    Branch structure (verbatim from Exp4 spectrum_encoder.py):

      xmu branch (E-space XANES):
          Conv1d(1→32, k=7) → SiLU → Conv1d(32→64, k=5) → SiLU
          → AdaptiveAvgPool1d(16) → Flatten → Linear(1024→256) → SiLU

      chi1 branch (k-space EXAFS):
          Conv1d(1→32, k=7) → SiLU → Conv1d(32→64, k=5) → SiLU
          → AdaptiveAvgPool1d(16) → Flatten → Linear(1024→128) → SiLU

      feff branch (physical-prior MLP):
          Linear(74→128) → SiLU → Linear(128→64) → SiLU

    Fusion (Exp6 modification: removed trailing Linear):
          cat(256, 128, 64) = 448 → Linear(448→256) → SiLU
    """

    def __init__(self, xmu_dim=150, chi_dim=200, feat_dim=74, latent_dim=256):
        super().__init__()

        # E-space XANES branch (verbatim from Exp4)
        self.xmu_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3), nn.SiLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.SiLU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
            nn.Linear(64 * 16, 256), nn.SiLU(),
        )

        # k-space EXAFS branch (verbatim from Exp4)
        self.chi_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3), nn.SiLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.SiLU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
            nn.Linear(64 * 16, 128), nn.SiLU(),
        )

        # Physical-prior MLP branch (verbatim from Exp4)
        self.feat_encoder = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.SiLU(),
            nn.Linear(128, 64), nn.SiLU(),
        )

        # Fusion: Exp6 modification — drop the trailing Linear(256, 256).
        # Output is post-SiLU activated, suitable as transformer input token
        # (transformer's first LayerNorm normalizes downstream).
        self.fusion = nn.Sequential(
            nn.Linear(448, latent_dim), nn.SiLU(),
        )

    def forward(self, xmu_xanes, chi1, feff_feats):
        """
        Args:
            xmu_xanes  : (B, 150)
            chi1       : (B, 200)
            feff_feats : (B, 74)
        Returns:
            (B, latent_dim=256)
        """
        xmu_out = self.xmu_encoder(xmu_xanes.unsqueeze(1))  # (B, 256)
        chi_out = self.chi_encoder(chi1.unsqueeze(1))        # (B, 128)
        feat_out = self.feat_encoder(feff_feats)             # (B, 64)

        fused = torch.cat([xmu_out, chi_out, feat_out], dim=-1)  # (B, 448)
        return self.fusion(fused)                                  # (B, 256)
