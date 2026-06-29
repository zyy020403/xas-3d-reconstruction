# =============================================================================
# 脚本编号: step2.2
# 脚本名称: step2.2_multisite_aggregator.py
# 输入:
#   - step2.1_spectrum_encoder.py 中的 SpectrumEncoder（作为上游输出）
# 输出:
#   - 本文件作为模块被 step2.3 和 Step 3 import
#   - 提供: MultiSiteAggregator, collate_multisite_batch()
# 说明:
#   将 N 个（可变数量）site embedding 聚合为固定维度 structure embedding。
#   架构: 单层 Self-Attention + Attention Pooling → Linear 投影
#   保证排列不变性（permutation invariance）。
#   d_struct = 256，与 DiffCSP hparams.latent_dim 槽位匹配（Step 3 在 config
#   中设 latent_dim=256，decoder 将收到 latent_dim=256+time_dim 的 t 向量）。
#   不含任何训练循环，仅模块定义 + __main__ 验证。
# =============================================================================

import os
import sys
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

# 项目根目录
PROJECT_ROOT = r"C:\Users\T-Cat\Desktop\DiffCSP-main"
EXPERIMENT_DIR = os.path.join(PROJECT_ROOT, "experiment")
STEP2_DIR = os.path.join(EXPERIMENT_DIR, "step2")
os.makedirs(STEP2_DIR, exist_ok=True)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据整理辅助函数
# ---------------------------------------------------------------------------

def collate_multisite_batch(
    site_embedding_list: List[torch.Tensor],
    quality_weight_list: Optional[List[torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    将不同化合物（不同 N）的 site embedding 列表 padding 为统一 batch。

    输入:
        site_embedding_list  : list[Tensor]，每个 shape = [n_i, d_site]，n_i 各不同
        quality_weight_list  : list[Tensor] | None，每个 shape = [n_i]，
                               元素值为 A=1.0 / B=0.5 / C=0.1；
                               若为 None，则所有权重默认 1.0

    输出:
        padded         : [batch, N_max, d_site]    padding 位置填 0
        padding_mask   : [batch, N_max]            True = padding 位置（需屏蔽）
        quality_weights: [batch, N_max] | None     padding 位置填 0.0
    """
    batch_size = len(site_embedding_list)
    assert batch_size > 0, "site_embedding_list 不能为空"

    d_site = site_embedding_list[0].shape[-1]
    n_max  = max(e.shape[0] for e in site_embedding_list)

    padded  = torch.zeros(batch_size, n_max, d_site,
                          dtype=site_embedding_list[0].dtype,
                          device=site_embedding_list[0].device)
    mask    = torch.ones(batch_size, n_max, dtype=torch.bool,
                         device=site_embedding_list[0].device)

    q_weights = None
    if quality_weight_list is not None:
        q_weights = torch.zeros(batch_size, n_max,
                                dtype=torch.float32,
                                device=site_embedding_list[0].device)

    for i, emb in enumerate(site_embedding_list):
        n_i = emb.shape[0]
        padded[i, :n_i, :] = emb
        mask[i, :n_i]      = False  # 真实位点：不屏蔽

        if quality_weight_list is not None and q_weights is not None:
            w = quality_weight_list[i]
            q_weights[i, :n_i] = w.to(dtype=torch.float32,
                                       device=site_embedding_list[0].device)

    return padded, mask, q_weights


# ---------------------------------------------------------------------------
# 多位点排列不变聚合器
# ---------------------------------------------------------------------------

class MultiSiteAggregator(nn.Module):
    """
    排列不变的多位点聚合器（Self-Attention + Attention Pooling）

    输入:
        site_embeddings : [batch, N, d_site]   N 个位点的 embedding
        padding_mask    : [batch, N]           True = padding，需屏蔽
        quality_weights : [batch, N] | None    site 级权重（可选）

    输出:
        structure_embedding : [batch, d_struct]

    架构:
        1. LayerNorm + MultiheadAttention（自注意力，4 头）+ 残差
           → 捕捉不同位点间的相互关系
        2. FFN（d_site → d_site×2 → d_site）+ LayerNorm + 残差
           → 增强单位点表达能力
        3. Attention Pooling（Linear(d_site→1) + softmax）
           → 排列不变地聚合为 [batch, d_site]
           如果提供 quality_weights，在 softmax 之前叠加到 logits 上
        4. Linear(d_site → d_struct) + ReLU + Linear(d_struct → d_struct)
           → 投影到 structure embedding 空间
    """

    def __init__(
        self,
        d_site: int   = 256,
        d_struct: int = 256,
        num_heads: int = 4,
    ):
        super().__init__()
        self.d_site   = d_site
        self.d_struct = d_struct

        # ── 自注意力层 ────────────────────────────────────────────────────────
        self.norm1 = nn.LayerNorm(d_site)
        self.attn  = nn.MultiheadAttention(
            embed_dim=d_site,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.0,
        )

        # ── FFN ───────────────────────────────────────────────────────────────
        self.norm2 = nn.LayerNorm(d_site)
        self.ffn   = nn.Sequential(
            nn.Linear(d_site, d_site * 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_site * 2, d_site),
        )

        # ── Attention Pooling ─────────────────────────────────────────────────
        self.pool_attn = nn.Linear(d_site, 1)

        # ── 投影 MLP ──────────────────────────────────────────────────────────
        self.proj = nn.Sequential(
            nn.Linear(d_site, d_struct),
            nn.ReLU(inplace=True),
            nn.Linear(d_struct, d_struct),
        )

    def forward(
        self,
        site_embeddings: torch.Tensor,                  # [B, N, d_site]
        padding_mask: torch.Tensor,                     # [B, N] bool
        quality_weights: Optional[torch.Tensor] = None, # [B, N] float | None
    ) -> torch.Tensor:                                  # [B, d_struct]

        B, N, _ = site_embeddings.shape

        # ── 1. Self-Attention（排列不变：注意力本身不依赖位置）────────────────
        x = self.norm1(site_embeddings)                 # [B, N, d_site]
        attn_out, _ = self.attn(
            x, x, x,
            key_padding_mask=padding_mask,              # True 位置被屏蔽
        )
        x = site_embeddings + attn_out                  # 残差

        # ── 2. FFN ────────────────────────────────────────────────────────────
        x2 = self.norm2(x)
        x  = x + self.ffn(x2)                          # [B, N, d_site]

        # ── 3. Attention Pooling ──────────────────────────────────────────────
        # logits: [B, N, 1]
        logits = self.pool_attn(x)                      # [B, N, 1]

        # 叠加 quality_weights（log 空间，相当于乘以先验概率）
        if quality_weights is not None:
            # 将权重转为 log 空间偏置（避免 0 权重导致 -inf，加 eps）
            log_qw = torch.log(quality_weights.unsqueeze(-1).clamp(min=1e-6))
            logits = logits + log_qw                    # [B, N, 1]

        # 对 padding 位置填 -inf，保证 softmax 后权重为 0
        logits = logits.masked_fill(padding_mask.unsqueeze(-1), float("-inf"))
        attn_weights = torch.softmax(logits, dim=1)     # [B, N, 1]

        # 加权求和 → [B, d_site]
        pooled = (attn_weights * x).sum(dim=1)          # [B, d_site]

        # ── 4. 投影 ───────────────────────────────────────────────────────────
        return self.proj(pooled)                         # [B, d_struct]


# ---------------------------------------------------------------------------
# 工具：统计模型参数量
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# 快速验证（__main__）
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    print("=" * 60)
    print("Step 2.2 快速验证")
    print("=" * 60)

    d_site, d_struct = 256, 256
    agg = MultiSiteAggregator(d_site=d_site, d_struct=d_struct, num_heads=4)
    agg.eval()

    # ── 1. 不同 N 的 forward ─────────────────────────────────────────────────
    print("\n[1] 不同位点数的 forward pass")
    for n_sites in [1, 3, 7]:
        emb_list = [torch.randn(n_sites, d_site)]
        padded, mask, qw = collate_multisite_batch(emb_list)
        with torch.no_grad():
            out = agg(padded, mask)
        print(f"  N={n_sites:2d} → structure_embedding shape: {out.shape}",
              f"  has_nan={torch.isnan(out).any().item()}")
        assert out.shape == (1, d_struct)

    # ── 2. 排列不变性验证 ────────────────────────────────────────────────────
    print("\n[2] 排列不变性验证")
    N = 5
    sites = torch.randn(1, N, d_site)
    mask  = torch.zeros(1, N, dtype=torch.bool)   # 全部有效，无 padding

    with torch.no_grad():
        emb1 = agg(sites, mask)
        perm = torch.randperm(N)
        emb2 = agg(sites[:, perm, :], mask[:, perm])

    diff = (emb1 - emb2).abs().max().item()
    print(f"  max diff after permutation: {diff:.2e}  (阈值 1e-5)")
    assert diff < 1e-5, f"排列不变性验证失败！diff={diff}"
    print("  ✓ 排列不变性验证通过")

    # ── 3. quality_weights 测试 ──────────────────────────────────────────────
    print("\n[3] quality_weights 测试")
    n_list = [3, 1, 5, 2]
    emb_list = [torch.randn(n, d_site) for n in n_list]
    # A=1.0, B=0.5, C=0.1 混合权重
    qw_list = [
        torch.tensor([1.0, 0.5, 0.1]),
        torch.tensor([1.0]),
        torch.tensor([1.0, 1.0, 0.5, 0.1, 0.5]),
        torch.tensor([0.5, 1.0]),
    ]
    padded, mask, qw = collate_multisite_batch(emb_list, qw_list)
    print(f"  padded shape : {padded.shape}")   # [4, 5, 256]
    print(f"  mask   shape : {mask.shape}")     # [4, 5]
    print(f"  qw     shape : {qw.shape}")       # [4, 5]

    with torch.no_grad():
        out = agg(padded, mask, qw)
    print(f"  structure_embedding shape: {out.shape}")  # [4, 256]
    assert out.shape == (4, d_struct)
    assert not torch.isnan(out).any()
    print("  ✓ quality_weights 测试通过")

    # ── 4. N=1 的特殊情况 ────────────────────────────────────────────────────
    print("\n[4] N=1 单位点边界情况")
    single = torch.randn(1, 1, d_site)
    single_mask = torch.zeros(1, 1, dtype=torch.bool)
    with torch.no_grad():
        out_s = agg(single, single_mask)
    assert out_s.shape == (1, d_struct)
    assert not torch.isnan(out_s).any()
    print("  ✓ N=1 通过")

    # ── 5. collate 混合 N 测试 ───────────────────────────────────────────────
    print("\n[5] collate_multisite_batch 混合 N 测试")
    mixed = [torch.randn(1, d_site), torch.randn(3, d_site),
             torch.randn(5, d_site), torch.randn(2, d_site)]
    padded2, mask2, _ = collate_multisite_batch(mixed)
    expected_n_max = 5
    print(f"  padded shape : {padded2.shape}")   # [4, 5, 256]
    print(f"  mask   shape : {mask2.shape}")     # [4, 5]
    assert padded2.shape == (4, expected_n_max, d_site)
    # 验证 mask：N=1 的样本只有第一个位点有效
    assert mask2[0, 1:].all(),  "N=1 样本的 padding mask 错误"
    assert not mask2[2, :].any(), "N=5（最大）样本不应有 padding"
    print("  ✓ collate 测试通过")

    # ── 6. 参数量 ─────────────────────────────────────────────────────────────
    n_params = count_parameters(agg)
    print(f"\n[参数量] MultiSiteAggregator: {n_params:,} 个可训练参数 "
          f"（约 {n_params/1e4:.1f} 万）")

    print("\n" + "=" * 60)
    print("Step 2.2 所有验证通过 ✓")
    print("=" * 60)