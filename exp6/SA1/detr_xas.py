"""
Exp6 main model: DETR-style set prediction for XAS → atomic structure.

Per proposal §4 architecture + handoff §3.7 forward decisions.

Components (all in experiment6/shared/):
  - SpectrumTokenizer (spectrum_tokenizer.py): xmu+chi1+feff → 256d token
  - center_token_embed (here): center element Z → 256d learnable token
  - token_pos_embed (here): 1D learned pos embed for 2 encoder tokens
  - Transformer (transformer.py): DETR encoder-decoder, seq-first modified
  - 20 object queries: nn.Embedding(20, 256)
  - class_head: 3-layer MLP → (n_neighbor_types + 1) logits
  - pos_head: 3-layer MLP → 3, then tanh*0.5 → frac coords [-0.5, 0.5]

Forward inputs (batch dict, fields produced by Exp4 dataset/datamodule + a
caller-side mapper from atomic Z to dense vocab idx):
  xmu        : (B, 150)
  chi1       : (B, 200)
  feff       : (B, 74)
  center_idx : (B,) int64 — already mapped to dense [0, N_CENTER_TYPES)

Forward outputs (dict matching DETR convention):
  pred_logits : (B, 20, N_NEIGHBOR_TYPES + 1)
  pred_pos    : (B, 20, 3) in [-0.5, 0.5]
  aux_outputs : list[5] of {'pred_logits':..., 'pred_pos':...}  (when aux_loss=True)

Reference: handoff §3.7, §4 delta map, §5 tensor shape contracts.
"""
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import Transformer
from .spectrum_tokenizer import SpectrumTokenizer


class MLP(nn.Module):
    """
    Simple multi-layer perceptron (DETR-style).

    Verbatim copy from DETR detr.py MLP class — see handoff §4 delta map note
    "MLP 单独在 detr_xas.py 里写自己的". 3 layers default (proposal §4 "各 ~3 层").
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class DETRXas(nn.Module):
    """
    DETR-style XAS → atomic neighborhood predictor.

    Args:
        vocab_path:           path to exp6_element_vocab.json (Step 1.0 output).
                              Class counts (N_NEIGHBOR_TYPES, N_CENTER_TYPES) and
                              no_object_idx are loaded from here.
        d_model:              transformer hidden dim. Default 256 (proposal §6.1).
        nhead:                attention heads. Default 8.
        num_encoder_layers:   default 6 (proposal §6.1).
        num_decoder_layers:   default 6.
        dim_feedforward:      default 2048 (DETR convention).
        dropout:              default 0.1 (DETR).
        num_queries:          default 20 (proposal §4.1(b), matches Exp4 max neighbors).
        lengths:              box edges Å, default (6,6,6) per proposal §4.1(d).
        aux_loss:             enable auxiliary decoding loss (DETR convention,
                              proposal §6.1 "auxiliary_loss 启用,沿用").
    """

    def __init__(
        self,
        vocab_path: str,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        num_queries: int = 20,
        lengths=(6.0, 6.0, 6.0),
        aux_loss: bool = True,
    ):
        super().__init__()

        # ------ Load vocab ------
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)

        self.n_neighbor_types = int(vocab['neighbor']['N_TYPES'])
        self.n_center_types = int(vocab['center']['N_TYPES'])
        self.no_object_idx = int(vocab['neighbor']['no_object_idx'])
        # JSON serializes int keys as strings; cast back.
        self.center_Z_to_idx = {
            int(k): int(v) for k, v in vocab['center']['Z_to_idx'].items()
        }
        self.neighbor_Z_to_idx = {
            int(k): int(v) for k, v in vocab['neighbor']['Z_to_idx'].items()
        }

        assert self.no_object_idx == self.n_neighbor_types, (
            f"Vocab inconsistency: no_object_idx={self.no_object_idx} != "
            f"n_neighbor_types={self.n_neighbor_types}"
        )

        # ------ Hyperparameters ------
        self.num_queries = num_queries
        self.aux_loss = aux_loss
        self.d_model = d_model

        # ------ Spectrum tokenizer (Exp4 frontend, fusion last layer dropped) ------
        self.tokenizer = SpectrumTokenizer(
            xmu_dim=150, chi_dim=200, feat_dim=74, latent_dim=d_model,
        )

        # ------ Center element token (proposal §3.2 "separate learnable token") ------
        self.center_token_embed = nn.Embedding(self.n_center_types, d_model)

        # ------ 1D learned pos embedding for 2 encoder tokens (spectrum + center)
        # Phase 1 single spectrum: 2 tokens. Phase 2 multi-spectrum: extend to N+1.
        # Replaces DETR's 2D image sin pos (proposal §2.3 decision).
        self.token_pos_embed = nn.Embedding(2, d_model)

        # ------ Transformer (DETR encoder-decoder, seq-first modified) ------
        self.transformer = Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            return_intermediate_dec=True,  # for aux losses
        )

        # ------ Object queries (DETR convention) ------
        self.query_embed = nn.Embedding(num_queries, d_model)

        # ------ Output heads ------
        # Class head: 3-layer MLP → (N_NEIGHBOR_TYPES + 1) logits
        self.class_head = MLP(
            input_dim=d_model, hidden_dim=d_model,
            output_dim=self.n_neighbor_types + 1, num_layers=3,
        )
        # Pos head: 3-layer MLP → 3, then tanh*0.5 in forward (proposal §4.1(d))
        self.pos_head = MLP(
            input_dim=d_model, hidden_dim=d_model,
            output_dim=3, num_layers=3,
        )

        # ------ lengths buffer (Å) for downstream loss/eval ------
        # Stored here so caller can self.lengths.to(device) automatically.
        self.register_buffer('lengths', torch.tensor(lengths, dtype=torch.float32))

    def forward(self, batch):
        """
        Args:
            batch: dict with keys
                xmu        : (B, 150) float
                chi1       : (B, 200) float
                feff       : (B, 74)  float (RobustScaler-normalized via Exp4 dataset)
                center_idx : (B,) int64 — caller pre-maps Z → dense center vocab idx

        Returns:
            dict:
                pred_logits : (B, num_queries=20, n_neighbor_types + 1) — last decoder layer
                pred_pos    : (B, num_queries=20, 3) frac coords in [-0.5, 0.5]
                aux_outputs : list[num_decoder_layers - 1] of {'pred_logits':..., 'pred_pos':...}
                              (only present when self.aux_loss is True)

        Tensor shape contract per handoff §5.
        """
        xmu = batch['xmu']
        chi1 = batch['chi1']
        feff = batch['feff']
        center_idx = batch['center_idx']
        B = xmu.shape[0]

        # 1. Encode spectrum to single 256d vector → unsqueeze to token form
        spectrum_vec = self.tokenizer(xmu, chi1, feff)        # (B, 256)
        spectrum_token = spectrum_vec.unsqueeze(1)             # (B, 1, 256)

        # 2. Center element token (separate learnable, proposal §3.2)
        center_vec = self.center_token_embed(center_idx)       # (B, 256)
        center_token = center_vec.unsqueeze(1)                  # (B, 1, 256)

        # 3. Concatenate to 2-token sequence
        src_bf = torch.cat([spectrum_token, center_token], dim=1)  # (B, 2, 256)

        # 4. Permute to seq-first (2, B, 256) for our modified transformer
        src = src_bf.permute(1, 0, 2)                          # (2, B, 256)

        # 5. Position embedding: (2, 256) → (2, B, 256) via expand
        pos = self.token_pos_embed.weight.unsqueeze(1).expand(-1, B, -1)

        # 6. Object queries: (num_queries, 256) → (num_queries, B, 256)
        query = self.query_embed.weight.unsqueeze(1).expand(-1, B, -1)

        # 7. Transformer — Phase 1 single spectrum, no padding mask
        # Returns (num_layers, B, num_queries, d_model)
        hs = self.transformer(src, None, query, pos)

        # 8. Apply heads to ALL decoder layers (for aux loss)
        outputs_class = self.class_head(hs)                    # (L, B, Q, K+1)
        outputs_pos_raw = self.pos_head(hs)                    # (L, B, Q, 3)
        outputs_pos = torch.tanh(outputs_pos_raw) * 0.5         # constrain to [-0.5, 0.5]

        # 9. Output dict — last layer is primary, others are aux
        out = {
            'pred_logits': outputs_class[-1],   # (B, Q, K+1)
            'pred_pos': outputs_pos[-1],        # (B, Q, 3)
        }
        if self.aux_loss:
            out['aux_outputs'] = [
                {'pred_logits': outputs_class[i], 'pred_pos': outputs_pos[i]}
                for i in range(outputs_class.shape[0] - 1)
            ]

        return out
