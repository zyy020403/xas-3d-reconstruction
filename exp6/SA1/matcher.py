# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Hungarian matcher for Exp6, adapted from DETR matcher.py (commit 29901c5).

Per handoff §3.5 + §4 delta map:
  - Removed: util.box_ops imports (no GIoU, no box-format conversion)
  - Removed: cost_giou (no IoU concept for atomic positions)
  - Renamed: cost_bbox → cost_pos
  - Renamed: outputs["pred_boxes"] → outputs["pred_pos"]
  - Replaced: torch.cdist(p=1) (L1 box) → min_image_l2 (periodic-aware L2)
  - Added: lengths argument (passes box edge lengths in Å for min-image)

Preserved (per handoff §3.5 "绝对保留"):
  - @torch.no_grad() decorator on forward
  - linear_sum_assignment per-batch via .cpu().numpy()
  - Return format: list of (LongTensor i, LongTensor j) per batch element
  - batch-flatten optimization (out_prob = pred_logits.flatten(0, 1))

Target dict format (Exp6, NOT DETR's box format):
  targets[b]: {
    'labels': LongTensor (n_b,)  — neighbor element class indices in [0, N_NEIGHBOR_TYPES)
    'pos':    Tensor (n_b, 3)    — neighbor frac coords in [-0.5, 0.5]
  }
where n_b is number of real (non-padding) neighbors for sample b.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

# Reuse the broadcast-friendly min_image_l2 from eval_metrics
# (same module dir, sibling import in detr_xas.py via `from .eval_metrics import min_image_l2`)
from .eval_metrics import min_image_l2


class HungarianMatcher(nn.Module):
    """
    Computes 1-to-1 assignment between predictions and ground-truth atoms.

    For Exp6, predictions are 20 queries each producing (logits over
    N_NEIGHBOR_TYPES + 1 classes, 3D position). Ground-truth has variable
    n ≤ 20 real atoms. Predictions exceeding ground-truth count are
    un-matched (treated as no-object via the +1 class index).
    """

    def __init__(self, cost_class: float = 1.0, cost_pos: float = 5.0,
                 lengths=(6.0, 6.0, 6.0)):
        """
        Args:
            cost_class: weight on classification cost in the matching objective
            cost_pos:   weight on min-image-L2 position cost
            lengths:    box edge lengths in Å for min-image folding (default L=6)

        Per proposal §6.1 + §5: cost_class=1.0, cost_pos=5.0 starting values
        (DETR-style; SA2 may need to tune cost_pos given L2 vs DETR's L1 difference,
        see proposal §5 caveat).
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_pos = cost_pos
        # Register lengths as buffer so it follows model device (cuda/cpu)
        self.register_buffer('lengths', torch.tensor(lengths, dtype=torch.float32))

        assert cost_class != 0 or cost_pos != 0, \
            "Both cost_class and cost_pos can't be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Args:
            outputs: dict containing
                - "pred_logits": (B, num_queries, N_NEIGHBOR_TYPES+1) classification logits
                - "pred_pos":    (B, num_queries, 3) frac coords in [-0.5, 0.5]
            targets: list of length B, each dict with
                - "labels": (n_b,) int64, class indices in [0, N_NEIGHBOR_TYPES)
                - "pos":    (n_b, 3) float32, frac coords

        Returns:
            list of size B; each element is (i, j) where
              i: LongTensor — selected prediction indices (in order)
              j: LongTensor — corresponding selected target indices (in order)
            len(i) == len(j) == min(num_queries, n_b)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # Batch-flatten layout for efficient cost computation
        # out_prob: (B*Q, N_NEIGHBOR_TYPES+1)
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
        # out_pos: (B*Q, 3)
        out_pos = outputs["pred_pos"].flatten(0, 1)

        # Concatenate target labels and positions across batch
        # tgt_ids: (sum_n,) int
        # tgt_pos: (sum_n, 3) float
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_pos = torch.cat([v["pos"] for v in targets])

        # Classification cost: -prob[gt class] (proposal §5 + DETR original)
        # cost_class shape: (B*Q, sum_n)
        cost_class = -out_prob[:, tgt_ids]

        # Position cost: min-image L2 in Cartesian Å.
        # min_image_l2 supports broadcast: (B*Q, 3) and (sum_n, 3) → (B*Q, sum_n)
        # Verified equivalent to per-sample (Q, n) layout (eval_metrics.py docstring).
        cost_pos = min_image_l2(out_pos, tgt_pos, self.lengths)

        # Final cost matrix (B*Q, sum_n) → reshape to (B, Q, sum_n)
        C = self.cost_pos * cost_pos + self.cost_class * cost_class
        C = C.view(bs, num_queries, -1).cpu()

        # Per-sample Hungarian solve via scipy
        sizes = [len(v["pos"]) for v in targets]
        indices = [
            linear_sum_assignment(c[i])
            for i, c in enumerate(C.split(sizes, -1))
        ]

        return [
            (torch.as_tensor(i, dtype=torch.int64),
             torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]
