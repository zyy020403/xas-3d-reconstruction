# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
SetCriterion for Exp6, adapted from DETR detr.py (commit 29901c5).

Per handoff §3.6 + §4 delta map:

REMOVED:
  - util.box_ops imports (no GIoU)
  - loss_masks (entire method, no segmentation)
  - PostProcess class, DETR class, MLP class, build() — only SetCriterion kept
  - GIoU computation in loss_pos
  - 'masks' from losses list

REPLACED:
  - loss_boxes → loss_pos
    * 'pred_boxes' → 'pred_pos'
    * 'boxes' (target) → 'pos' (target)
    * F.l1_loss(L1) + GIoU → squared min-image L2 / num_boxes
  - 'boxes' / 'masks' in losses list → 'pos'

PRESERVED (per handoff §3.6 "绝对保留"):
  - loss_labels: CE with empty_weight, empty_weight[-1] = eos_coef = 0.1
  - loss_cardinality: @torch.no_grad() diagnostic (NOT a real loss)
  - _get_src_permutation_idx, _get_tgt_permutation_idx
  - forward() main flow including aux_outputs loop
  - num_classes naming convention: "omit no_object", index N_NEIGHBOR_TYPES = no_object slot

NOTE on cardinality vs no_object_ratio (handoff §3.6 footnote):
  loss_cardinality (DETR built-in diagnostic) and no_object_ratio (Exp6 custom,
  proposal §附录B.5) coexist. Both are logged separately, no merging attempted.
  cardinality returns L1(card_pred - target_count); no_object_ratio is the
  fraction of queries predicting NO_OBJECT_IDX. Different lenses on the same
  health signal.
"""
import torch
import torch.nn.functional as F
from torch import nn


# ---------------------------------------------------------------------------
# Helpers (replaces util.misc imports we don't want to depend on)
# ---------------------------------------------------------------------------
def is_dist_avail_and_initialized():
    """Minimal substitute for util.misc.is_dist_avail_and_initialized."""
    if not torch.distributed.is_available():
        return False
    if not torch.distributed.is_initialized():
        return False
    return True


def get_world_size():
    """Minimal substitute for util.misc.get_world_size."""
    if not is_dist_avail_and_initialized():
        return 1
    return torch.distributed.get_world_size()


def accuracy(output, target, topk=(1,)):
    """
    Top-k accuracy. Adapted from util.misc.accuracy (DETR).
    Used only for logging in loss_labels (class_error metric).
    """
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def _min_image_l2_sq_paired(src, tgt, lengths):
    """
    Squared min-image L2 distance, pair-wise (NOT all-pairs broadcast).

    Args:
        src: (N, 3) frac coords
        tgt: (N, 3) frac coords (paired with src element-wise)
        lengths: (3,) box edges Å

    Returns:
        (N,) squared distance in Å²

    Used in loss_pos. This complements eval_metrics.min_image_l2 (all-pairs version):
    eval/matcher do all-pairs (M, 3) × (N, 3) → (M, N); criterion's loss_pos
    works on already-matched pairs (N, 3) × (N, 3) → (N,).
    """
    diff = src - tgt
    diff = diff - torch.round(diff)
    cart = diff * lengths
    return (cart ** 2).sum(dim=-1)


# ---------------------------------------------------------------------------
# SetCriterion — main class (adapted from detr.py SetCriterion, ~lines 90-260)
# ---------------------------------------------------------------------------
class SetCriterion(nn.Module):
    """
    Computes the loss for DETR-style Exp6 set prediction.

    Process:
      1) Hungarian matching between ground-truth atoms and 20 query predictions
      2) Supervise each matched pair: classification (CE) + position (L2 squared)
      3) Unmatched queries: target class is no_object (index N_NEIGHBOR_TYPES),
         no position loss

    Args:
        num_classes: number of element categories, **omitting** no_object.
                     Must equal N_NEIGHBOR_TYPES (= 89 from Step 1.0).
                     no_object will live at index `num_classes`.
        matcher: HungarianMatcher instance
        weight_dict: {'loss_ce': λ_cls, 'loss_pos': λ_pos, ...}
                     (aux losses get auto-suffixed _0, _1, ... by forward())
        eos_coef: weight on no_object class in CE (DETR default 0.1)
        losses: list of loss names to compute. Exp6 uses ['labels', 'pos', 'cardinality'].
        lengths: (3,) box edge lengths in Å for min-image folding.
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 lengths=(6.0, 6.0, 6.0)):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses

        # empty_weight: shape (num_classes + 1,)
        # Last index = no_object slot, weighted by eos_coef
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

        # lengths for min-image L2 in loss_pos
        self.register_buffer('lengths', torch.tensor(lengths, dtype=torch.float32))

    # -----------------------------------------------------------------------
    # loss_labels — VERBATIM from DETR (no Exp6-specific change)
    # -----------------------------------------------------------------------
    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """
        Classification loss (NLL via CE with empty_weight).

        targets[b]['labels']: (n_b,) int64 in [0, num_classes).
        Unmatched queries get filled with self.num_classes (= no_object).
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes,
            dtype=torch.int64, device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2), target_classes, self.empty_weight
        )
        losses = {'loss_ce': loss_ce}

        if log:
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    # -----------------------------------------------------------------------
    # loss_cardinality — VERBATIM from DETR (diagnostic, no gradient)
    # -----------------------------------------------------------------------
    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """
        Diagnostic: |# predicted non-empty - # GT|. Not a real loss.

        Coexists with no_object_ratio (proposal §附录B.5); both logged.
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor(
            [len(v["labels"]) for v in targets], device=device
        )
        # num_classes index is the no_object slot.
        # Count predictions that are NOT no_object.
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    # -----------------------------------------------------------------------
    # loss_pos — Exp6 REPLACEMENT for DETR's loss_boxes
    # -----------------------------------------------------------------------
    def loss_pos(self, outputs, targets, indices, num_boxes):
        """
        Position regression loss: squared min-image L2 in Cartesian Å, summed
        over matched pairs and normalized by num_boxes (DETR convention).

        Args:
            outputs['pred_pos']: (B, Q, 3) frac coords
            targets[b]['pos']:   (n_b, 3) frac coords
            indices: list of (i_idx, j_idx) per sample from matcher
            num_boxes: scalar, total GT atoms across batch (after world reduction)

        Returns:
            {'loss_pos': scalar}
        """
        assert 'pred_pos' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_pos = outputs['pred_pos'][idx]  # (sum_matched, 3)
        target_pos = torch.cat(
            [t['pos'][i] for t, (_, i) in zip(targets, indices)], dim=0
        )  # (sum_matched, 3)

        # Squared min-image L2 per matched pair, summed and normalized
        loss_pos = _min_image_l2_sq_paired(src_pos, target_pos, self.lengths)

        losses = {'loss_pos': loss_pos.sum() / num_boxes}
        return losses

    # -----------------------------------------------------------------------
    # Permutation index helpers — VERBATIM from DETR
    # -----------------------------------------------------------------------
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    # -----------------------------------------------------------------------
    # Loss dispatcher — Exp6 has 3 losses (labels, pos, cardinality)
    # -----------------------------------------------------------------------
    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'pos': self.loss_pos,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    # -----------------------------------------------------------------------
    # forward — main flow (aux_outputs loop preserved verbatim, masks handling removed)
    # -----------------------------------------------------------------------
    def forward(self, outputs, targets):
        """
        Computes all losses and returns a dict.

        Args:
            outputs: dict with 'pred_logits', 'pred_pos', optionally 'aux_outputs'
            targets: list[B] of dicts, each with 'labels' and 'pos'
        """
        outputs_without_aux = {
            k: v for k, v in outputs.items() if k != 'aux_outputs'
        }

        # Match last decoder layer outputs against targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute average num_boxes for normalization
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float,
            device=next(iter(outputs.values())).device,
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Last-layer losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # Auxiliary losses — repeat for each intermediate decoder layer
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels':
                        # Suppress class_error logging for aux layers
                        kwargs = {'log': False}
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices, num_boxes, **kwargs
                    )
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses
