"""
Apply v7 criterion patch to experiment6_v7/shared/criterion.py.
Adds: repulsion hinge + pairwise_min + shell_dist + shell_count losses.
Run from /home/tcat/experiment6_v7:
  /home/tcat/conda_envs/mlff/bin/python /home/tcat/criterion_patch.py
"""
from pathlib import Path

SRC = Path("/home/tcat/experiment6_v7/shared/criterion.py")
text = SRC.read_text()

# ── Patch 1: __init__ signature — add no_object_idx + min_pdist ────────────
OLD_INIT = "    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,\n                 lengths=(6.0, 6.0, 6.0)):"
NEW_INIT = "    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,\n                 lengths=(6.0, 6.0, 6.0), no_object_idx=None, min_pdist=1.5076):"
assert OLD_INIT in text, "PATCH 1 anchor not found"
text = text.replace(OLD_INIT, NEW_INIT, 1)

# ── Patch 2: store no_object_idx + min_pdist after self.losses = losses ────
OLD_STORE = "        self.losses = losses\n\n        # empty_weight"
NEW_STORE = """        self.losses = losses
        self.no_object_idx = no_object_idx if no_object_idx is not None else num_classes
        self.min_pdist = float(min_pdist)

        # empty_weight"""
assert OLD_STORE in text, "PATCH 2 anchor not found"
text = text.replace(OLD_STORE, NEW_STORE, 1)

# ── Patch 3: insert four new loss methods before get_loss ──────────────────
NEW_LOSSES = '''
    # -----------------------------------------------------------------------
    # loss_repulsion — v4: pairwise repulsion hinge (pred-pred, no_object filtered)
    # -----------------------------------------------------------------------
    def loss_repulsion(self, outputs, targets, indices, num_boxes, **kwargs):
        """
        Hinge loss penalising pred-pred pairs closer than min_pdist.
        Continuous: loss = mean over violating pairs of (min_pdist - d)^2.
        Only valid (non-no_object) queries contribute.
        """
        pred_pos = outputs["pred_pos"]          # (B, 20, 3) frac
        argmax   = outputs["pred_logits"].detach().argmax(-1)  # (B, 20)
        B = pred_pos.shape[0]
        batch_loss = []
        for b in range(B):
            valid = (argmax[b] != self.no_object_idx)
            if valid.sum() < 2:
                batch_loss.append(pred_pos.new_zeros(()))
                continue
            vp = pred_pos[b][valid]                         # (k, 3)
            diff = vp[:, None] - vp[None, :]               # (k, k, 3)
            diff = diff - torch.round(diff)
            cart = diff * self.lengths
            pdist = torch.norm(cart, dim=-1)               # (k, k)
            eye = torch.eye(vp.shape[0], dtype=torch.bool, device=pdist.device)
            pd_off = pdist[~eye]
            violation = torch.clamp(self.min_pdist - pd_off, min=0.0)
            batch_loss.append((violation ** 2).mean())
        return {"loss_rep": torch.stack(batch_loss).mean()}

    # -----------------------------------------------------------------------
    # loss_pairwise_min — v7: soft sigmoid hard-indicator (persistent gradient)
    # -----------------------------------------------------------------------
    def loss_pairwise_min(self, outputs, targets, indices, num_boxes, **kwargs):
        """
        Soft per-sample indicator: sigmoid(10*(min_pdist - min_pair_dist)).
        Approaches 1 when any pair < min_pdist, 0 when all pairs >= min_pdist.
        Complements repulsion hinge: hinge gradient dies out once d >> min_pdist,
        sigmoid provides persistent gradient near the boundary.
        """
        pred_pos = outputs["pred_pos"]
        argmax   = outputs["pred_logits"].detach().argmax(-1)
        B = pred_pos.shape[0]
        flags = []
        for b in range(B):
            valid = (argmax[b] != self.no_object_idx)
            if valid.sum() < 2:
                flags.append(pred_pos.new_zeros(()))
                continue
            vp = pred_pos[b][valid]
            diff = vp[:, None] - vp[None, :]
            diff = diff - torch.round(diff)
            cart = diff * self.lengths
            pdist = torch.norm(cart, dim=-1)
            eye = torch.eye(vp.shape[0], dtype=torch.bool, device=pdist.device)
            min_d = pdist[~eye].min()
            flags.append(torch.sigmoid(10.0 * (self.min_pdist - min_d)))
        return {"loss_pmin": torch.stack(flags).mean()}

    # -----------------------------------------------------------------------
    # loss_shell_dist — v7: sort-aligned distance loss vs GT shell midpoints
    # -----------------------------------------------------------------------
    def loss_shell_dist(self, outputs, targets, indices, num_boxes, **kwargs):
        """
        Encourage pred atoms to lie at GT shell midpoint distances.
        Algorithm: sort valid pred by radial distance, align to GT shell midpoints
        (each shell_mid repeated shell_n_atoms times), compute MSE.
        """
        pred_pos = outputs["pred_pos"]
        argmax   = outputs["pred_logits"].detach().argmax(-1)
        B = pred_pos.shape[0]
        losses = []
        for b in range(B):
            t = targets[b]
            if "shell_starts" not in t or len(t["shell_starts"]) == 0:
                continue
            valid = (argmax[b] != self.no_object_idx)
            if valid.sum() == 0:
                continue
            vp = pred_pos[b][valid]
            pred_dist = (vp * self.lengths).norm(dim=-1)
            pred_dist_sorted, _ = pred_dist.sort()

            gt_targets = []
            n_shells = min(2, len(t["shell_starts"]))
            for s in range(n_shells):
                mid = (float(t["shell_starts"][s]) + float(t["shell_ends"][s])) / 2.0
                n   = int(t["shell_n_atoms"][s])
                gt_targets.extend([mid] * n)
            if len(gt_targets) == 0:
                continue
            gt_t = pred_pos.new_tensor(gt_targets)
            n_cmp = min(len(gt_t), len(pred_dist_sorted))
            losses.append(((pred_dist_sorted[:n_cmp] - gt_t[:n_cmp]) ** 2).mean())

        if len(losses) == 0:
            return {"loss_sdist": outputs["pred_pos"].new_zeros(())}
        return {"loss_sdist": torch.stack(losses).mean()}

    # -----------------------------------------------------------------------
    # loss_shell_count — v7: differentiable shell occupancy vs GT count
    # -----------------------------------------------------------------------
    def loss_shell_count(self, outputs, targets, indices, num_boxes, **kwargs):
        """
        Penalise mismatch between predicted shell occupancy and GT count.
        Uses sigmoid-based differentiable counting within GT shell boundaries
        expanded by TOL_BAND = 0.1 Å.
        """
        TOL_BAND = 0.1
        TAU      = 20.0
        pred_pos = outputs["pred_pos"]
        argmax   = outputs["pred_logits"].detach().argmax(-1)
        B = pred_pos.shape[0]
        losses = []
        for b in range(B):
            t = targets[b]
            if "shell_starts" not in t or len(t["shell_starts"]) == 0:
                continue
            valid = (argmax[b] != self.no_object_idx)
            if valid.sum() == 0:
                continue
            vp = pred_pos[b][valid]
            pred_dist = (vp * self.lengths).norm(dim=-1)

            n_shells = min(2, len(t["shell_starts"]))
            for s in range(n_shells):
                lo     = float(t["shell_starts"][s]) - TOL_BAND
                hi     = float(t["shell_ends"][s])   + TOL_BAND
                n_gt   = float(t["shell_n_atoms"][s])
                n_soft = (torch.sigmoid(TAU * (pred_dist - lo)) -
                          torch.sigmoid(TAU * (pred_dist - hi))).sum()
                losses.append(torch.abs(n_soft - n_gt))

        if len(losses) == 0:
            return {"loss_scount": outputs["pred_pos"].new_zeros(())}
        return {"loss_scount": torch.stack(losses).mean()}

'''

# Insert before the get_loss dispatcher
OLD_DISPATCHER = '''    # -----------------------------------------------------------------------
    # Loss dispatcher — Exp6 has 3 losses (labels, pos, cardinality)
    # -----------------------------------------------------------------------
    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'pos': self.loss_pos,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)'''

NEW_DISPATCHER = NEW_LOSSES + '''    # -----------------------------------------------------------------------
    # Loss dispatcher — Exp6 v7: 7 losses
    # -----------------------------------------------------------------------
    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels':       self.loss_labels,
            'cardinality':  self.loss_cardinality,
            'pos':          self.loss_pos,
            'repulsion':    self.loss_repulsion,
            'pairwise_min': self.loss_pairwise_min,
            'shell_dist':   self.loss_shell_dist,
            'shell_count':  self.loss_shell_count,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)'''

assert OLD_DISPATCHER in text, "PATCH 3 anchor not found"
text = text.replace(OLD_DISPATCHER, NEW_DISPATCHER, 1)

# Write back
SRC.write_text(text)
print(f"Patched {SRC}  ({SRC.stat().st_size} bytes)")

# Quick verify
t2 = SRC.read_text()
for check in ["loss_repulsion", "loss_pairwise_min", "loss_shell_dist", "loss_shell_count",
              "no_object_idx", "min_pdist", "'repulsion'", "'pairwise_min'",
              "'shell_dist'", "'shell_count'"]:
    ok = check in t2
    print(f"  {'✓' if ok else 'MISSING'} {check}")
print("Patch complete.")
