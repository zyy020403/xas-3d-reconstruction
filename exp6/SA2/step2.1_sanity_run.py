"""
step2.1_sanity_run.py — Sanity training (5 ep × 1000 train subset).

Per handoff §3.

Validates BEFORE Step 2.2 full train:
  [1] cls:pos ratio at ep5 end ∈ [1/3, 3]            (handoff §3.4)
  [2] val no_object_ratio at ep5 ∈ [1/20, 6/20]
  [3] val query_diversity at ep5 > 0.05              (no query_pile-up)
  [4] zero NaN/Inf grad batches across all epochs
  [5] val_rmsd at ep5 < 2.0 Å
  [6] val_setlevel_typeacc at ep5 > 0.10

ANY fail → handoff §10 push trigger. SA2 stops, push MA1, does NOT self-tune.

ALL hyperparameters from handoff §3.2 are constants in this file. Editing
those constants is forbidden by handoff §9 row 8.

Run:
    cd /home/tcat/experiment6
    /home/tcat/conda_envs/mlff/bin/python step2/step2.1_sanity_run.py 2>&1 \\
        | tee step2/step2.1_sanity_log.txt
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SHARED = REPO_ROOT / "shared"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SHARED))

import torch
from torch.amp import autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset

from shared.detr_xas import DETRXas
from shared.matcher import HungarianMatcher
from shared.criterion import SetCriterion
from shared.exp6_data_adapter import adapt
from shared.xas_local_datamodule_v2 import xas_collate_fn_v2
from shared.xas_local_dataset_v2 import XasLocalDatasetV2
from shared.eval_metrics import (
    hungarian_rmsd, set_level_type_acc, set_no_object_idx,
)

# =========================================================================
# Handoff §3.2 LOCKED. SA2 not allowed to modify these. Sanity FAIL = push.
# =========================================================================
BATCH_SIZE         = 32
NUM_WORKERS        = 4
LR_TRANSFORMER     = 1e-4    # main params
LR_TOKENIZER       = 1e-5    # SpectrumTokenizer (Exp4 frontend, smaller LR)
WEIGHT_DECAY       = 1e-4
GRAD_CLIP          = 0.1
LR_DROP_STEP       = 200
LR_DROP_GAMMA      = 0.1
COST_CLASS         = 1.0     # matcher
COST_POS           = 1.0     # matcher (NOT 5.0 — see proposal §5 caveat)
LAMBDA_CLS         = 1.0     # criterion weight
LAMBDA_POS         = 1.0     # criterion weight (MA1 reverse-engineered)
EOS_COEF           = 0.1     # no_object class weight in CE
NUM_DECODER_LAYERS = 6
N_AUX              = NUM_DECODER_LAYERS - 1
SEED               = 42

VOCAB_PATH = SHARED / "exp6_element_vocab.json"
DATA_DIR   = "/home/tcat/diffcsp_exp4/data"

# =========================================================================
# Helpers
# =========================================================================

def build_weight_dict() -> dict:
    """Aux layers carry same weight as main per DETR convention."""
    wd = {"loss_ce": LAMBDA_CLS, "loss_pos": LAMBDA_POS}
    for i in range(N_AUX):
        wd[f"loss_ce_{i}"] = LAMBDA_CLS
        wd[f"loss_pos_{i}"] = LAMBDA_POS
    return wd


def build_idx_to_label(model: DETRXas):
    """idx (0..N_NEIGHBOR_TYPES) → human-readable label ('Z=8 O' or 'no_object').
    Used for val argmax histogram stdout (MA1 Option B spec)."""
    from pymatgen.core import Element
    inv = {int(v): int(k) for k, v in model.neighbor_Z_to_idx.items()}
    no_obj = int(model.no_object_idx)
    def _label(idx: int) -> str:
        idx = int(idx)
        if idx == no_obj:
            return "no_object"
        if idx not in inv:
            return f"idx{idx}_OOV"
        Z = inv[idx]
        return f"Z={Z} {Element.from_Z(Z).symbol}"
    return _label


def build_param_groups(model: DETRXas) -> list[dict]:
    """Two LR groups per handoff §3.2: tokenizer 1e-5, all else 1e-4."""
    tok = [p for n, p in model.named_parameters() if n.startswith("tokenizer.")]
    rest = [p for n, p in model.named_parameters() if not n.startswith("tokenizer.")]
    assert len(tok) > 0, "no params under model.tokenizer.* — name mismatch?"
    print(f"  param groups: tokenizer={sum(p.numel() for p in tok):,} @ lr={LR_TOKENIZER}, "
          f"rest={sum(p.numel() for p in rest):,} @ lr={LR_TRANSFORMER}")
    return [
        {"params": tok, "lr": LR_TOKENIZER},
        {"params": rest, "lr": LR_TRANSFORMER},
    ]


def make_subset_loader(split: str, n_samples: int | None, shuffle: bool) -> DataLoader:
    """Construct a subset DataLoader with deterministic random selection."""
    ds = XasLocalDatasetV2(
        split=split, data_dir=DATA_DIR, verbose_init_benchmark=False
    )
    if n_samples is not None and n_samples < len(ds):
        g = torch.Generator().manual_seed(SEED)
        idx = torch.randperm(len(ds), generator=g)[:n_samples].tolist()
        ds = Subset(ds, idx)
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        collate_fn=xas_collate_fn_v2,
        drop_last=shuffle,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
    )


def compute_diagnostics(pred_logits, pred_pos, no_object_idx) -> tuple[float, float]:
    """proposal §附录 B.5 verbatim: no_object_ratio + query_diversity."""
    argmax = pred_logits.argmax(dim=-1)                            # (B, 20)
    is_no_obj = (argmax == no_object_idx).float()
    no_obj_ratio = is_no_obj.mean(dim=-1).mean().item()
    query_div = pred_pos.std(dim=1).mean(dim=-1).mean().item()
    return no_obj_ratio, query_div


class TrainAcc:
    """Per-epoch sample-weighted running averages."""
    def __init__(self):
        self.n = 0
        self.s_total = self.s_ce = self.s_pos = 0.0
        self.s_no_obj = self.s_qdiv = 0.0

    def add(self, loss_total, ce, pos, no_obj, qdiv, B):
        self.n += B
        self.s_total += loss_total * B
        self.s_ce += ce * B
        self.s_pos += pos * B
        self.s_no_obj += no_obj * B
        self.s_qdiv += qdiv * B

    @property
    def avg(self) -> dict:
        if self.n == 0:
            return {k: float("nan") for k in
                    ("loss_total", "loss_ce", "loss_pos", "no_obj", "qdiv", "ratio")}
        ce, pos = self.s_ce / self.n, self.s_pos / self.n
        return {
            "loss_total": self.s_total / self.n,
            "loss_ce": ce, "loss_pos": pos,
            "no_obj": self.s_no_obj / self.n, "qdiv": self.s_qdiv / self.n,
            "ratio": ce / (LAMBDA_POS * pos) if pos > 0 else float("inf"),
        }


# =========================================================================
# Validation
# =========================================================================

@torch.no_grad()
def run_val(model, val_loader, device, autocast_enabled, autocast_dtype):
    model.eval()
    no_obj_idx = model.no_object_idx
    n_classes_total = model.n_neighbor_types + 1   # 90 incl. no_object
    lengths = model.lengths.to(device)

    rmsds, typeaccs, no_obj_list, qdiv_list = [], [], [], []
    all_argmax_chunks = []  # MA1 Option B: collect argmax across val for histogram

    for pyg_batch in val_loader:
        if pyg_batch is None:
            continue
        pyg_batch = pyg_batch.to(device)
        model_batch, targets = adapt(pyg_batch, model)
        if model_batch is None:
            continue
        with autocast(device_type="cuda", dtype=autocast_dtype, enabled=autocast_enabled):
            out = model(model_batch)
        # Cast to fp32 for stable eval (scipy + Hungarian operate cpu fp32 anyway)
        pred_logits = out["pred_logits"].float()
        pred_pos = out["pred_pos"].float()
        argmax = pred_logits.argmax(dim=-1)
        all_argmax_chunks.append(argmax.flatten().cpu())   # accumulate

        for i in range(pred_pos.shape[0]):
            pp = pred_pos[i]
            pa = argmax[i]
            gp = targets[i]["pos"].float()
            gt = targets[i]["labels"]
            rmsd, _ = hungarian_rmsd(pp, pa, gp, gt, lengths)
            tacc = set_level_type_acc(pa, gt)
            if rmsd != float("inf"):
                rmsds.append(rmsd)
            typeaccs.append(tacc)

        no_obj, qdiv = compute_diagnostics(pred_logits, pred_pos, no_obj_idx)
        no_obj_list.append(no_obj)
        qdiv_list.append(qdiv)

    # ---- argmax histogram (MA1 Option B spec) ----
    if all_argmax_chunks:
        all_argmax = torch.cat(all_argmax_chunks)
        counts = torch.bincount(all_argmax, minlength=n_classes_total)
        total = int(counts.sum().item())
        pcts = counts.float() / total if total > 0 else counts.float()
        sorted_idx = counts.argsort(descending=True)
        top5 = [
            (int(sorted_idx[k].item()),
             int(counts[sorted_idx[k]].item()),
             float(pcts[sorted_idx[k]].item()))
            for k in range(min(5, n_classes_total))
        ]
        n_classes_ge_5pct = int((pcts >= 0.05).sum().item())
    else:
        top5 = []
        total = 0
        n_classes_ge_5pct = 0

    model.train()
    return {
        "rmsd":    sum(rmsds) / len(rmsds) if rmsds else float("inf"),
        "typeacc": sum(typeaccs) / len(typeaccs) if typeaccs else 0.0,
        "no_obj":  sum(no_obj_list) / len(no_obj_list) if no_obj_list else 0.0,
        "qdiv":    sum(qdiv_list) / len(qdiv_list) if qdiv_list else 0.0,
        "n_val":   len(typeaccs),
        "top5":    top5,
        "argmax_total": total,
        "n_classes_ge_5pct": n_classes_ge_5pct,
    }


# =========================================================================
# Main
# =========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--train-subset", type=int, default=1000)
    ap.add_argument("--val-subset", type=int, default=500,
                    help="cap val for sanity speed; pass -1 for full val")
    ap.add_argument("--precision", choices=["bf16-mixed", "fp32"],
                    default="bf16-mixed")
    ap.add_argument("--csv-out", type=str,
                    default=str(REPO_ROOT / "step2" / "step2.1_sanity_metrics.csv"))
    args = ap.parse_args()

    print("=" * 72)
    print("Exp6 Step 2.1 — sanity (5 ep × 1000 train subset)")
    print("=" * 72)
    print(f"  precision     : {args.precision}")
    print(f"  epochs        : {args.epochs}")
    print(f"  train_subset  : {args.train_subset}")
    print(f"  val_subset    : {args.val_subset if args.val_subset > 0 else 'FULL'}")
    print(f"  csv_out       : {args.csv_out}")
    print(f"  vocab         : {VOCAB_PATH}")
    print(f"  data_dir      : {DATA_DIR}")
    print(f"  hyperparams   : LOCKED — λ_cls={LAMBDA_CLS}, λ_pos={LAMBDA_POS}, "
          f"cost_pos={COST_POS}, eos={EOS_COEF}, lr={LR_TRANSFORMER}/{LR_TOKENIZER}, "
          f"clip={GRAD_CLIP}")

    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "sanity requires GPU"

    autocast_enabled = (args.precision == "bf16-mixed")
    autocast_dtype = torch.bfloat16

    # ---------- DataLoaders ----------
    print("\n[setup] DataLoaders ...")
    train_loader = make_subset_loader("train", args.train_subset, shuffle=True)
    val_loader = make_subset_loader(
        "val", args.val_subset if args.val_subset > 0 else None, shuffle=False
    )
    print(f"  train batches/ep ≈ {len(train_loader)}")
    print(f"  val batches      ≈ {len(val_loader)}")

    # ---------- Model + criterion ----------
    print("\n[setup] DETRXas + matcher + criterion ...")
    model = DETRXas(vocab_path=str(VOCAB_PATH)).to(device)
    matcher = HungarianMatcher(cost_class=COST_CLASS, cost_pos=COST_POS).to(device)
    criterion = SetCriterion(
        num_classes=model.n_neighbor_types,
        matcher=matcher,
        weight_dict=build_weight_dict(),
        eos_coef=EOS_COEF,
        losses=["labels", "pos", "cardinality"],
    ).to(device)
    weight_dict = criterion.weight_dict

    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == 18_226_205, f"param count drift: {n_params} (expected 18,226,205)"
    print(f"  total params: {n_params:,} ✓")

    # Inject NO_OBJECT_IDX into eval_metrics module (SA1 anti-drift design:
    # eval_metrics doesn't re-read vocab.json, gets it from model attr).
    # MUST be called before any val_step / hungarian_rmsd / set_level_type_acc.
    set_no_object_idx(model.no_object_idx)
    print(f"  eval_metrics.NO_OBJECT_IDX ← {model.no_object_idx}")

    # idx → human-readable label (for stdout histogram, MA1 Option B)
    idx_to_label = build_idx_to_label(model)

    optimizer = AdamW(build_param_groups(model), weight_decay=WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=LR_DROP_STEP, gamma=LR_DROP_GAMMA)

    # ---------- CSV ----------
    csv_path = Path(args.csv_out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_fields = [
        "epoch", "n_train_steps", "epoch_time_s", "val_time_s",
        "train_loss_total", "train_loss_ce", "train_loss_pos", "train_cls_pos_ratio",
        "train_no_obj_ratio", "train_query_diversity",
        "val_rmsd", "val_setlevel_typeacc", "val_no_obj_ratio", "val_query_diversity",
        # MA1 Option B additions — argmax histogram per epoch
        "val_top1_idx", "val_top1_pct", "val_top2_idx", "val_top2_pct",
        "val_n_classes_ge_5pct",
    ]
    csv_f = open(csv_path, "w", newline="")
    csv_w = csv.DictWriter(csv_f, fieldnames=csv_fields)
    csv_w.writeheader()

    # ---------- Train loop ----------
    print("\n[train] starting ...\n")
    global_step = 0
    n_nan_batches_total = 0
    last_acc_avg = None
    last_val = None

    # Trajectory for MA1 Option B decision-tree analysis at end
    traj = {
        "epoch": [], "qdiv": [], "rmsd": [], "typeacc": [],
        "no_obj": [], "top1_idx": [], "top1_pct": [],
        "top2_idx": [], "top2_pct": [], "n_classes_ge_5pct": [],
    }

    for epoch in range(args.epochs):
        model.train()
        ep_t0 = time.time()
        acc = TrainAcc()

        for batch_i, pyg_batch in enumerate(train_loader):
            if pyg_batch is None:
                continue
            pyg_batch = pyg_batch.to(device)
            model_batch, targets = adapt(pyg_batch, model)
            if model_batch is None:
                continue
            B = model_batch["xmu"].shape[0]

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", dtype=autocast_dtype, enabled=autocast_enabled):
                out = model(model_batch)
                loss_dict = criterion(out, targets)
                total = sum(
                    loss_dict[k] * weight_dict[k]
                    for k in weight_dict if k in loss_dict
                )

            if not torch.isfinite(total):
                n_nan_batches_total += 1
                print(f"  ⚠️  ep{epoch} step{batch_i}: non-finite loss = {total.item()}, skip")
                continue

            total.backward()

            n_nan_grad = sum(
                1 for p in model.parameters()
                if p.grad is not None and torch.isnan(p.grad).any()
            )
            if n_nan_grad > 0:
                csv_f.close()
                raise RuntimeError(
                    f"NaN gradient at ep{epoch} step{global_step}: "
                    f"{n_nan_grad} params affected. STOP per handoff §10 row 5."
                )

            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            # NOTE: scheduler.step() moved to epoch level (after val) per MA1 B' spec.
            # PyTorch StepLR with step_size=200 was being called per-batch — would
            # fire LR decay every 200 iters (~6.5 ep), reaching LR=1e-8 by ep25.
            # Correct semantic: step_size=200 = 200 epochs per DETR convention.

            with torch.no_grad():
                no_obj, qdiv = compute_diagnostics(
                    out["pred_logits"].float(), out["pred_pos"].float(),
                    model.no_object_idx,
                )
            acc.add(total.item(),
                    loss_dict["loss_ce"].item(),
                    loss_dict["loss_pos"].item(),
                    no_obj, qdiv, B)
            global_step += 1

            if global_step % 50 == 0 or global_step == 1:
                a = acc.avg
                lr = optimizer.param_groups[1]["lr"]
                print(f"  ep{epoch} step{global_step:5d}  "
                      f"L={a['loss_total']:6.3f}  ce={a['loss_ce']:.3f}  "
                      f"pos={a['loss_pos']:.3f}  ratio={a['ratio']:.2f}  "
                      f"no_obj={a['no_obj']:.3f}  qdiv={a['qdiv']:.4f}  "
                      f"lr={lr:.2e}")

        ep_time = time.time() - ep_t0
        a = acc.avg

        val_t0 = time.time()
        val_metrics = run_val(model, val_loader, device,
                              autocast_enabled, autocast_dtype)
        val_time = time.time() - val_t0

        print(f"\n  EPOCH {epoch} done ({ep_time:.1f}s train + {val_time:.1f}s val, "
              f"n_val={val_metrics['n_val']})")
        print(f"    train  L={a['loss_total']:.3f}  ce={a['loss_ce']:.3f}  "
              f"pos={a['loss_pos']:.3f}  ratio={a['ratio']:.2f}  "
              f"no_obj={a['no_obj']:.3f}  qdiv={a['qdiv']:.4f}")
        print(f"    val    rmsd={val_metrics['rmsd']:.4f} Å  "
              f"typeacc={val_metrics['typeacc']:.4f}  "
              f"no_obj={val_metrics['no_obj']:.3f}  qdiv={val_metrics['qdiv']:.4f}")

        # ---- MA1 Option B: per-epoch argmax top-5 ----
        print(f"    val argmax top-5 (n={val_metrics['argmax_total']}):")
        for idx, count, pct in val_metrics["top5"]:
            print(f"      [{idx:3d}] {idx_to_label(idx):18s} : {count:6d}  ({pct*100:6.2f}%)")
        print(f"    val n_classes ≥ 5% share = {val_metrics['n_classes_ge_5pct']}\n")

        top1_idx = val_metrics["top5"][0][0] if val_metrics["top5"] else -1
        top1_pct = val_metrics["top5"][0][2] if val_metrics["top5"] else 0.0
        top2_idx = val_metrics["top5"][1][0] if len(val_metrics["top5"]) > 1 else -1
        top2_pct = val_metrics["top5"][1][2] if len(val_metrics["top5"]) > 1 else 0.0

        traj["epoch"].append(epoch)
        traj["qdiv"].append(val_metrics["qdiv"])
        traj["rmsd"].append(val_metrics["rmsd"])
        traj["typeacc"].append(val_metrics["typeacc"])
        traj["no_obj"].append(val_metrics["no_obj"])
        traj["top1_idx"].append(top1_idx)
        traj["top1_pct"].append(top1_pct)
        traj["top2_idx"].append(top2_idx)
        traj["top2_pct"].append(top2_pct)
        traj["n_classes_ge_5pct"].append(val_metrics["n_classes_ge_5pct"])

        csv_w.writerow({
            "epoch": epoch, "n_train_steps": global_step,
            "epoch_time_s": ep_time, "val_time_s": val_time,
            "train_loss_total": a["loss_total"], "train_loss_ce": a["loss_ce"],
            "train_loss_pos": a["loss_pos"], "train_cls_pos_ratio": a["ratio"],
            "train_no_obj_ratio": a["no_obj"], "train_query_diversity": a["qdiv"],
            "val_rmsd": val_metrics["rmsd"], "val_setlevel_typeacc": val_metrics["typeacc"],
            "val_no_obj_ratio": val_metrics["no_obj"],
            "val_query_diversity": val_metrics["qdiv"],
            "val_top1_idx": top1_idx, "val_top1_pct": top1_pct,
            "val_top2_idx": top2_idx, "val_top2_pct": top2_pct,
            "val_n_classes_ge_5pct": val_metrics["n_classes_ge_5pct"],
        })
        csv_f.flush()
        last_acc_avg = a
        last_val = val_metrics

        # Per-epoch scheduler step (MA1 B' fix). step_size=200 means sanity (30 ep)
        # never triggers decay — that's correct, sanity doesn't need it.
        scheduler.step()
        cur_lr_main = optimizer.param_groups[1]["lr"]
        cur_lr_tok = optimizer.param_groups[0]["lr"]
        print(f"    LR after scheduler.step(): main={cur_lr_main:.2e}, tokenizer={cur_lr_tok:.2e}")

    csv_f.close()

    # ---------- MA1 Option B decision-tree input dump ----------
    # NOT a verdict — SA2 does NOT auto-pick A/C/push. MA1 maps to decision
    # tree based on the trajectory below. SA2 reports & waits.
    print("\n" + "=" * 72)
    print("DECISION-TREE INPUTS (for MA1 — Option B trajectory + histogram)")
    print("=" * 72)
    print()

    # Per-epoch table
    print(f"  {'ep':>3s}  {'qdiv':>7s}  {'rmsd':>7s}  {'tacc':>6s}  "
          f"{'no_obj':>7s}  {'top1':>6s}  {'top1%':>7s}  {'top2':>6s}  {'top2%':>7s}  {'n≥5%':>5s}")
    for i in range(len(traj["epoch"])):
        ep = traj["epoch"][i]
        print(
            f"  {ep:3d}  "
            f"{traj['qdiv'][i]:7.4f}  "
            f"{traj['rmsd'][i]:7.4f}  "
            f"{traj['typeacc'][i]:6.4f}  "
            f"{traj['no_obj'][i]:7.4f}  "
            f"{traj['top1_idx'][i]:6d}  "
            f"{traj['top1_pct'][i]*100:6.2f}%  "
            f"{traj['top2_idx'][i]:6d}  "
            f"{traj['top2_pct'][i]*100:6.2f}%  "
            f"{traj['n_classes_ge_5pct'][i]:5d}"
        )

    # Trajectory shape diagnostics (MA1 needs these to map to decision tree)
    qdivs = traj["qdiv"]
    n = len(qdivs)
    qdiv_final = qdivs[-1] if qdivs else float("nan")

    if n >= 6:
        first_third = sum(qdivs[: n // 3]) / max(1, n // 3)
        last_third = sum(qdivs[-(n // 3):]) / max(1, n // 3)
    else:
        first_third = qdivs[0] if qdivs else float("nan")
        last_third = qdivs[-1] if qdivs else float("nan")

    qdiv_max = max(qdivs) if qdivs else float("nan")
    qdiv_min = min(qdivs) if qdivs else float("nan")
    monotone_increases = sum(
        1 for i in range(1, n) if qdivs[i] > qdivs[i - 1]
    )
    monotone_frac = monotone_increases / max(1, n - 1)

    # Plateau test on last 10 epochs
    if n >= 10:
        last10 = qdivs[-10:]
        last10_mean = sum(last10) / 10
        last10_var = sum((x - last10_mean) ** 2 for x in last10) / 10
        last10_std = last10_var ** 0.5
    else:
        last10_std = float("nan")

    print()
    print("  TRAJECTORY SHAPE:")
    print(f"    qdiv @ ep{traj['epoch'][-1]:>3d}    = {qdiv_final:.4f}")
    print(f"    qdiv first 1/3 avg     = {first_third:.4f}")
    print(f"    qdiv last  1/3 avg     = {last_third:.4f}")
    print(f"    qdiv max - min         = {qdiv_max - qdiv_min:.4f}  (range)")
    print(f"    monotone-increase frac = {monotone_frac:.2f}  ({monotone_increases}/{n - 1} steps)")
    print(f"    last-10-ep std         = {last10_std:.5f}  (plateau if << range)")
    print()
    print(f"  TOP-1 CLASS @ FINAL EPOCH:")
    print(f"    idx                    = {traj['top1_idx'][-1]}  "
          f"({idx_to_label(traj['top1_idx'][-1])})")
    print(f"    pct                    = {traj['top1_pct'][-1]*100:.2f}%")
    print(f"    n_classes ≥ 5%         = {traj['n_classes_ge_5pct'][-1]}")

    # MA1 decision-tree row mapping (informational, NOT auto-pick)
    print()
    print("  MA1 DECISION TREE ROW MATCH (informational — MA1 picks):")
    rows_matched = []
    if qdiv_final > 0.05:
        rows_matched.append("Row 1 → A: full train, qdiv healthy")
    if 0.02 <= qdiv_final <= 0.05 and monotone_frac >= 0.6 and traj["n_classes_ge_5pct"][-1] >= 2:
        rows_matched.append("Row 2 → A with close monitor; ≥2 classes & monotone")
    if 0.02 <= qdiv_final <= 0.05 and (monotone_frac < 0.6 or last10_std < 0.005):
        rows_matched.append("Row 3 → C-then-A: lr warmup, pile-up plateau")
    if qdiv_final < 0.02 and last10_std < 0.005 and traj["top1_pct"][-1] >= 0.90:
        rows_matched.append("Row 4 → push MA1: arch-level concern, top1 dominates")
    if qdiv_final < 0.005:
        rows_matched.append("Row 5 → IMMEDIATE push MA1")
    if not rows_matched:
        rows_matched.append("(no clean row match — MA1 decides based on trajectory shape)")
    for r in rows_matched:
        print(f"    • {r}")

    print()
    print(f"  csv: {csv_path}")
    print(f"  log: tee target")
    print()
    print("STATUS: Phase 2.2 (full train) BLOCKED until MA1 reviews this trajectory.")
    print("=" * 72)
    sys.exit(0)


if __name__ == "__main__":
    main()
