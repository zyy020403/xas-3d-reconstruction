# =============================================================================
# 脚本编号: step4.1_qt
# 脚本名称: step4.1_qt_sample.py
# 输入:
#   - experiment/quicktest/qt_output/best checkpoint (.ckpt)
#   - experiment/quicktest/qt_inventory.csv
#   - experiment/quicktest/qt_test_ids.txt
#   - experiment/quicktest/qt_val_ids.txt
# 输出:
#   - experiment/quicktest/qt_step4/predictions_val.pt
#   - experiment/quicktest/qt_step4/predictions_test.pt
# 说明:
#   QuickTest 版采样脚本（对应正式服 step4.1_sample_predictions.py）。
#   差异：
#     - 路径全部指向 quicktest/
#     - 无 holdout 过滤
#     - 复用 step3_qt_train.py 的 sys.modules 注入逻辑（必须在 import diffusion 前完成）
#     - 自动选取 qt_output/ 下 val_loss 最小的 checkpoint
# =============================================================================

import os
import sys
import glob
import types
import importlib.util
import warnings
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
warnings.filterwarnings("ignore")

PROJECT_ROOT   = r"C:\Users\T-Cat\Desktop\DiffCSP-main"
EXPERIMENT_DIR = os.path.join(PROJECT_ROOT, "experiment")
QT_DIR         = os.path.join(EXPERIMENT_DIR, "quicktest")
QT_OUTPUT_DIR  = os.path.join(QT_DIR, "qt_output")
QT_STEP4_DIR   = os.path.join(QT_DIR, "qt_step4")
STEP2_DIR      = os.path.join(EXPERIMENT_DIR, "step2")
os.makedirs(QT_STEP4_DIR, exist_ok=True)

os.environ["PROJECT_ROOT"] = PROJECT_ROOT
for p in [PROJECT_ROOT, QT_DIR, STEP2_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# =============================================================================
# ★ sys.modules 注入（与 step3_qt_train.py 完全相同，必须在 import diffusion 前）
# =============================================================================
import torch
import torch.nn as nn

# 1. k¹ 编码器
_qt_encoder_path = os.path.join(QT_DIR, "step2.1_qt_encoder.py")
_spec = importlib.util.spec_from_file_location("step2_1_spectrum_encoder", _qt_encoder_path)
_qt_encoder_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_qt_encoder_mod)
sys.modules["step2_1_spectrum_encoder"] = _qt_encoder_mod
logging.getLogger(__name__).info("✓ 已注入 step2_1_spectrum_encoder（k¹）")

# 2. stub aggregator
class _StubAggregator(nn.Module):
    def __init__(self, d_site=256, d_struct=256):
        super().__init__()
        self._unused = nn.Linear(1, 1)
    def forward(self, padded, mask):
        raise RuntimeError("[QT] StubAggregator 不应被调用")

_stub_mod = types.ModuleType("step2_2_multisite_aggregator")
_stub_mod.MultiSiteAggregator     = _StubAggregator
_stub_mod.collate_multisite_batch = lambda *a, **k: (_ for _ in ()).throw(RuntimeError())
sys.modules["step2_2_multisite_aggregator"] = _stub_mod
logging.getLogger(__name__).info("✓ 已注入 stub step2_2_multisite_aggregator")

# =============================================================================
# 现在可以安全 import
# =============================================================================
import hydra
from tqdm import tqdm
from diffcsp.pl_modules.diffusion import CSPDiffusion
from diffcsp.common.data_utils import lattice_params_to_matrix_torch

sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
from eval_utils import lattices_to_params_shape

STEP_LR = 1e-5


# =============================================================================
# 工具：自动找 val_loss 最小的 checkpoint
# =============================================================================
def find_best_ckpt(qt_output_dir: str) -> str:
    """
    扫描 qt_output/ 下所有 .ckpt，从文件名中解析 val_loss，返回最小的。
    文件名格式: epochepoch=XXX-valval_loss=Y.YYYY.ckpt
    """
    pattern = os.path.join(qt_output_dir, "*.ckpt")
    ckpts = [p for p in glob.glob(pattern) if "last" not in os.path.basename(p)]
    if not ckpts:
        raise FileNotFoundError(f"qt_output/ 下找不到 checkpoint: {qt_output_dir}")

    def parse_val_loss(path):
        name = os.path.basename(path)
        try:
            return float(name.split("valval_loss=")[1].replace(".ckpt", ""))
        except Exception:
            return float("inf")

    best = min(ckpts, key=parse_val_loss)
    logging.getLogger(__name__).info(f"自动选取最优 checkpoint: {best}")
    return best


# =============================================================================
# 加载模型 + monkey-patch（与 train 脚本保持一致）
# =============================================================================
def load_model(ckpt_path: str) -> CSPDiffusion:
    logging.getLogger(__name__).info(f"加载 checkpoint: {ckpt_path}")

    # load_from_checkpoint 会重新执行 __init__，
    # 因为 sys.modules 已注入，import 会命中 qt 版本
    model = CSPDiffusion.load_from_checkpoint(
        ckpt_path,
        map_location="cpu",
        strict=False,
    )

    # monkey-patch（与 train 完全一致）
    def _encode_xas_qt(self, batch):
        return self.spectrum_encoder(
            batch.spectra[:, 0],
            batch.site_elements[:, 0],
            batch.is_ionic[:, 0],
        )

    def _forward_qt(self, batch):
        raise RuntimeError("采样时不应调用 forward，请检查调用路径")

    def _configure_optimizers_qt(self):
        return hydra.utils.instantiate(
            self.hparams.optim.optimizer, params=self.parameters(), _convert_="partial"
        )

    model._encode_xas        = types.MethodType(_encode_xas_qt, model)
    model.forward            = types.MethodType(_forward_qt, model)
    model.configure_optimizers = types.MethodType(_configure_optimizers_qt, model)

    # sample 也要 patch（与 train 脚本一致）
    from tqdm import tqdm as _tqdm
    from diffcsp.pl_modules.diff_utils import d_log_p_wrapped_normal as _dlp

    def _sample_qt(self, batch, step_lr=1e-5):
        batch_size = batch.num_graphs
        l_T = torch.randn([batch_size, 3, 3]).to(self.device)
        x_T = torch.rand([batch.num_nodes, 3]).to(self.device)
        if self.keep_coords:  x_T = batch.frac_coords
        if self.keep_lattice: l_T = lattice_params_to_matrix_torch(batch.lengths, batch.angles)

        time_start = self.beta_scheduler.timesteps
        traj = {time_start: {
            "num_atoms":   batch.num_atoms,
            "atom_types":  batch.atom_types,
            "frac_coords": x_T % 1.,
            "lattices":    l_T,
        }}

        struct_emb = self._encode_xas(batch)

        for t in _tqdm(range(time_start, 0, -1)):
            times    = torch.full((batch_size,), t, device=self.device)
            time_emb = self.time_embedding(times)
            cond_emb = torch.cat([struct_emb, time_emb], dim=-1)

            alphas         = self.beta_scheduler.alphas[t]
            alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]
            sigmas         = self.beta_scheduler.sigmas[t]
            sigma_x        = self.sigma_scheduler.sigmas[t]
            sigma_norm     = self.sigma_scheduler.sigmas_norm[t]

            c0 = 1.0 / torch.sqrt(alphas)
            c1 = (1 - alphas) / torch.sqrt(1 - alphas_cumprod)

            x_t = traj[t]["frac_coords"]
            l_t = traj[t]["lattices"]
            if self.keep_coords:  x_t = x_T
            if self.keep_lattice: l_t = l_T

            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            step_size = step_lr * (sigma_x / self.sigma_scheduler.sigma_begin) ** 2
            std_x     = torch.sqrt(2 * step_size)

            _, pred_x = self.decoder(cond_emb, batch.atom_types, x_t, l_t, batch.num_atoms, batch.batch)
            pred_x = pred_x * torch.sqrt(sigma_norm)
            x_t_minus_05 = x_t - step_size * pred_x + std_x * rand_x if not self.keep_coords else x_t

            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            adj_sigma = self.sigma_scheduler.sigmas[t - 1]
            step_size = sigma_x ** 2 - adj_sigma ** 2
            std_x     = torch.sqrt((adj_sigma ** 2 * step_size) / (sigma_x ** 2))

            pred_l, pred_x = self.decoder(cond_emb, batch.atom_types, x_t_minus_05, l_t, batch.num_atoms, batch.batch)
            pred_x = pred_x * torch.sqrt(sigma_norm)
            x_t_minus_1 = x_t_minus_05 - step_size * pred_x + std_x * rand_x if not self.keep_coords else x_t
            l_t_minus_1 = c0 * (l_t - c1 * pred_l) + sigmas * rand_l if not self.keep_lattice else l_t

            traj[t - 1] = {
                "num_atoms":   batch.num_atoms,
                "atom_types":  batch.atom_types,
                "frac_coords": x_t_minus_1 % 1.,
                "lattices":    l_t_minus_1,
            }

        traj_stack = {
            "num_atoms":       batch.num_atoms,
            "atom_types":      batch.atom_types,
            "all_frac_coords": torch.stack([traj[i]["frac_coords"] for i in range(time_start, -1, -1)]),
            "all_lattices":    torch.stack([traj[i]["lattices"]    for i in range(time_start, -1, -1)]),
        }
        return traj[0], traj_stack

    model.sample = types.MethodType(_sample_qt, model)

    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        logging.getLogger(__name__).info("模型已移至 CUDA")
    return model


# =============================================================================
# DataModule（内联定义，与 step3_qt_train.py 保持一致）
# =============================================================================
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch

QUALITY_WEIGHT = {"A": 1.0, "B": 0.5, "C": 0.1}
QUALITY_WEIGHT_UNKNOWN = 0.3
SPECTRA_LEN = 512

def _parse_poscar(poscar_path):
    try:
        from pymatgen.core import Structure
        struct = Structure.from_file(poscar_path)
        frac_coords = torch.tensor(struct.frac_coords, dtype=torch.float32)
        atom_types  = torch.tensor([s.specie.Z for s in struct], dtype=torch.long)
        lengths     = torch.tensor(list(struct.lattice.abc),    dtype=torch.float32)
        angles      = torch.tensor(list(struct.lattice.angles), dtype=torch.float32)
        return frac_coords, atom_types, lengths, angles, len(struct)
    except Exception as e:
        return None

class QTCrystalDataset(Dataset):
    def __init__(self, mp_ids, inventory_df):
        super().__init__()
        from pymatgen.core.periodic_table import Element as PmgElement
        preprocess_chi = sys.modules["step2_1_spectrum_encoder"].preprocess_chi

        mp_ids_set = set(int(x) for x in mp_ids)
        sub_df = inventory_df[inventory_df["mp_id"].isin(mp_ids_set)].copy()

        self._cache = []
        self.valid_mp_ids = []

        for _, row in sub_df.iterrows():
            mp_id = int(row["mp_id"])
            parsed = _parse_poscar(os.path.join(row["source_path"], "POSCAR_supercell_fixed"))
            if parsed is None:
                continue
            frac_coords, atom_types, lengths, angles, num_atoms = parsed

            chi_path = os.path.join(row["source_path"], "chi.dat")
            try:
                spec = preprocess_chi(chi_path)
            except Exception:
                spec = torch.zeros(1, SPECTRA_LEN, dtype=torch.float32)
            spectra = spec.unsqueeze(0)

            try:
                z = PmgElement(row["element"]).Z
            except Exception:
                z = 26

            tier = row.get("quality_tier", "")
            qw   = QUALITY_WEIGHT.get(str(tier).upper(), QUALITY_WEIGHT_UNKNOWN)

            self._cache.append({
                "frac_coords":     frac_coords,
                "atom_types":      atom_types,
                "lengths":         lengths,
                "angles":          angles,
                "num_atoms":       num_atoms,
                "spectra":         spectra,
                "site_elements":   torch.tensor([z],  dtype=torch.long),
                "is_ionic":        torch.tensor([0],  dtype=torch.long),
                "quality_weights": torch.tensor([qw], dtype=torch.float32),
                "n_sites":         1,
                "mp_id":           mp_id,
            })
            self.valid_mp_ids.append(mp_id)

    def __len__(self):  return len(self._cache)
    def __getitem__(self, idx): return self._cache[idx]

def qt_collate_fn(batch):
    data_list = [
        Data(
            frac_coords=item["frac_coords"],
            atom_types=item["atom_types"],
            lengths=item["lengths"].unsqueeze(0),
            angles=item["angles"].unsqueeze(0),
            num_atoms=item["num_atoms"],
            num_nodes=item["num_atoms"],
        )
        for item in batch
    ]
    pyg_batch = Batch.from_data_list(data_list)
    pyg_batch.spectra         = torch.stack([item["spectra"]         for item in batch])
    pyg_batch.site_elements   = torch.stack([item["site_elements"]   for item in batch])
    pyg_batch.is_ionic        = torch.stack([item["is_ionic"]        for item in batch])
    pyg_batch.quality_weights = torch.stack([item["quality_weights"] for item in batch])
    pyg_batch.n_sites         = torch.ones(len(batch), dtype=torch.long)
    # ★ 保留 mp_id，让采样脚本能直接用（正式服 batch 没有这个，所以正式服用 patch_mpid 修复）
    pyg_batch.mp_id           = [item["mp_id"] for item in batch]
    return pyg_batch


# =============================================================================
# 采样主逻辑
# =============================================================================
@torch.no_grad()
def run_sampling(loader, model, split_name: str) -> dict:
    results = {}
    skipped_error = 0

    for batch_idx, batch in enumerate(tqdm(loader, desc=f"Sampling [{split_name}]")):
        batch_mp_ids = batch.mp_id  # list[int]，由 qt_collate_fn 保留

        if torch.cuda.is_available():
            batch = batch.cuda()

        try:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                 enabled=torch.cuda.is_available()):
                outputs, _traj = model.sample(batch, step_lr=STEP_LR)
        except Exception as e:
            logging.getLogger(__name__).error(f"采样失败 batch {batch_idx}: {e}")
            skipped_error += 1
            continue

        pred_frac_coords = outputs["frac_coords"].detach().cpu()
        pred_num_atoms   = outputs["num_atoms"].detach().cpu()
        pred_atom_types  = outputs["atom_types"].detach().cpu()
        pred_lattices    = outputs["lattices"].detach().cpu()
        pred_lengths, pred_angles = lattices_to_params_shape(pred_lattices)

        gt_frac_coords = batch.frac_coords.detach().cpu()
        gt_num_atoms   = batch.num_atoms.detach().cpu()
        gt_atom_types  = batch.atom_types.detach().cpu()
        gt_lengths_raw = batch.lengths.detach().cpu()
        gt_angles_raw  = batch.angles.detach().cpu()

        pred_start = gt_start = 0
        for i in range(len(batch_mp_ids)):
            mp_id  = batch_mp_ids[i]
            n_pred = pred_num_atoms[i].item()
            n_gt   = gt_num_atoms[i].item()

            results[mp_id] = {
                "pred_frac_coords": pred_frac_coords[pred_start : pred_start + n_pred],
                "pred_lengths":     pred_lengths[i],
                "pred_angles":      pred_angles[i],
                "pred_atom_types":  pred_atom_types[pred_start : pred_start + n_pred],
                "gt_frac_coords":   gt_frac_coords[gt_start : gt_start + n_gt],
                "gt_lengths":       gt_lengths_raw[i].squeeze(),
                "gt_angles":        gt_angles_raw[i].squeeze(),
                "gt_atom_types":    gt_atom_types[gt_start : gt_start + n_gt],
                "n_atoms":          n_gt,
            }
            pred_start += n_pred
            gt_start   += n_gt

    logging.getLogger(__name__).info(
        f"[{split_name}] 完成，共 {len(results)} 个化合物，采样失败 {skipped_error} 个 batch"
    )
    return results


def main():
    ckpt_path = find_best_ckpt(QT_OUTPUT_DIR)
    model     = load_model(ckpt_path)

    inventory_df = pd.read_csv(os.path.join(QT_DIR, "qt_inventory.csv"))

    def _read_ids(path):
        with open(path) as f:
            return [int(l.strip()) for l in f if l.strip()]

    val_ids  = _read_ids(os.path.join(QT_DIR, "qt_val_ids.txt"))
    test_ids = _read_ids(os.path.join(QT_DIR, "qt_test_ids.txt"))

    logging.getLogger(__name__).info(f"val={len(val_ids)}  test={len(test_ids)}")

    val_dataset  = QTCrystalDataset(val_ids,  inventory_df)
    test_dataset = QTCrystalDataset(test_ids, inventory_df)

    val_loader  = DataLoader(val_dataset,  batch_size=8, shuffle=False,
                             num_workers=0, collate_fn=qt_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False,
                             num_workers=0, collate_fn=qt_collate_fn)

    logging.getLogger(__name__).info("===== Val 采样 =====")
    val_preds = run_sampling(val_loader, model, "val")
    torch.save(val_preds, os.path.join(QT_STEP4_DIR, "predictions_val.pt"))
    logging.getLogger(__name__).info(f"已保存 val predictions（{len(val_preds)} 个化合物）")

    logging.getLogger(__name__).info("===== Test 采样 =====")
    test_preds = run_sampling(test_loader, model, "test")
    torch.save(test_preds, os.path.join(QT_STEP4_DIR, "predictions_test.pt"))
    logging.getLogger(__name__).info(f"已保存 test predictions（{len(test_preds)} 个化合物）")

    # sanity check
    for split, preds in [("val", val_preds), ("test", test_preds)]:
        mp_id  = next(iter(preds))
        sample = preds[mp_id]
        print(f"\n[{split}] 共 {len(preds)} 个化合物，示例 mp_id={mp_id}")
        print(f"  n_atoms              : {sample['n_atoms']}")
        print(f"  pred_frac_coords     : {sample['pred_frac_coords'].shape}")
        print(f"  pred_lengths         : {sample['pred_lengths'].tolist()}")
        print(f"  pred_angles          : {sample['pred_angles'].tolist()}")
        print(f"  gt_lengths           : {sample['gt_lengths'].tolist()}")

    print("\nStep 4.1_qt 完成。")


if __name__ == "__main__":
    main()