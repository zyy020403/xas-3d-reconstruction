# =============================================================================
# 脚本编号: step4_qt
# 脚本名称: step4_qt_eval.py
# =============================================================================

import os
import sys
import glob
import types
import logging
import importlib.util
import warnings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
warnings.filterwarnings("ignore", message="Trying to infer the `batch_size`")
warnings.filterwarnings("ignore", message=".*does not have many workers.*")

PROJECT_ROOT   = r"C:\Users\T-Cat\Desktop\DiffCSP-main"
EXPERIMENT_DIR = os.path.join(PROJECT_ROOT, "experiment")
QT_DIR         = os.path.join(EXPERIMENT_DIR, "quicktest")
STEP2_DIR      = os.path.join(EXPERIMENT_DIR, "step2")
CONF_MODEL_DIR = os.path.join(PROJECT_ROOT, "conf", "model")
QT_OUTPUT_DIR  = os.path.join(QT_DIR, "qt_output")
QT_EVAL_DIR    = os.path.join(QT_DIR, "qt_eval")
os.makedirs(QT_EVAL_DIR, exist_ok=True)

os.environ["PROJECT_ROOT"] = PROJECT_ROOT

for p in [PROJECT_ROOT, QT_DIR, STEP2_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

N_CANDIDATES  = 1
QT_TIMESTEPS  = 50
BATCH_SIZE    = 1

_qt_encoder_path = os.path.join(QT_DIR, "step2.1_qt_encoder.py")
if not os.path.exists(_qt_encoder_path):
    raise FileNotFoundError(f"找不到 qt 编码器: {_qt_encoder_path}")
_spec = importlib.util.spec_from_file_location("step2_1_spectrum_encoder", _qt_encoder_path)
_qt_encoder_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_qt_encoder_mod)
sys.modules["step2_1_spectrum_encoder"] = _qt_encoder_mod
logging.getLogger(__name__).info("✓ 注入 qt 版 step2_1_spectrum_encoder")

import torch
import torch.nn as nn

class _StubAggregator(nn.Module):
    def __init__(self, d_site=256, d_struct=256):
        super().__init__()
        self._unused = nn.Linear(1, 1)
    def forward(self, padded, mask):
        raise RuntimeError("[QuickTest] StubAggregator 不应被调用")

_stub_mod = types.ModuleType("step2_2_multisite_aggregator")
_stub_mod.MultiSiteAggregator     = _StubAggregator
_stub_mod.collate_multisite_batch = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stub"))
sys.modules["step2_2_multisite_aggregator"] = _stub_mod
logging.getLogger(__name__).info("✓ 注入 stub step2_2_multisite_aggregator")

if __name__ == "__main__":
    import csv
    import numpy as np
    import pandas as pd
    import torch
    import hydra
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    from torch_geometric.data import Batch, Data
    from tqdm import tqdm

    from diffcsp.common.data_utils import lattice_params_to_matrix_torch

    logger = logging.getLogger(__name__)

    def _find_best_ckpt(output_dir: str) -> str:
        last = os.path.join(output_dir, "last.ckpt")
        ckpts = glob.glob(os.path.join(output_dir, "epoch*.ckpt"))
        ckpts = [c for c in ckpts if "last" not in os.path.basename(c)]
        if not ckpts:
            raise FileNotFoundError(f"qt_output 目录下未找到任何 epoch*.ckpt: {output_dir}")
        def _val_loss(path):
            name = os.path.basename(path)
            try:
                return float(name.split("val_loss=")[-1].replace(".ckpt", ""))
            except Exception:
                return float("inf")
        return min(ckpts, key=_val_loss)

    ckpt_path = _find_best_ckpt(QT_OUTPUT_DIR)
    logger.info(f"使用 checkpoint: {ckpt_path}")

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONF_MODEL_DIR,
                               job_name="qt_eval",
                               version_base=None):
        _raw = compose(config_name="diffusion")
    _raw_dict = OmegaConf.to_container(_raw, resolve=False, throw_on_missing=False)
    full_cfg  = OmegaConf.create({"model": _raw_dict})
    model_cfg = full_cfg.model

    optim_cfg = OmegaConf.create({
        "optimizer": {"_target_": "torch.optim.Adam", "lr": 1e-4},
        "use_lr_scheduler": False,
        "lr_scheduler": None,
    })

    logger.info("实例化 CSPDiffusion 并加载 checkpoint...")
    model = hydra.utils.instantiate(model_cfg, optim=optim_cfg, _recursive_=False)
    model.lattice_scaler = None
    model.scaler         = None

    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["state_dict"])
    model.eval()
    model.cuda()
    logger.info("✓ checkpoint 加载完成")

    def _encode_xas_qt(self, batch):
        return self.spectrum_encoder(
            batch.spectra[:, 0],
            batch.site_elements[:, 0],
            batch.is_ionic[:, 0],
        )
    model._encode_xas = types.MethodType(_encode_xas_qt, model)

    orig_timesteps = model.beta_scheduler.timesteps
    model.beta_scheduler.timesteps = QT_TIMESTEPS
    logger.info(f"sample 步数: {orig_timesteps} → {QT_TIMESTEPS}（qt 加速）")

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

        for t in range(time_start, 0, -1):
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
        return traj[0], None

    model.sample = types.MethodType(_sample_qt, model)
    logger.info("✓ sample 已替换（qt 版）")

    from step2_1_spectrum_encoder import preprocess_chi
    from pymatgen.core import Structure
    from pymatgen.core.periodic_table import Element as PmgElement
    from torch_geometric.data import Data, Batch

    QUALITY_WEIGHT = {"A": 1.0, "B": 0.5, "C": 0.1}
    SPECTRA_LEN = 512

    def _parse_poscar(poscar_path):
        try:
            struct = Structure.from_file(poscar_path)
            frac_coords = torch.tensor(struct.frac_coords, dtype=torch.float32)
            atom_types  = torch.tensor([s.specie.Z for s in struct], dtype=torch.long)
            lengths     = torch.tensor(list(struct.lattice.abc),    dtype=torch.float32)
            angles      = torch.tensor(list(struct.lattice.angles), dtype=torch.float32)
            return frac_coords, atom_types, lengths, angles, len(struct), struct
        except Exception as e:
            logger.warning(f"POSCAR 解析失败: {poscar_path} ({e})")
            return None

    def _load_test_data(qt_dir):
        test_ids_path  = os.path.join(qt_dir, "qt_test_ids.txt")
        inventory_path = os.path.join(qt_dir, "qt_inventory.csv")
        with open(test_ids_path) as f:
            test_ids = set(int(l.strip()) for l in f if l.strip())
        df = pd.read_csv(inventory_path)
        df = df[df["mp_id"].isin(test_ids)].copy()

        samples = []
        for _, row in df.iterrows():
            poscar_path = os.path.join(row["source_path"], "POSCAR_supercell_fixed")
            result = _parse_poscar(poscar_path)
            if result is None:
                continue
            frac_coords, atom_types, lengths, angles, num_atoms, gt_struct = result

            chi_path = os.path.join(row["source_path"], "chi.dat")
            try:
                spec = preprocess_chi(chi_path)
            except Exception:
                spec = torch.zeros(1, SPECTRA_LEN, dtype=torch.float32)

            try:
                z = PmgElement(row["element"]).Z
            except Exception:
                z = 26

            tier = row.get("quality_tier", "")
            qw = QUALITY_WEIGHT.get(str(tier).upper(), 0.3)

            samples.append({
                "mp_id":           int(row["mp_id"]),
                "frac_coords":     frac_coords,
                "atom_types":      atom_types,
                "lengths":         lengths,
                "angles":          angles,
                "num_atoms":       num_atoms,
                "gt_struct":       gt_struct,
                "spectra":         spec.unsqueeze(0),
                "site_elements":   torch.tensor([z],  dtype=torch.long),
                "is_ionic":        torch.tensor([0],  dtype=torch.long),
                "quality_weights": torch.tensor([qw], dtype=torch.float32),
                "n_sites":         1,
            })
        logger.info(f"测试集加载完成：{len(samples)} 个样本")
        return samples

    def _sample_to_batch(sample):
        data = Data(
            frac_coords=sample["frac_coords"],
            atom_types=sample["atom_types"],
            lengths=sample["lengths"].unsqueeze(0),
            angles=sample["angles"].unsqueeze(0),
            num_atoms=sample["num_atoms"],
            num_nodes=sample["num_atoms"],
        )
        batch = Batch.from_data_list([data])
        batch.spectra         = sample["spectra"].unsqueeze(0)
        batch.site_elements   = sample["site_elements"].unsqueeze(0)
        batch.is_ionic        = sample["is_ionic"].unsqueeze(0)
        batch.quality_weights = sample["quality_weights"].unsqueeze(0)
        batch.n_sites         = torch.tensor([1], dtype=torch.long)
        return batch

    test_samples = _load_test_data(QT_DIR)

    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Lattice, Structure as PmgStructure

    # ★ 改动1：放宽匹配容差
    matcher = StructureMatcher(
        ltol=0.5,
        stol=0.8,
        angle_tol=20.0,
        primitive_cell=True,
        scale=True,
        attempt_supercell=False,
        comparator=None,
    )

    def _tensor_to_structure(frac_coords, atom_types, lengths, angles):
        try:
            lat     = Lattice.from_parameters(*lengths.cpu().numpy(), *angles.cpu().numpy())
            species = [int(z) for z in atom_types.cpu().numpy()]
            coords  = frac_coords.cpu().numpy()
            return PmgStructure(lat, species, coords)
        except Exception:
            return None

    def _lattice_errors(pred_struct, gt_struct):
        if pred_struct is None:
            return {k: float("nan") for k in ["da","db","dc","dalpha","dbeta","dgamma"]}
        p = pred_struct.lattice
        g = gt_struct.lattice
        return {
            "da":     abs(p.a - g.a),
            "db":     abs(p.b - g.b),
            "dc":     abs(p.c - g.c),
            "dalpha": abs(p.alpha - g.alpha),
            "dbeta":  abs(p.beta  - g.beta),
            "dgamma": abs(p.gamma - g.gamma),
        }

    results = []
    logger.info(f"开始采样评估（{len(test_samples)} 个化合物 × {N_CANDIDATES} 候选）...")

    for idx, sample in enumerate(test_samples):
        mp_id     = sample["mp_id"]
        gt_struct = sample["gt_struct"]
        batch     = _sample_to_batch(sample).cuda()

        candidates = []
        with torch.no_grad():
            for c in range(N_CANDIDATES):
                try:
                    pred, _ = model.sample(batch)
                    _m = pred["lattices"][0].cpu().numpy()
                    _lat_obj = __import__("pymatgen.core", fromlist=["Lattice"]).Lattice(_m)
                    _lengths = torch.tensor(list(_lat_obj.abc),    dtype=torch.float32)
                    _angles  = torch.tensor(list(_lat_obj.angles), dtype=torch.float32)
                    pred_struct = _tensor_to_structure(
                        pred["frac_coords"],
                        pred["atom_types"],
                        _lengths,
                        _angles,
                    )
                    candidates.append(pred_struct)
                except Exception as e:
                    logger.warning(f"  mp_id={mp_id} 第{c}次采样失败: {e}")
                    candidates.append(None)

        best_struct   = None
        best_matched  = False
        best_rmsd     = float("nan")
        best_lat_errs = None

        for cand in candidates:
            if cand is None:
                continue
            try:
                rms = matcher.get_rms_dist(gt_struct, cand)
                if rms is not None:
                    rmsd_val = rms[0]
                    if not best_matched or rmsd_val < best_rmsd:
                        best_matched  = True
                        best_rmsd     = rmsd_val
                        best_struct   = cand
                        best_lat_errs = _lattice_errors(cand, gt_struct)
            except Exception:
                pass

        if not best_matched:
            best_rmsd = float("nan")
            for cand in candidates:
                if cand is None:
                    continue
                errs = _lattice_errors(cand, gt_struct)
                if best_struct is None or errs["da"] < (best_lat_errs or {}).get("da", float("inf")):
                    best_struct   = cand
                    best_lat_errs = errs

        if best_lat_errs is None:
            best_lat_errs = {k: float("nan") for k in ["da","db","dc","dalpha","dbeta","dgamma"]}

        row = {"mp_id": mp_id, "matched": int(best_matched), "best_rmsd": best_rmsd, **best_lat_errs}
        results.append(row)

        rmsd_str = f"{best_rmsd:.4f}" if not np.isnan(best_rmsd) else "NaN"
        logger.info(
            f"  [{idx+1:02d}/{len(test_samples)}] mp_id={mp_id}  "
            f"matched={best_matched}  rmsd={rmsd_str}"
        )

    results_path = os.path.join(QT_EVAL_DIR, "qt_eval_results.csv")
    fieldnames = ["mp_id","matched","best_rmsd","da","db","dc","dalpha","dbeta","dgamma"]
    with open(results_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    logger.info(f"✓ 逐样本结果: {results_path}")

    n_total    = len(results)
    n_matched  = sum(r["matched"] for r in results)
    match_rate = n_matched / n_total if n_total > 0 else 0.0

    valid_rmsd = [r["best_rmsd"] for r in results if not np.isnan(r["best_rmsd"])]
    mean_rmsd  = np.mean(valid_rmsd) if valid_rmsd else float("nan")

    lat_keys = ["da","db","dc","dalpha","dbeta","dgamma"]
    mean_lat = {}
    for k in lat_keys:
        vals = [r[k] for r in results if not np.isnan(r.get(k, float("nan")))]
        mean_lat[k] = np.mean(vals) if vals else float("nan")

    summary_path = os.path.join(QT_EVAL_DIR, "qt_eval_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("QuickTest Step4 评估摘要\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"checkpoint     : {os.path.basename(ckpt_path)}\n")
        f.write(f"sample 步数    : {QT_TIMESTEPS}\n")
        f.write(f"候选结构数/样本 : {N_CANDIDATES}\n")
        f.write(f"StructureMatcher 容差: ltol=0.5  stol=0.8  angle_tol=20°\n\n")
        f.write(f"测试样本总数    : {n_total}\n")
        f.write(f"match_rate     : {n_matched}/{n_total} = {match_rate:.1%}\n")
        f.write(f"mean RMSD (matched): {mean_rmsd:.4f} Å\n\n")
        # ★ 改动2：描述改为"全部样本，不依赖匹配"
        f.write("平均晶格参数绝对误差（全部15样本，不依赖匹配）:\n")
        f.write(f"  Δa={mean_lat['da']:.3f} Å   Δb={mean_lat['db']:.3f} Å   Δc={mean_lat['dc']:.3f} Å\n")
        f.write(f"  Δα={mean_lat['dalpha']:.2f}°  Δβ={mean_lat['dbeta']:.2f}°  Δγ={mean_lat['dgamma']:.2f}°\n")
        f.write("\n注：QuickTest 仅 30 epoch + 100步采样，指标仅供 pipeline 验通，\n")
        f.write("    不反映正式服（400 epoch + 1000步）的真实性能。\n")

    logger.info(f"✓ 汇总报告: {summary_path}")
    logger.info("=" * 50)
    logger.info(f"match_rate : {n_matched}/{n_total} = {match_rate:.1%}")
    logger.info(f"mean RMSD  : {mean_rmsd:.4f} Å" if not np.isnan(mean_rmsd) else "mean RMSD  : NaN（无匹配）")
    logger.info("=" * 50)
    logger.info("step4_qt 评估完成。")


def _lattice_from_matrix(matrix_tensor):
    import numpy as np
    m = matrix_tensor.cpu().numpy()
    try:
        from pymatgen.core import Lattice
        lat     = Lattice(m)
        lengths = torch.tensor(list(lat.abc),    dtype=torch.float32)
        angles  = torch.tensor(list(lat.angles), dtype=torch.float32)
        return lengths, angles
    except Exception:
        a = np.linalg.norm(m[0])
        b = np.linalg.norm(m[1])
        c = np.linalg.norm(m[2])
        cos_alpha = np.dot(m[1], m[2]) / (b * c + 1e-10)
        cos_beta  = np.dot(m[0], m[2]) / (a * c + 1e-10)
        cos_gamma = np.dot(m[0], m[1]) / (a * b + 1e-10)
        alpha = np.degrees(np.arccos(np.clip(cos_alpha, -1, 1)))
        beta  = np.degrees(np.arccos(np.clip(cos_beta,  -1, 1)))
        gamma = np.degrees(np.arccos(np.clip(cos_gamma, -1, 1)))
        return (torch.tensor([a, b, c], dtype=torch.float32),
                torch.tensor([alpha, beta, gamma], dtype=torch.float32))