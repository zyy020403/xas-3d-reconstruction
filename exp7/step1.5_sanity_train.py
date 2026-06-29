cat > /home/tcat/experiment7/step1/step1.5_sanity_train.py << 'EOF'
"""
Step 1.5 — 10 epoch sanity training
Config: batch_size=32, n_critic=5, G_lr=1e-4, D_lr=4e-4
Curriculum: epoch 0-9 全在 Phase 0, min_pdist = 0.33 × 1.5076 = 0.4975
"""
import sys, json, pickle, torch, torch.nn as nn, numpy as np
from torch.utils.data import DataLoader
sys.path.insert(0, '/home/tcat/experiment7/shared')

from cond_wgan_gp import (LocalStructureGenerator, LocalStructureDiscriminator,
                           compute_gradient_penalty, pairwise_min_distance_penalty,
                           LAMBDA_GP, N_ATOMS, N_NEIGHBOR_TYPES, NOISE_DIM,
                           SPECTRUM_DIM, CENTER_EMB_DIM, N_TYPES_WITH_NO_OBJ)
from curriculum_callbacks import get_curriculum_min_pdist
from spectrum_encoder import SpectrumEncoder
from xas_local_dataset_v2 import XasLocalDatasetV2

device = torch.device('cuda:0')
MIN_PDIST_CAL = 1.5075718402862548
LAMBDA_PMIN   = 1.0
LAMBDA_CE     = 0.1
MAX_EPOCHS    = 10
BATCH_SIZE    = 32
N_CRITIC      = 5

# ── Dataset ──────────────────────────────────────────────────────────────────
ds_train = XasLocalDatasetV2(split='train',
                              data_dir='/home/tcat/diffcsp_exp5_prime/data',
                              verbose_init_benchmark=False)
ds_val   = XasLocalDatasetV2(split='val',
                              data_dir='/home/tcat/diffcsp_exp5_prime/data',
                              verbose_init_benchmark=False)

def collate_fn(batch):
    result = {}
    for k in batch[0].keys():
        v0 = batch[0][k]
        if isinstance(v0, torch.Tensor):
            result[k] = torch.stack([s[k] for s in batch])
        elif isinstance(v0, bool):
            result[k] = torch.tensor([s[k] for s in batch], dtype=torch.bool)
        elif isinstance(v0, int):
            result[k] = torch.tensor([s[k] for s in batch], dtype=torch.long)
        elif isinstance(v0, float):
            result[k] = torch.tensor([s[k] for s in batch], dtype=torch.float32)
        else:
            result[k] = [s[k] for s in batch]  # str 等保持 list
    return result

train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, collate_fn=collate_fn, drop_last=True)
val_loader   = DataLoader(ds_val,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, collate_fn=collate_fn, drop_last=False)

# ── Models ────────────────────────────────────────────────────────────────────
enc = SpectrumEncoder().to(device)
G   = LocalStructureGenerator().to(device)
D   = LocalStructureDiscriminator().to(device)

opt_G = torch.optim.Adam(list(enc.parameters()) + list(G.parameters()),
                          lr=1e-4, betas=(0.0, 0.9))
opt_D = torch.optim.Adam(D.parameters(), lr=4e-4, betas=(0.0, 0.9))

# ── Vocab ─────────────────────────────────────────────────────────────────────
vocab = json.load(open('/home/tcat/experiment7/shared/exp7_element_vocab.json'))
Z_to_idx = {int(k): int(v) for k,v in vocab['neighbor']['Z_to_idx'].items()}
NO_OBJ = 88

def build_D_inputs(frac_coords, atom_types_Z):
    cart = frac_coords * 20.0
    diff = cart.unsqueeze(2) - cart.unsqueeze(1)
    dist_mat    = diff.norm(dim=-1)
    dist_center = cart.norm(dim=-1)
    idx = atom_types_Z.clone()
    for b in range(idx.shape[0]):
        for n in range(idx.shape[1]):
            z = int(idx[b,n].item())
            idx[b,n] = Z_to_idx.get(z, NO_OBJ)
    idx_c = idx.clamp(0, N_NEIGHBOR_TYPES-1)
    types_oh = torch.zeros(*idx_c.shape, N_NEIGHBOR_TYPES, device=frac_coords.device)
    types_oh.scatter_(2, idx_c.unsqueeze(-1), 1.0)
    return dist_mat, dist_center, types_oh

def build_fake_D_inputs(pred_coords, pred_logits):
    cart = pred_coords * 20.0
    diff = cart.unsqueeze(2) - cart.unsqueeze(1)
    dist_mat    = diff.norm(dim=-1)
    dist_center = cart.norm(dim=-1)
    fake_idx = pred_logits.argmax(dim=-1).clamp(0, N_NEIGHBOR_TYPES-1)
    types_oh = torch.zeros(*fake_idx.shape, N_NEIGHBOR_TYPES, device=pred_coords.device)
    types_oh.scatter_(2, fake_idx.unsqueeze(-1), 1.0)
    return dist_mat, dist_center, types_oh

def gt_types_to_idx(atom_types_Z):
    idx = atom_types_Z.clone()
    for b in range(idx.shape[0]):
        for n in range(idx.shape[1]):
            z = int(idx[b,n].item())
            idx[b,n] = Z_to_idx.get(z, NO_OBJ)
    return idx.clamp(0, N_TYPES_WITH_NO_OBJ-1)

# ── Training loop ─────────────────────────────────────────────────────────────
print("Starting 10-epoch sanity training...")
print(f"{'Ep':>3} {'D_critic':>9} {'D_gp':>7} {'G_adv':>7} {'G_pmin':>7} {'G_ce':>7} {'cur_pdist':>10} {'mode_div':>9}")

for epoch in range(MAX_EPOCHS):
    curriculum_pdist = get_curriculum_min_pdist(epoch, MIN_PDIST_CAL)
    enc.train(); G.train(); D.train()

    d_critic_ep, d_gp_ep, g_adv_ep, g_pmin_ep, g_ce_ep = [], [], [], [], []
    batch_iter = iter(train_loader)
    n_batches  = len(train_loader) // (N_CRITIC + 1)

    for _ in range(n_batches):
        # D step × N_CRITIC
        for _ in range(N_CRITIC):
            try:
                batch = next(batch_iter)
            except StopIteration:
                break
            xmu  = batch['xmu'].to(device)
            chi1 = batch['chi1'].to(device)
            feff = batch['feff'].to(device)
            fc   = batch['frac_coords'].to(device)
            at   = batch['atom_types'].to(device)
            cz   = batch['center_element_Z'].to(device)

            with torch.no_grad():
                enc_out = enc(xmu, chi1, feff, cz)
            spec_lat = enc_out[:, :SPECTRUM_DIM]
            cen_emb  = enc_out[:, SPECTRUM_DIM:]

            z = torch.randn(xmu.shape[0], NOISE_DIM, device=device)
            fake_coords, fake_logits = G(z, spec_lat, cen_emb)

            real_dm, real_dc, real_to = build_D_inputs(fc, at)
            fake_dm, fake_dc, fake_to = build_fake_D_inputs(
                fake_coords.detach(), fake_logits.detach())

            d_real = D(real_dm, real_dc, real_to, spec_lat.detach())
            d_fake = D(fake_dm, fake_dc, fake_to, spec_lat.detach())
            gp     = compute_gradient_penalty(D, real_dm, real_dc, real_to,
                                               fake_dm, fake_dc, fake_to,
                                               spec_lat.detach(), device)
            d_loss = d_fake.mean() - d_real.mean() + LAMBDA_GP * gp
            opt_D.zero_grad(); d_loss.backward(); opt_D.step()
            d_critic_ep.append((d_fake.mean() - d_real.mean()).item())
            d_gp_ep.append(gp.item())

        # G step × 1
        try:
            batch = next(batch_iter)
        except StopIteration:
            break
        xmu  = batch['xmu'].to(device)
        chi1 = batch['chi1'].to(device)
        feff = batch['feff'].to(device)
        fc   = batch['frac_coords'].to(device)
        at   = batch['atom_types'].to(device)
        cz   = batch['center_element_Z'].to(device)

        enc_out  = enc(xmu, chi1, feff, cz)
        spec_lat = enc_out[:, :SPECTRUM_DIM]
        cen_emb  = enc_out[:, SPECTRUM_DIM:]

        z = torch.randn(xmu.shape[0], NOISE_DIM, device=device)
        fake_coords, fake_logits = G(z, spec_lat, cen_emb)
        fake_dm, fake_dc, fake_to = build_fake_D_inputs(fake_coords, fake_logits)
        d_fake_g = D(fake_dm, fake_dc, fake_to, spec_lat)

        g_adv  = -d_fake_g.mean()
        g_pmin = pairwise_min_distance_penalty(fake_coords, curriculum_pdist) * LAMBDA_PMIN
        gt_idx = gt_types_to_idx(at)
        g_ce   = nn.functional.cross_entropy(
                     fake_logits.view(-1, N_TYPES_WITH_NO_OBJ),
                     gt_idx.view(-1)) * LAMBDA_CE
        g_loss = g_adv + g_pmin + g_ce
        opt_G.zero_grad(); g_loss.backward(); opt_G.step()
        g_adv_ep.append(g_adv.item())
        g_pmin_ep.append(g_pmin.item())
        g_ce_ep.append(g_ce.item())

    # mode diversity: 1 spectrum × 4 z noise
    enc.eval(); G.eval()
    with torch.no_grad():
        s    = ds_train[0]
        xmu0 = s['xmu'].unsqueeze(0).expand(4,-1).to(device)
        chi0 = s['chi1'].unsqueeze(0).expand(4,-1).to(device)
        ff0  = s['feff'].unsqueeze(0).expand(4,-1).to(device)
        cz0  = torch.tensor([s['center_element_Z']]).expand(4).to(device)
        eo   = enc(xmu0, chi0, ff0, cz0)
        sl   = eo[:, :SPECTRUM_DIM]; ce = eo[:, SPECTRUM_DIM:]
        zz   = torch.randn(4, NOISE_DIM, device=device)
        coords_s, _ = G(zz, sl, ce)
        mode_div = coords_s.std(dim=0).mean().item()

    all_vals = d_critic_ep + d_gp_ep + g_adv_ep + g_pmin_ep + g_ce_ep
    if any(np.isnan(v) for v in all_vals):
        print(f"Ep {epoch}: NaN detected — RAISE MA1"); break

    print(f"{epoch:>3} {np.mean(d_critic_ep):>9.4f} {np.mean(d_gp_ep):>7.4f} "
          f"{np.mean(g_adv_ep):>7.4f} {np.mean(g_pmin_ep):>7.4f} "
          f"{np.mean(g_ce_ep):>7.4f} {curriculum_pdist:>10.4f} {mode_div:>9.5f}")

print("\n10-epoch sanity training complete.")
print("curriculum_pdist epoch 0-9 should all be 0.4975 (Phase 0)")
EOF

python3 /home/tcat/experiment7/step1/step1.5_sanity_train.py 2>&1