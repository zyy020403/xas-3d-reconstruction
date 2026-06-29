cat >> /home/tcat/experiment7/shared/cond_wgan_gp.py << 'WGAN_EOF'


# ── WGANGPModule (pytorch_lightning) ─────────────────────────────────────────
import pytorch_lightning as pl

class WGANGPModule(pl.LightningModule):

    def __init__(self, encoder, generator, discriminator,
                 calibrated_min_pdist, lambda_gp, lambda_pmin, lambda_type,
                 n_critic, noise_dim, spectrum_dim, center_dim,
                 curriculum_boundaries, curriculum_fractions,
                 eval_cps_fn, eval_step5_3_fn, shell_boundaries_path,
                 dual_delta_raise_threshold=0.50):
        super().__init__()
        self.encoder = encoder
        self.generator = generator
        self.discriminator = discriminator
        self.calibrated_min_pdist = calibrated_min_pdist
        self.lambda_gp = lambda_gp
        self.lambda_pmin = lambda_pmin
        self.lambda_type = lambda_type
        self.n_critic = n_critic
        self.noise_dim = noise_dim
        self.spectrum_dim = spectrum_dim
        self.center_dim = center_dim
        self.curriculum_boundaries = curriculum_boundaries
        self.curriculum_fractions = curriculum_fractions
        self.eval_cps_fn = eval_cps_fn
        self.eval_step5_3_fn = eval_step5_3_fn
        self.dual_delta_raise_threshold = dual_delta_raise_threshold
        self.automatic_optimization = False
        import pickle
        with open(shell_boundaries_path, 'rb') as f:
            self.shell_boundaries = pickle.load(f)
        self._current_min_pdist = calibrated_min_pdist * curriculum_fractions[0]
        self._fixed_5_spectra = None

    def configure_optimizers(self):
        opt_D = torch.optim.Adam(self.discriminator.parameters(),
                                 lr=4e-4, betas=(0.0, 0.9))
        opt_G = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.generator.parameters()),
            lr=1e-4, betas=(0.0, 0.9))
        return [opt_D, opt_G]

    def on_train_epoch_start(self):
        from curriculum_callbacks import get_curriculum_min_pdist
        ep = self.current_epoch
        new_pdist = get_curriculum_min_pdist(ep, self.calibrated_min_pdist)
        self._current_min_pdist = new_pdist
        if ep in [0] + self.curriculum_boundaries:
            phase = sum(1 for b in self.curriculum_boundaries if ep >= b)
            print(f"\n[Curriculum] epoch={ep} Phase {phase}: "
                  f"min_pdist={new_pdist:.4f} Å "
                  f"({self.curriculum_fractions[phase]:.2f} × {self.calibrated_min_pdist:.4f})")
        self.log('train_curriculum_min_pdist', new_pdist)

    def training_step(self, batch, batch_idx):
        opt_D, opt_G = self.optimizers()
        xmu  = batch['xmu'].to(self.device)
        chi1 = batch['chi1'].to(self.device)
        feff = batch['feff'].to(self.device)
        fc   = batch['frac_coords'].to(self.device)
        at   = batch['atom_types'].to(self.device)
        cz   = batch['center_element_Z'].to(self.device)

        enc_out  = self.encoder(xmu, chi1, feff, cz)
        spec_lat = enc_out[:, :self.spectrum_dim]
        cen_emb  = enc_out[:, self.spectrum_dim:]

        # ── D steps ──
        for _ in range(self.n_critic):
            z = torch.randn(fc.shape[0], self.noise_dim, device=self.device)
            with torch.no_grad():
                fake_coords, fake_logits = self.generator(z, spec_lat, cen_emb)
            real_dm, real_dc, real_to = self._build_real(fc, at)
            fake_dm, fake_dc, fake_to = self._build_fake(fake_coords, fake_logits)
            d_real = self.discriminator(real_dm, real_dc, real_to, spec_lat.detach())
            d_fake = self.discriminator(fake_dm, fake_dc, fake_to, spec_lat.detach())
            gp = compute_gradient_penalty(
                self.discriminator, real_dm, real_dc, real_to,
                fake_dm, fake_dc, fake_to, spec_lat.detach(), self.device)
            loss_D = d_fake.mean() - d_real.mean() + self.lambda_gp * gp
            opt_D.zero_grad(); self.manual_backward(loss_D); opt_D.step()
            self.log_dict({'D_critic': (d_fake.mean()-d_real.mean()),
                           'D_gp': gp}, on_step=False, on_epoch=True)

        # ── G step ──
        z = torch.randn(fc.shape[0], self.noise_dim, device=self.device)
        fake_coords, fake_logits = self.generator(z, spec_lat, cen_emb)
        fake_dm, fake_dc, fake_to = self._build_fake(fake_coords, fake_logits)
        d_fake_g = self.discriminator(fake_dm, fake_dc, fake_to, spec_lat)
        loss_adv  = -d_fake_g.mean()
        loss_pmin = pairwise_min_distance_penalty(fake_coords, self._current_min_pdist) * self.lambda_pmin
        loss_type = torch.nn.functional.cross_entropy(
            fake_logits.view(-1, fake_logits.shape[-1]),
            self._gt_type_idx(at).view(-1)) * self.lambda_type
        loss_G = loss_adv + loss_pmin + loss_type
        opt_G.zero_grad(); self.manual_backward(loss_G); opt_G.step()
        self.log_dict({'G_adv': loss_adv, 'G_pmin': loss_pmin,
                       'G_type_ce': loss_type}, on_step=False, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        xmu  = batch['xmu'].to(self.device)
        chi1 = batch['chi1'].to(self.device)
        feff = batch['feff'].to(self.device)
        fc   = batch['frac_coords'].to(self.device)
        at   = batch['atom_types'].to(self.device)
        cz   = batch['center_element_Z'].to(self.device)
        enc_out  = self.encoder(xmu, chi1, feff, cz)
        spec_lat = enc_out[:, :self.spectrum_dim]
        cen_emb  = enc_out[:, self.spectrum_dim:]
        z = torch.randn(spec_lat.shape[0], self.noise_dim, device=self.device)
        with torch.no_grad():
            pred_coords, pred_types = self.generator(z, spec_lat, cen_emb)
        cps_scores   = self.eval_cps_fn(pred_coords, pred_types, batch)
        step5_scores = self.eval_step5_3_fn(pred_coords, pred_types, batch)
        cps_mean     = cps_scores['cps_mean']
        s5_composite = step5_scores['composite']
        delta = abs(cps_mean - s5_composite) / max(cps_mean, s5_composite, 1e-8)
        if delta > self.dual_delta_raise_threshold:
            print(f"\n[DUAL EVAL RAISE] epoch={self.current_epoch} "
                  f"delta={delta:.1%} CPS={cps_mean:.4f} step5_3={s5_composite:.4f}")
        ckpt_score = 0.7 * cps_mean + 0.3 * s5_composite
        self.log('val_composite_ckpt_score', ckpt_score)
        self.log_dict({
            'val_cps_mean': cps_mean,
            'val_pv_pass_rate': cps_scores['pv_pass_rate'],
            'val_step5_3_composite': s5_composite,
            'val_step5_3_gate_pass_rate': step5_scores['gate_pass_rate'],
            'val_step5_3_collapse_rate': step5_scores['collapse_rate'],
            'val_dual_delta_relative': delta,
        })

    def on_validation_epoch_end(self):
        if self.current_epoch % 5 == 0:
            self._log_mode_diversity()

    def _log_mode_diversity(self):
        if self._fixed_5_spectra is None:
            return
        diversities = []
        for spec_lat, cen_emb in self._fixed_5_spectra:
            preds = []
            for _ in range(10):
                z = torch.randn(1, self.noise_dim, device=self.device)
                with torch.no_grad():
                    coords, _ = self.generator(z, spec_lat, cen_emb)
                preds.append(coords)
            preds = torch.cat(preds, dim=0)
            div = preds.std(dim=0).mean().item()
            diversities.append(div)
        mean_div = sum(diversities) / len(diversities)
        self.log('mode_diversity_per_spectrum', mean_div)
        if mean_div < 0.005 and self.current_epoch <= 30:
            print(f"\n[MODE COLLAPSE WARNING] epoch={self.current_epoch} "
                  f"mode_diversity={mean_div:.5f} < 0.005 — RAISE MA1")
        if mean_div < 0.003 and self.current_epoch <= 50:
            print(f"\n[MODE COLLAPSE RAISE] epoch={self.current_epoch} "
                  f"mode_diversity={mean_div:.5f} — RAISE MA1 立即")

    def _build_real(self, frac_coords, atom_types_Z):
        import json
        if not hasattr(self, '_Z_to_idx'):
            vocab = json.load(open('/home/tcat/experiment7/shared/exp7_element_vocab.json'))
            self._Z_to_idx = {int(k): int(v) for k,v in vocab['neighbor']['Z_to_idx'].items()}
        cart = frac_coords * 20.0
        diff = cart.unsqueeze(2) - cart.unsqueeze(1)
        dist_mat    = diff.norm(dim=-1)
        dist_center = cart.norm(dim=-1)
        N_TYPES = 89
        idx = atom_types_Z.clone()
        for b in range(idx.shape[0]):
            for n in range(idx.shape[1]):
                z = int(idx[b,n].item())
                idx[b,n] = self._Z_to_idx.get(z, 88)
        idx = idx.clamp(0, N_TYPES-1)
        types_oh = torch.zeros(*idx.shape, N_TYPES, device=frac_coords.device)
        types_oh.scatter_(2, idx.unsqueeze(-1), 1.0)
        return dist_mat, dist_center, types_oh

    def _build_fake(self, pred_coords, pred_logits):
        cart = pred_coords * 20.0
        diff = cart.unsqueeze(2) - cart.unsqueeze(1)
        dist_mat    = diff.norm(dim=-1)
        dist_center = cart.norm(dim=-1)
        N_TYPES = pred_logits.shape[-1]
        fake_idx = pred_logits.argmax(dim=-1).clamp(0, N_TYPES-1)
        types_oh = torch.zeros(*fake_idx.shape, N_TYPES, device=pred_coords.device)
        types_oh.scatter_(2, fake_idx.unsqueeze(-1), 1.0)
        return dist_mat, dist_center, types_oh

    def _gt_type_idx(self, atom_types_Z):
        if not hasattr(self, '_Z_to_idx'):
            import json
            vocab = json.load(open('/home/tcat/experiment7/shared/exp7_element_vocab.json'))
            self._Z_to_idx = {int(k): int(v) for k,v in vocab['neighbor']['Z_to_idx'].items()}
        idx = atom_types_Z.clone()
        for b in range(idx.shape[0]):
            for n in range(idx.shape[1]):
                z = int(idx[b,n].item())
                idx[b,n] = self._Z_to_idx.get(z, 88)
        return idx.clamp(0, 88)
WGAN_EOF