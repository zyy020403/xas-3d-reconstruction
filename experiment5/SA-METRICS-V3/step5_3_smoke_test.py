"""Smoke test for step5_3_composite_score.py — dummy data, no torch deps required for sb."""
import sys
sys.path.insert(0, '/home/claude')

import numpy as np
import torch
from step5_3_composite_score import (
    compute_min_pairwise, assign_pred_shells,
    score_coord_n, score_distance, score_element, cno_token,
    compute_one_sample, MIN_PAIRWISE_DIST, SHELL_GAP_THRESHOLD,
    W_SHELL1, W_SHELL2,
)

print(f'Constants: MIN_PAIRWISE={MIN_PAIRWISE_DIST}, GAP={SHELL_GAP_THRESHOLD}, '
      f'W1={W_SHELL1}, W2={W_SHELL2}, max_total={3*W_SHELL1 + 3*W_SHELL2}')
print()

# ----------------------------------------------------------------------------
# Test 1: scoring function unit tests
# ----------------------------------------------------------------------------
print('=== Test 1: scoring functions ===')
# coord_n: tol=1.5
assert score_coord_n(4, 4, 1.5) == 1.0
assert score_coord_n(5, 4, 1.5) == 1.0      # delta=1 ≤ tol
# delta=2, tol=1.5, falls in else: 1 - (2-1.5)/3 = 1 - 0.5/3 = 0.8333
assert abs(score_coord_n(6, 4, 1.5) - 0.8333333) < 1e-5
print(f'  coord_n(6,4,1.5) = {score_coord_n(6, 4, 1.5):.4f} (expect 0.8333)')
print(f'  coord_n(10,4,1.5) = {score_coord_n(10, 4, 1.5):.4f} (expect 0.0, since 1-(6-1.5)/3 = -0.5)')
assert score_coord_n(10, 4, 1.5) == 0.0  # max(0, ...) clip

# distance: tol=0.2
assert score_distance(2.0, 2.0) == 1.0
assert score_distance(2.15, 2.0) == 1.0     # delta=0.15 ≤ tol=0.2
print(f'  distance(2.5, 2.0) = {score_distance(2.5, 2.0):.4f} (expect 0.4: 1-(0.5-0.2)/0.5)')
assert abs(score_distance(2.5, 2.0) - 0.4) < 1e-9
assert score_distance(None, 2.0) == 0.0
assert score_distance(2.0, None) == 0.0

# CNO equivalence
assert cno_token(6) == -1   # C
assert cno_token(7) == -1   # N
assert cno_token(8) == -1   # O
assert cno_token(53) == 53  # I
assert cno_token(31) == 31  # Ga

# element with CNO
pred_Z = torch.tensor([6, 7, 53], dtype=torch.int64)   # C, N, I
true_Z = torch.tensor([8, 8, 53], dtype=torch.int64)   # O, O, I
# tokens: pred=[-1,-1,53], true=[-1,-1,53] → perfect match
assert score_element(pred_Z, true_Z) == 1.0
print(f'  element(C+N+I vs O+O+I) = {score_element(pred_Z, true_Z):.4f} (expect 1.0 via CNO eq)')

# element partial
pred_Z = torch.tensor([6, 53], dtype=torch.int64)
true_Z = torch.tensor([6, 6], dtype=torch.int64)
# pred tokens=[-1, 53] count={-1:1, 53:1}; true tokens=[-1,-1] count={-1:2}
# inter = min(1,2) for -1 + min(1,0) for 53 = 1; total = max(2, 2) = 2 → 0.5
assert score_element(pred_Z, true_Z) == 0.5
print(f'  element(C+I vs C+C) = {score_element(pred_Z, true_Z):.4f} (expect 0.5)')

# element with None (shell missing)
assert score_element(None, true_Z) == 0.0
assert score_element(pred_Z, None) == 0.0
print('  PASS')
print()

# ----------------------------------------------------------------------------
# Test 2: gate behavior (min_d < 1.5 → total = 0)
# ----------------------------------------------------------------------------
print('=== Test 2: gate fail → total = 0 (sample[0] mp-10009 mimic) ===')
# Real sample[0] data from probe: pred[0] cart radial 1.61–4.19, min_d = 0.357
pred_fc = torch.tensor([
    [-0.2341,  0.4805,  0.1400],   # close to atom 1 (d=0.357 in real probe)
    [ 0.3196, -0.4773,  0.3699],
    [-0.2328, -0.1285,  0.4956],
] + [[0.4, 0.4, 0.4]] * 17, dtype=torch.float32)
pred_at = torch.tensor([31] * 20, dtype=torch.int64)  # all Ga (Z=31)
L = 6.0
xyz = pred_fc * L
md = compute_min_pairwise(xyz)
print(f'  dummy min_d = {md:.4f} (should be < 1.5 → gate fail)')

# minimal sb_i mock
sb_i = {
    'threshold':     0.1563,
    'distances':     np.array([2.44, 2.68, 2.68, 2.68, 4.09, 4.09], dtype=np.float32),
    'species_Z':     np.array([31, 52, 52, 52, 31, 31], dtype=np.int8),
    'shell_starts':  np.zeros(20, dtype=np.float32),
    'shell_ends':    np.zeros(20, dtype=np.float32),
    'shell_n_atoms': np.array([1, 3, 6] + [0] * 17, dtype=np.int32),
    'shell_of_atom': np.array([0, 1, 1, 1, 2, 2], dtype=np.int32),
    'eval_cutoff':   4.890556335449219,
    'n_center_sites': 1,
}
row = compute_one_sample('mp-test-1', pred_fc, pred_at, 4.890556335449219, L, sb_i)
print(f'  row: {row}')
assert row['gate_pass'] == 0
assert row['total_score'] == 0.0
assert row['score_shell1_coord'] == 0.0  # gate fail → all 6 sub-scores = 0
print('  PASS')
print()

# ----------------------------------------------------------------------------
# Test 3: gate pass + perfect match → total = 3*W_SHELL1 + 3*W_SHELL2 = 0.9
# ----------------------------------------------------------------------------
print('=== Test 3: gate pass + perfect match → total ≈ 0.9 ===')
# Construct pred with 1 atom at shell-1 (d=2.44, Z=31) and 3 at shell-2 (d=2.68, Z=52)
# All other 16 atoms parked far away with mutual spacing > 1.5
pred_fc_perfect = torch.zeros(20, 3, dtype=torch.float32)
# Atom 0: r=2.44/6 = 0.4067 along x
pred_fc_perfect[0] = torch.tensor([2.44 / 6.0, 0.0, 0.0])
# Atoms 1, 2, 3: r=2.68/6 along orthogonal axes
pred_fc_perfect[1] = torch.tensor([0.0, 2.68 / 6.0, 0.0])
pred_fc_perfect[2] = torch.tensor([0.0, 0.0, 2.68 / 6.0])
pred_fc_perfect[3] = torch.tensor([0.0, -2.68 / 6.0, 0.0])
# Park atoms 4-19 outside eval_cutoff (4.89/6 = 0.815 → place at 5.0/6 = 0.833 with separation)
# Use a 4x4 grid in a thin slice far from origin (radial > 4.89 will be filtered out)
# But gate uses pre-cutoff! So they need pairwise > 1.5 too.
# Place 16 atoms in an 8-spaced cube far from origin (will be filtered for shells but counted for gate)
spacing = 2.0
positions_far = []
for i in range(4):
    for j in range(4):
        # offset to keep min_d > 1.5: x in [10, 16], y in [0, 6], z = 0
        x_frac = (10.0 + i * spacing) / 6.0
        y_frac = (j * spacing) / 6.0 + 0.01 * i
        positions_far.append([x_frac, y_frac, 0.0])
for k, pos in enumerate(positions_far):
    pred_fc_perfect[4 + k] = torch.tensor(pos)

pred_at_perfect = torch.tensor([31] + [52, 52, 52] + [99] * 16, dtype=torch.int64)
xyz_p = pred_fc_perfect * 6.0
md_p = compute_min_pairwise(xyz_p)
print(f'  perfect dummy min_d = {md_p:.4f} (should be > 1.5)')

row_p = compute_one_sample('mp-test-2', pred_fc_perfect, pred_at_perfect, 4.890556335449219, 6.0, sb_i)
print(f'  row: gate={row_p["gate_pass"]}, n_pred_shells={row_p["n_pred_shells"]}, '
      f'min_d={row_p["min_d"]:.4f}')
print(f'    s1n={row_p["score_shell1_coord"]} s1d={row_p["score_shell1_dist"]} '
      f's1e={row_p["score_shell1_elem"]}')
print(f'    s2n={row_p["score_shell2_coord"]} s2d={row_p["score_shell2_dist"]} '
      f's2e={row_p["score_shell2_elem"]}')
print(f'    total={row_p["total_score"]:.4f} (expect ≈ 0.9 if perfect match)')
print()

# ----------------------------------------------------------------------------
# Test 4: missing shell-2 in true → score_distance/element return 0
# ----------------------------------------------------------------------------
print('=== Test 4: true shell-2 missing → s2d=0, s2e=0, s2n by spec ===')
sb_no_s2 = dict(sb_i)
sb_no_s2['shell_n_atoms'] = np.array([1, 0, 0] + [0] * 17, dtype=np.int32)
sb_no_s2['shell_of_atom'] = np.array([0, 0, 0, 0, 0, 0], dtype=np.int32)
sb_no_s2['distances']     = np.array([2.44, 2.45, 2.46, 2.47, 2.48, 2.49], dtype=np.float32)

row_ns2 = compute_one_sample('mp-test-3', pred_fc_perfect, pred_at_perfect,
                              4.890556335449219, 6.0, sb_no_s2)
print(f'  gate={row_ns2["gate_pass"]}, n_pred_shells={row_ns2["n_pred_shells"]}')
print(f'  s2n={row_ns2["score_shell2_coord"]} (pred has shell-2, true=0, delta likely > tol)')
print(f'  s2d={row_ns2["score_shell2_dist"]} (true=None → 0)')
print(f'  s2e={row_ns2["score_shell2_elem"]} (true=None → 0)')
assert row_ns2['score_shell2_dist'] == 0.0
assert row_ns2['score_shell2_elem'] == 0.0
print('  PASS')
print()

# ----------------------------------------------------------------------------
# Test 5: eval_cutoff mismatch → raise
# ----------------------------------------------------------------------------
print('=== Test 5: eval_cutoff mismatch → RuntimeError ===')
try:
    compute_one_sample('mp-test-4', pred_fc_perfect, pred_at_perfect,
                       4.50, 6.0, sb_i)  # pred_eval_cutoff=4.50, sb=4.89
    print('  FAIL: should have raised')
except RuntimeError as e:
    print(f'  PASS: raised "{e}"')
print()

print('=== ALL SMOKE TESTS PASSED ===')
