cat > /tmp/cps_patch.py << 'EOF'
import re

path = "/home/tcat/experiment6_v7/step3/step3.1_eval_full_val.py"
with open(path) as f:
    code = f.read()

old = '''            try:
                cps_full, bd = cs.composite_physical_score(
                    ppos_i, argmax_i, sample_name, lengths_i, IDX_TO_Z,
                )
                cps_bypass, bd2 = cs.composite_physical_score(
                    ppos_i, argmax_i, sample_name, lengths_i, IDX_TO_Z,
                    
                )
                pv_pass = bool(bd["PV"])
                # outside_shells_ratio: fraction of valid pred atoms
                # that fall outside both GT shell 1 and shell 2
                n_valid = int((argmax_i != NO_OBJECT_IDX).sum().item())
                in_shell = bd2.get("n_pred_in_any_shell", None)
                if in_shell is not None and n_valid > 0:
                    outside_ratio = 1.0 - in_shell / n_valid
                else:
                    outside_ratio = float("nan")
                all_cps.append(cps_full)
                all_cps_bypass.append(cps_bypass)
                all_pv_pass.append(float(pv_pass))
                all_outside_ratio.append(outside_ratio)
                all_C1.append(bd2.get("C1", 0.0))
                all_D1.append(bd2.get("D1", 0.0))
                all_T1.append(bd2.get("T1", 0.0))
                all_C2.append(bd2.get("C2", 0.0))
                all_D2.append(bd2.get("D2", 0.0))
                all_T2.append(bd2.get("T2", 0.0))'''

new = '''            try:
                cps_full, bd = cs.composite_physical_score(
                    ppos_i, argmax_i, sample_name, lengths_i, IDX_TO_Z,
                )
                pv_pass = bool(bd["PV"])
                n_valid = int(bd.get("pred_n_valid", 0))
                n_in_s1 = int(bd.get("pred_n_in_shell1", 0))
                n_in_s2 = int(bd.get("pred_n_in_shell2", 0))
                if n_valid > 0:
                    outside_ratio = 1.0 - (n_in_s1 + n_in_s2) / n_valid
                else:
                    outside_ratio = float("nan")
                all_cps.append(cps_full)
                all_cps_bypass.append(cps_full)  # no bypass_pv in this impl
                all_pv_pass.append(float(pv_pass))
                all_outside_ratio.append(outside_ratio)
                all_C1.append(bd.get("C1") or 0.0)
                all_D1.append(bd.get("D1") or 0.0)
                all_T1.append(bd.get("T1") or 0.0)
                all_C2.append(bd.get("C2") or 0.0)
                all_D2.append(bd.get("D2") or 0.0)
                all_T2.append(bd.get("T2") or 0.0)'''

assert old in code, "패턴을 찾을 수 없습니다 — 현재 파일 내용 확인 필요"
code = code.replace(old, new)
with open(path, "w") as f:
    f.write(code)
print("patch applied")
EOF
/home/tcat/conda_envs/mlff/bin/python /tmp/cps_patch.py