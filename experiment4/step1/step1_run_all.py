# step1_run_all.py
# ------------------------------------------------------------
# Serial driver: runs step1_1 through step1_6 in order.
# Stops immediately if any script exits nonzero.
# Run from anywhere; assumes all step1_*.py are co-located.
# ------------------------------------------------------------

import os
import sys
import time
import subprocess

SCRIPTS = [
    "step1_1_scan_and_parse.py",
    "step1_2_filter_outliers.py",
    "step1_3_feff_imputation.py",
    "step1_4_split_stratified.py",
    "step1_5_fit_scaler.py",
    "step1_6_summary.py",
]

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    t0_all = time.time()
    for script in SCRIPTS:
        path = os.path.join(HERE, script)
        if not os.path.isfile(path):
            print(f"[DRIVER] MISSING: {path}")
            sys.exit(1)
        print("\n" + "#" * 72)
        print(f"# {script}")
        print("#" * 72)
        t0 = time.time()
        rc = subprocess.call([sys.executable, path])
        dt = time.time() - t0
        print(f"\n[DRIVER] {script}  rc={rc}  elapsed={dt:.1f}s")
        if rc != 0:
            print(f"[DRIVER] stop: nonzero rc from {script}")
            sys.exit(rc)

    print(f"\n[DRIVER] ALL SCRIPTS OK  total elapsed={time.time() - t0_all:.1f}s")


if __name__ == "__main__":
    main()
