# save as: check_poscar.py, 在本地跑
import os, random
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

POSCAR_DIR = r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment4\data\POSCAR_zip\MP_all_POSCAR_flat"

files = [f for f in os.listdir(POSCAR_DIR) if f.endswith("_POSCAR")]
print(f"总 POSCAR 数: {len(files)}")

random.seed(42)
samples = random.sample(files, min(20, len(files)))

print(f"\n{'文件名':<30} {'原子数':>8} {'is_primitive?':>15} {'空间群':>20}")
print("-" * 80)
n_prim = 0
for f in samples:
    try:
        s = Structure.from_file(os.path.join(POSCAR_DIR, f))
        sga = SpacegroupAnalyzer(s, symprec=0.1)
        prim = sga.get_primitive_standard_structure()
        is_prim = (len(s) == len(prim))
        if is_prim: n_prim += 1
        print(f"{f:<30} {len(s):>8d} {str(is_prim):>15} {sga.get_space_group_symbol():>20}")
    except Exception as e:
        print(f"{f:<30} ERROR: {e}")
print(f"\n{n_prim}/{len(samples)} 已经是 primitive")