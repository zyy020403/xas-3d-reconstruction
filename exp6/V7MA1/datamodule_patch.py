"""
Patch experiment6_v7/shared/xas_local_datamodule_v2.py:
Add shell_starts / shell_ends / shell_n_atoms to _dict_to_pyg_data.
These are list[float/int] stored as Python lists on Data objects;
adapter reads them via pyg_batch.shell_starts[i].

Run from /home/tcat/experiment6_v7:
  /home/tcat/conda_envs/mlff/bin/python /home/tcat/datamodule_patch.py
"""
from pathlib import Path

SRC = Path("/home/tcat/experiment6_v7/shared/xas_local_datamodule_v2.py")
text = SRC.read_text()

OLD = '    data.sample_name          = s["sample_name"]\n    [data.mp](http://data.mp)_id                = s["mp_id"]\n    [data.center](http://data.center)_element       = s["center_element"]\n    [data.site](http://data.site)_equivalence_tag = s["site_equivalence_tag"]\n    return data'

# Handle both rendered (with hyperlinks) and plain text versions
OLD_PLAIN = '    data.sample_name          = s["sample_name"]\n    data.mp_id                = s["mp_id"]\n    data.center_element       = s["center_element"]\n    data.site_equivalence_tag = s["site_equivalence_tag"]\n    return data'

NEW = '''    data.sample_name          = s["sample_name"]
    data.mp_id                = s["mp_id"]
    data.center_element       = s["center_element"]
    data.site_equivalence_tag = s["site_equivalence_tag"]
    # v7 shell fields — list[float/int], len <= 2 (first 2 GT shells)
    data.shell_starts  = s.get("shell_starts",  [])
    data.shell_ends    = s.get("shell_ends",    [])
    data.shell_n_atoms = s.get("shell_n_atoms", [])
    return data'''

if OLD in text:
    text = text.replace(OLD, NEW, 1)
    print("Patched (hyperlink variant)")
elif OLD_PLAIN in text:
    text = text.replace(OLD_PLAIN, NEW, 1)
    print("Patched (plain variant)")
else:
    # Try to find the return data line after site_equivalence_tag
    import re
    pattern = r'(    data\.site_equivalence_tag\s*=\s*s\["site_equivalence_tag"\]\s*\n)(    return data)'
    replacement = (r'\1'
                   '    # v7 shell fields — list[float/int], len <= 2 (first 2 GT shells)\n'
                   '    data.shell_starts  = s.get("shell_starts",  [])\n'
                   '    data.shell_ends    = s.get("shell_ends",    [])\n'
                   '    data.shell_n_atoms = s.get("shell_n_atoms", [])\n'
                   r'\2')
    new_text, n = re.subn(pattern, replacement, text)
    if n == 1:
        text = new_text
        print("Patched (regex fallback)")
    else:
        print("ERROR: no anchor found — check file manually")
        import sys; sys.exit(1)

SRC.write_text(text)
print(f"Written {SRC}")

t2 = SRC.read_text()
for c in ["shell_starts", "shell_ends", "shell_n_atoms", "v7 shell fields"]:
    print(f"  {'OK' if c in t2 else 'MISSING'} {c}")
