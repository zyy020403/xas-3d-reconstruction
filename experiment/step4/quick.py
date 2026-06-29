import glob, os
d = r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment\step4\finetune_output"
files = glob.glob(os.path.join(d, "*.ckpt"))
for f in files:
    print(os.path.basename(f))