# save as: check_feff_nan.py
import pandas as pd
FEFF_CSV = r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment4\data\feff_features_all_csv_75cols(in).csv"
df = pd.read_csv(FEFF_CSV)
print(f"总行数: {len(df)}, 总列数: {len(df.columns)}")
print(f"\n各列 NaN 计数（只显示有 NaN 的列）:")
nan_cnt = df.isna().sum()
print(nan_cnt[nan_cnt > 0])
print(f"\n数值列的 min/max（看有没有 -999 类哨兵值）:")
numeric_cols = df.select_dtypes(include='number').columns
stat = df[numeric_cols].describe().T[['min', 'max']]
print(stat)
print(f"\n任意列 = -999 的行数: {(df[numeric_cols] == -999).any(axis=1).sum()}")