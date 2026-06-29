# save as: check_feff_outliers.py
import pandas as pd
import numpy as np

FEFF_CSV = r"C:\Users\T-Cat\Desktop\DiffCSP-main\experiment4\data\feff_features_all_csv_75cols(in).csv"
df = pd.read_csv(FEFF_CSV)

# 1. mu_at_E0 明显异常的样本
bad_mu = df[(df['mu_at_E0'].abs() > 10) | (df['mu_at_E0'] < 0)]
print(f"mu_at_E0 异常（|值|>10 或 <0）的行数: {len(bad_mu)} / {len(df)} ({100*len(bad_mu)/len(df):.2f}%)")
print("前 5 个异常样本的 sample_name 和 mu_at_E0:")
print(bad_mu[['sample_name', 'mu_at_E0', 'E0']].head())

# 2. 用四分位距方法找离群点
print(f'\n各数值列的"极端异常行数"(|z-score by IQR| > 50, 非常宽容的阈值):')
num_df = df.select_dtypes(include='number')
extreme_count = {}
for col in num_df.columns:
    s = num_df[col].dropna()
    if len(s) == 0: continue
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0: continue
    n_extreme = ((s - s.median()).abs() > 50 * iqr).sum()
    if n_extreme > 0:
        extreme_count[col] = n_extreme
for c, n in sorted(extreme_count.items(), key=lambda x: -x[1])[:15]:
    print(f"  {c}: {n} 行")

# 3. 统计"任意特征列极端异常"的样本总数
any_extreme = np.zeros(len(df), dtype=bool)
for col in num_df.columns:
    s = num_df[col]
    if s.isna().all(): continue
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0 or np.isnan(iqr): continue
    any_extreme |= ((s - s.median()).abs() > 50 * iqr).fillna(False)
print(f"\n至少一列极端异常的样本总数: {any_extreme.sum()} / {len(df)}")
print(f"这些异常样本的元素分布 (Top 10):")
bad_df = df[any_extreme].copy()
bad_df['elem'] = bad_df['sample_name'].str.extract(r'-EXAFS-([A-Z][a-z]?)-K')
print(bad_df['elem'].value_counts().head(10))