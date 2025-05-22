import pandas as pd
import numpy as np

# 1. Load
file_path = r"neural_networks\rect_section_n1\dataset\dataset_rect_n1_test5_100k.parquet"
df = pd.read_parquet(file_path)

# 2. Define features
features = ["b", "h", "d", "fi", "fck", "ro1", "MRd", "Wk", "Mcr", "Cost"]

# 3. Compute summary statistics + z-outlier counts
records = []
for feat in features:
    col = df[feat].dropna()
    μ, σ = col.mean(), col.std()
    # IQR
    Q1, Q3 = col.quantile([0.25, 0.75])
    iqr = Q3 - Q1
    # z-scores
    zs = (col - μ) / σ
    z_out_cnt = np.sum(np.abs(zs) > 3)
    z_out_pct = 100 * z_out_cnt / col.shape[0]
    records.append({
        "feature":           feat,
        "skewness":          col.skew(),
        "kurtosis":          col.kurtosis(),
        "IQR":               iqr,
        "mean":              μ,
        "std":               σ,
        "z>3σ count":        z_out_cnt,
        "z>3σ percent (%)":  z_out_pct
    })

stats_df = pd.DataFrame(records).set_index("feature")

# 4. Save everything to Excel
with pd.ExcelWriter("feature_statistics.xlsx") as writer:
    stats_df.to_excel(writer, sheet_name="summary")
    
    # (Optional) full z-score matrix
    zscores = (df[features] - stats_df["mean"]) / stats_df["std"]
    zscores.to_excel(writer, sheet_name="z_scores")

print("Done!  • Summary + z>3σ outlier counts written to sheet ‘summary’ of feature_statistics.xlsx")
