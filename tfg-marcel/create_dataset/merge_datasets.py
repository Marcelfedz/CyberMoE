import pandas as pd
import glob

def merge_parquet_files(parquet_paths):
    dfs = []
    for path in parquet_paths:
        df = pd.read_parquet(path)
        dfs.append(df)
    merged_df = pd.concat(dfs, ignore_index=True)
    return merged_df

def undersample_benign_logs(df, drop_fraction=0.9):
    malicious_df = df[df['alert_label'].notna()]
    benign_df = df[df['alert_label'].isna()]
    
    benign_sampled_df = benign_df.sample(frac=(1 - drop_fraction), random_state=42)
    balanced_df = pd.concat([malicious_df, benign_sampled_df], ignore_index=True)
    
    return balanced_df

parquet_files = [
    "parquet_files/label_data/29oct2024_labeled.parquet",# 
    "parquet_files/label_data/4oct2024_labeled.parquet", # -
    "parquet_files/label_data/4nov2024_labeled.parquet", # -
    "parquet_files/label_data/8nov2024_labeled.parquet", # -
    "parquet_files/label_data/20feb2025_labeled.parquet", # -
    "parquet_files/label_data/28may2025_labeled.parquet", #
    "parquet_files/label_data/29may2025_labeled.parquet", #
    "parquet_files/label_data/25dec2024_labeled.parquet", # -
]

df_merged = merge_parquet_files(parquet_files)
df_balanced = undersample_benign_logs(df_merged, drop_fraction=0.997)

df_balanced.to_parquet("parquet_files/dataset_6.parquet", index=False)