import pandas as pd
from sklearn.model_selection import train_test_split

# === Config ===
INPUT_FILE = "dataset_6_clean.parquet"  
#INPUT_FILE = "dataset_6_clean.parquet"  
LABEL_COL = "alert_label_encoded"
TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_STATE = 42  

# === Load data ===
df = pd.read_parquet(INPUT_FILE)
print(f"Loaded dataset with {len(df)} samples.")

# === First split: train vs temp (val + test) ===
df_train, df_temp = train_test_split(
    df,
    test_size=VAL_SIZE + TEST_SIZE,
    stratify=df[LABEL_COL],
    random_state=RANDOM_STATE
)

# === Second split: val vs test ===
val_ratio = VAL_SIZE / (VAL_SIZE + TEST_SIZE)  # e.g., 0.15 / 0.30 = 0.5
df_val, df_test = train_test_split(
    df_temp,
    test_size=1 - val_ratio,
    stratify=df_temp[LABEL_COL],
    random_state=RANDOM_STATE
)

# === Save splits ===
df_train.to_parquet("data/df_train_d6.parquet", index=False)
df_val.to_parquet("data/df_val_d6.parquet", index=False)
df_test.to_parquet("data/df_test_d6.parquet", index=False)

print(f"✅ Done! Saved splits:")
print(f"   ➤ Train: {len(df_train)} samples")
print(f"   ➤ Val:   {len(df_val)} samples")
print(f"   ➤ Test:  {len(df_test)} samples")

# Separate benign and malicious logs
df_test_benign = df_test[df_test[LABEL_COL] == 0]
df_test_malicious = df_test[df_test[LABEL_COL] != 0]

# Downsample malicious logs to simulate realistic conditions (e.g., keep 20%)
df_test_malicious_downsampled = df_test_malicious.sample(
    frac=0.1, random_state=RANDOM_STATE)

# Recombine
df_test_realistic = pd.concat([df_test_benign, df_test_malicious_downsampled], ignore_index=True)

# Optionally shuffle the test set
df_test_realistic = df_test_realistic.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

# Save updated test set
df_test_realistic.to_parquet("data/df_test_d6_realistic.parquet", index=False)

print(f"   ➤ Test realistic: {len(df_test_realistic)} samples")
print(f"      (Benign: {len(df_test_benign)}, Malicious: {len(df_test_malicious_downsampled)})")
