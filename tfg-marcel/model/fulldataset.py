import fireducks.pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np

class FullDataset(Dataset):
    def __init__(self, df):
        if "onehot_label" in df.columns:
            self.X = df.drop(columns=["onehot_label"]).values
            self.y = np.stack(df["onehot_label"].values).astype(np.float32)
        else:
            self.X = df.drop(columns=["alert_label_encoded"]).values
            self.y = df["alert_label_encoded"].values.astype(np.float32)

        self.X_tensor = torch.tensor(self.X, dtype=torch.float32)
        self.y_tensor = torch.tensor(self.y, dtype=torch.float32)

    def __len__(self):
        return len(self.X_tensor)

    def __getitem__(self, idx):
        return self.X_tensor[idx], self.y_tensor[idx]
