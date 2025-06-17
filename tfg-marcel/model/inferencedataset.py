import fireducks.pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np

class InferenceDataset(Dataset):
    def __init__(self, df):
        self.X = df.drop(columns=["alert_label_encoded"]).values
        self.y = df["alert_label_encoded"].values 

        self.X_tensor = torch.tensor(self.X, dtype=torch.float32)
        self.y_tensor = torch.tensor(self.y, dtype=torch.float32)

    def __len__(self):
        return len(self.X_tensor)

    def __getitem__(self, idx):
        return self.X_tensor[idx], self.y_tensor[idx]
