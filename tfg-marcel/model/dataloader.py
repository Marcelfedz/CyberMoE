import fireducks.pandas as pd
import torch
from torch.utils.data import Dataset

class TTPDataLoader(Dataset):
    def __init__(self, df, target_class_id):
        """
        Each expert is trained to detect only its target_class_id.
        All data is shown, but labels are 1 (target) or 0 (non-target).
        """
        self.X = df.drop(columns=["alert_label_encoded"]).values
        y_original = df["alert_label_encoded"].values

        # Binary labels: 1 if matches expert's class, else 0
        self.y = (y_original == target_class_id).astype(int)

        # Convert to tensors
        self.X_tensor = torch.tensor(self.X, dtype=torch.float32)
        self.y_tensor = torch.tensor(self.y, dtype=torch.float32)  # For BCEWithLogitsLoss

    def __len__(self):
        return len(self.X_tensor)

    def __getitem__(self, idx):
        return self.X_tensor[idx], self.y_tensor[idx]
