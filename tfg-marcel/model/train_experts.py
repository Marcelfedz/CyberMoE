import torch
import torch.nn as nn
import torch.optim as optim
import fireducks.pandas as pd
import torch.optim as optim

from torch.utils.data import DataLoader
from tqdm import tqdm
from experts import Expert
from dataloader import TTPDataLoader

# Configuration
EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
INPUT_DIM = 7  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
df = pd.read_parquet("data/df_train_d6.parquet")

ttp_ids = [0, 1, 2, 3, 4]  

for expert_id, ttp_label in enumerate(ttp_ids):
    print(f"\n🔧 Training Expert {expert_id} for label {ttp_label}")

    # Filter dataset for this label
    df_filtered = df[df["alert_label_encoded"] == ttp_label]
    if df_filtered.empty:
        print(f"⚠️ Skipping expert {expert_id}: No data for label {ttp_label}")
        continue

    dataset = TTPDataLoader(df_filtered, target_class_id=ttp_label)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Expert is a binary classifier: learns to detect only its assigned label
    model = Expert(input_dim=INPUT_DIM, hidden_dims=[64, 32]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()  # Binary classification (logits)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for X_batch, _ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            # All targets are 1 (positive class) for this expert
            y_batch = torch.ones(X_batch.size(0), 1)  # shape: [B, 1]
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(X_batch)  # shape: [B, 1]
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X_batch.size(0)

        avg_loss = running_loss / len(dataset)
        print(f"📉 Expert {expert_id} - Epoch {epoch+1} Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), f"data/expert_{expert_id}_d6.pt")
    print(f"✅ Saved Expert {expert_id} model to expert_{expert_id}_d6.pt")
