import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from gn import GatingNetwork
from fulldataset import FullDataset

# Configuration
EPOCHS = 30
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
INPUT_DIM = 7
NUM_EXPERTS = 5  # Update this to your actual number of experts
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load datasets ---
def prepare_dataset(path):
    df = pd.read_parquet(path)

    # Map each alert_label_encoded to a one-hot vector for expert responsibility
    y_onehot = torch.nn.functional.one_hot(torch.tensor(df["alert_label_encoded"].values), num_classes=NUM_EXPERTS)
    df = df.drop(columns=["alert_label_encoded"])
    df["onehot_label"] = list(y_onehot.numpy())  # Store multi-hot in one column

    return FullDataset(df)

train_dataset = prepare_dataset("data/df_train_d6.parquet")
val_dataset   = prepare_dataset("data/df_val_d6.parquet")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- Initialize GN ---
gating_net = GatingNetwork(input_dim=INPUT_DIM, num_experts=NUM_EXPERTS).to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(gating_net.parameters(), lr=LEARNING_RATE)

best_val_loss = float('inf')

# --- Training loop ---
for epoch in range(EPOCHS):
    gating_net.train()
    total_train_loss = 0.0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)  # Already one-hot [B, num_experts]

        logits = gating_net(X_batch)  # [B, num_experts]
        loss = criterion(logits, y_batch.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    # --- Validation ---
    gating_net.eval()
    total_val_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            logits = gating_net(X_batch)
            val_loss = criterion(logits, y_batch.float())
            total_val_loss += val_loss.item()

    avg_val_loss = total_val_loss / len(val_loader)

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(gating_net.state_dict(), "data/gating_network_d6.pt")

    print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

print("Training complete. Best model saved to data/gating_network_d6.pt")
