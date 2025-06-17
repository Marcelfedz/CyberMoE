import torch
import torch.nn as nn
import fireducks.pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from moe import MoE
from gn import GatingNetwork
from experts import Expert  
from fulldataset import FullDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --- Hyperparameters ---
INPUT_DIM = 7  
NUM_EXPERTS = 5
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load test/unseen data ---
#df_test = pd.read_parquet("data/df_test_d6.parquet") 
df_test = pd.read_parquet("data/df_test_d6.parquet") 

test_dataset = FullDataset(df_test) 
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- Load pretrained experts ---
experts = []
for i in range(NUM_EXPERTS):
    expert = Expert(input_dim=INPUT_DIM).to(DEVICE)
    #expert.load_state_dict(torch.load(f'data/expert_{i}.pt', map_location=DEVICE))
    expert.load_state_dict(torch.load(f'data/expert_{i}_d6.pt', map_location=DEVICE))
    expert.eval()
    experts.append(expert)

# --- Load pretrained gating network ---
gating_net = GatingNetwork(input_dim=INPUT_DIM, num_experts=NUM_EXPERTS).to(DEVICE)
#gating_net.load_state_dict(torch.load('data/gating_network.pt', map_location=DEVICE))
gating_net.load_state_dict(torch.load('data/gating_network_d6.pt', map_location=DEVICE))

gating_net.eval()

# --- Initialize MoE ---
moe_model = MoE(experts, gating_net).to(DEVICE)
moe_model.eval()

# --- Evaluation ---
all_expert_outputs = []
all_gn_outputs = []
all_final_preds = []
all_y_true = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(DEVICE) 
        y_batch = y_batch.to(DEVICE)

        # Forward pass through MoE
        expert_logits, gn_logits = moe_model(X_batch)         # Both are [B, num_experts]
        expert_probs = torch.sigmoid(expert_logits)           # [B, num_experts]
        gn_probs = torch.sigmoid(gn_logits)                   # [B, num_experts]

        # Threshold both expert and GN outputs
        expert_active = (expert_probs > 0.5)                  # [B, num_experts]
        gn_active = (gn_probs > 0.5)                          # [B, num_experts]
        
        # Combine: if either the GN or any expert predicts active for any expert
        weighted_expert_probs = expert_probs * gn_probs  # shape: [B, num_experts]
        final_preds = weighted_expert_probs.argmax(dim=1)  # shape: [B]

        # Save for later evaluation
        all_expert_outputs.append(expert_probs.cpu())
        all_gn_outputs.append(gn_probs.cpu())
        all_final_preds.append(final_preds.cpu())
        all_y_true.append(y_batch.cpu())

# --- Concatenate results ---
all_expert_outputs = torch.cat(all_expert_outputs, dim=0).numpy()  # [N, num_experts]
all_gn_outputs = torch.cat(all_gn_outputs, dim=0).numpy()          # [N, num_experts]
all_final_preds = torch.cat(all_final_preds, dim=0).numpy()        # [N]

# --- Collect true labels ---
all_y_true = torch.cat(all_y_true, dim=0).numpy()  # [N]

print(np.unique(all_y_true))
print(np.unique(all_final_preds))

for i in range(10):
    print("True:", all_y_true[i], "Pred:", all_final_preds[i])

# --- Compute metrics ---
acc = accuracy_score(all_y_true, all_final_preds)
prec = precision_score(all_y_true, all_final_preds, average='micro')
rec = recall_score(all_y_true, all_final_preds, average='micro')
f1 = f1_score(all_y_true, all_final_preds, average='micro')

# --- Print metrics ---
print(f"\n--- Evaluation Metrics on Unseen Data ---")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")

print("Expert probs stats", expert_probs.mean(dim=0), expert_probs.std(dim=0))
print(classification_report(all_y_true, all_final_preds))

print("expert_probs.shape:", expert_probs.shape)
print("gn_probs.shape:", gn_probs.shape)

# --- Save  ---
np.save('data/expert_outputs_d6.npy', all_expert_outputs)
np.save('data/gn_outputs_d6.npy', all_gn_outputs)
np.save('data/final_preds_d6.npy', all_final_preds)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

os.makedirs("data/plots", exist_ok=True)

y_true = np.array(all_y_true)
y_pred = np.array(all_final_preds)

classes = np.unique(y_true)
num_classes = len(classes)

true_counts = np.zeros(num_classes)
correct_counts = np.zeros(num_classes)

for idx, cls in enumerate(classes):
    true_counts[idx] = np.sum(y_true == cls)
    correct_counts[idx] = np.sum((y_true == cls) & (y_pred == cls))

accuracy_per_class = correct_counts / true_counts


num_classes = len(true_counts)
classes = np.arange(num_classes)

x = np.arange(num_classes)
width = 0.35

plt.figure(figsize=(10, 6))
bars1 = plt.bar(x - width/2, true_counts, width, label='Total ground truth', color='skyblue')
bars2 = plt.bar(x + width/2, correct_counts, width, label='Correct Predictions', color='lightgreen')

# add black line in the graph
for i in range(num_classes):
    left = x[i] - width
    right = x[i] + width
    y = true_counts[i]
    plt.plot([left, right], [y, y], color='black', linewidth=1.2)

plt.xticks(x, [f'Label {int(cls)}' for cls in classes])
plt.xlabel('Label')
plt.ylabel('Number of samples')
plt.title('Comparing Ground truth with Prediction Label')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("data/plots/stadistics.png")
plt.show()