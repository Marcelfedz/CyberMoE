import torch
import torch.nn as nn
import torch.nn.functional as F


class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts, hidden_dims=[64, 32], dropout=0.2):
        super(GatingNetwork, self).__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dims[-1], num_experts))  # Output: [B, num_experts]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)  # Raw logits per expert [B, num_experts]