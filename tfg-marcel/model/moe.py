import torch
import torch.nn as nn


class MoE(nn.Module):
    def __init__(self, experts, gating_network):
        super(MoE, self).__init__()
        self.experts = nn.ModuleList(experts)
        self.gating_network = gating_network

    def forward(self, x, top_k=2):
        gating_logits = self.gating_network(x)  # [B, num_experts]
        gating_weights = torch.softmax(gating_logits, dim=1)  # [B, num_experts]

        topk_vals, topk_indices = torch.topk(gating_weights, k=top_k, dim=1)
        mask = torch.zeros_like(gating_weights).scatter_(1, topk_indices, 1.0)

        gating_weights = gating_weights * mask
        gating_weights = gating_weights / gating_weights.sum(dim=1, keepdim=True).clamp(min=1e-6)

        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # [B, num_experts, output_dim]

        output = torch.sum(gating_weights.unsqueeze(2) * expert_outputs, dim=1)  # [B, output_dim]

        return output, gating_weights
