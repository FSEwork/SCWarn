import torch.nn as nn
import torch


class CorrLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, H1, H2):
        h1, h2 = torch.sum(H1, 1), torch.sum(H2, 1)
        h1_mean, h2_mean = torch.mean(h1), torch.mean(h2)
        h1_v, h2_v = h1 - h1_mean, h2 - h2_mean
        cost = torch.sum(h1_v * h2_v) / (torch.sqrt(torch.sum(h1_v**2) * torch.sum(h2_v**2)) + 0.001)
        return cost