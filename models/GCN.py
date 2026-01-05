import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    def __init__(self, in_channels: int, hidden: int, num_classes: int, dropout: float = 0.5):
        super().__init__()
        self.gcn1 = GCNConv(in_channels, hidden)
        self.gcn2 = GCNConv(hidden, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, _unused=None):
        h = self.gcn1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        out = self.gcn2(h, edge_index)
        return out
