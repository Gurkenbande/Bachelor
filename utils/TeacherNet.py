import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class TeacherNet(nn.Module):
    def __init__(self, in_channels, hidden=32, num_classes=3):
        super().__init__()
        self.gcn1 = GCNConv(in_channels, hidden)
        self.gcn2 = GCNConv(hidden, hidden)
        self.out = GCNConv(hidden, num_classes)

    def forward(self, x, edge_index):
        h = torch.tanh(self.gcn1(x, edge_index))
        h = torch.tanh(self.gcn2(h, edge_index))
        return self.out(h, edge_index)
