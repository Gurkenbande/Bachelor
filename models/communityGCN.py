from torch_geometric.nn import GCNConv
import torch
import torch.nn as nn
from utils.communityPassing import CommunityPassing
import torch.nn.functional as F


class CommunityGCN(nn.Module):
    def __init__(self, in_channels: int, hidden: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.cp = CommunityPassing(aggr="mean")

        self.lin_in = nn.Linear(in_channels * 2, hidden)

        self.gcn1 = GCNConv(hidden, hidden)
        self.gcn2 = GCNConv(hidden, num_classes)

        self.dropout = dropout

    def forward(self, x, edge_index, community):
        x_comm = self.cp(x, community)
        h = torch.cat([x, x_comm], dim=-1)
        h = F.relu(self.lin_in(h))
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = F.relu(self.gcn1(h, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)

        out = self.gcn2(h, edge_index)
        return out
