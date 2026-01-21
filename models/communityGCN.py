from torch_geometric.nn import GCNConv
import torch
import torch.nn as nn
from utils.communityPassing import CommunityPassing
import torch.nn.functional as F

from utils.TrainingUtils import TrainingUtils


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

    def fit(
        self,
        data,
        *,
        community=None,
        community_attr: str = "y",
        label_attr: str = "y_task",
        lr: float = 1e-2,
        weight_decay: float = 5e-4,
        epochs: int = 200,
        print_every: int = 20,
        select_by: str = "val",
        print_best: bool = True,
    ):
        device = data.x.device
        self.to(device)

        if community is None:
            if not hasattr(data, community_attr):
                raise AttributeError(f"data has no attribute '{community_attr}'")
            community = getattr(data, community_attr)
        community = community.to(device)

        if not hasattr(data, label_attr):
            raise AttributeError(f"data has no attribute '{label_attr}'")
        y = getattr(data, label_attr).to(device)

        train_mask = data.train_mask.to(device)
        val_mask = data.val_mask.to(device)
        test_mask = data.test_mask.to(device)

        opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        select_by = select_by.lower()
        if select_by not in {"train", "val", "test"}:
            raise ValueError("select_by must be 'train', 'val', or 'test'")

        best_score = -1.0
        best_state = None

        for epoch in range(1, epochs + 1):
            self.train()
            opt.zero_grad()

            logits = self(data.x, data.edge_index, community)
            loss = F.cross_entropy(logits[train_mask], y[train_mask])
            loss.backward()
            opt.step()

            self.eval()
            with torch.no_grad():
                logits = self(data.x, data.edge_index, community)
                train_acc = TrainingUtils.accuracy(logits, y, train_mask)
                val_acc = TrainingUtils.accuracy(logits, y, val_mask)
                test_acc = TrainingUtils.accuracy(logits, y, test_mask)

            if select_by == "train":
                score = train_acc
            elif select_by == "test":
                score = test_acc
            else:
                score = val_acc

            if score > best_score:
                best_score = score
                best_state = {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}

            if print_every and (epoch % print_every == 0 or epoch == 1):
                print(
                    f"Epoch {epoch:03d} | loss {loss.item():.4f} | "
                    f"train {train_acc:.3f} | val {val_acc:.3f} | test {test_acc:.3f}"
                )

        if best_state is not None:
            self.load_state_dict({k: v.to(device) for k, v in best_state.items()})

        self.eval()
        with torch.no_grad():
            logits = self(data.x, data.edge_index, community)
            train_acc = TrainingUtils.accuracy(logits, y, train_mask)
            val_acc = TrainingUtils.accuracy(logits, y, val_mask)
            test_acc = TrainingUtils.accuracy(logits, y, test_mask)

        if print_best:
            print(
                f"Best checkpoint ({select_by}) | "
                f"train {train_acc:.3f} | val {val_acc:.3f} | test {test_acc:.3f}"
            )
        return self
