import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from utils.TrainingUtils import TrainingUtils


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

    def fit(
        self,
        data,
        *,
        label_attr: str = "y_task",
        lr: float = 1e-2,
        weight_decay: float = 5e-4,
        epochs: int = 200,
        print_every: int = 20,
        select_by: str = "val",
        print_best: bool = True,
        return_history: bool = False,
    ):
        device = data.x.device
        self.to(device)

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

        history = {"train": [], "val": [], "test": []} if return_history else None

        for epoch in range(1, epochs + 1):
            self.train()
            opt.zero_grad()

            logits = self(data.x, data.edge_index)
            loss = F.cross_entropy(logits[train_mask], y[train_mask])
            loss.backward()
            opt.step()

            self.eval()
            with torch.no_grad():
                logits = self(data.x, data.edge_index)
                train_acc = TrainingUtils.accuracy(logits, y, train_mask)
                val_acc = TrainingUtils.accuracy(logits, y, val_mask)
                test_acc = TrainingUtils.accuracy(logits, y, test_mask)

            if history is not None:
                history["train"].append(train_acc)
                history["val"].append(val_acc)
                history["test"].append(test_acc)

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
            logits = self(data.x, data.edge_index)
            train_acc = TrainingUtils.accuracy(logits, y, train_mask)
            val_acc = TrainingUtils.accuracy(logits, y, val_mask)
            test_acc = TrainingUtils.accuracy(logits, y, test_mask)

        if print_best:
            print(
                f"Best checkpoint ({select_by}) | "
                f"train {train_acc:.3f} | val {val_acc:.3f} | test {test_acc:.3f}"
            )
        if return_history:
            return self, history
        return self
