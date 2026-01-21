import torch


class TrainingUtils:
    @staticmethod
    def make_splits(num_nodes: int, train_ratio=0.6, val_ratio=0.2, seed=0, device="cpu"):
        g = torch.Generator(device=device).manual_seed(seed)
        perm = torch.randperm(num_nodes, generator=g, device=device)

        n_train = int(train_ratio * num_nodes)
        n_val = int(val_ratio * num_nodes)

        train_idx = perm[:n_train]
        val_idx = perm[n_train:n_train + n_val]
        test_idx = perm[n_train + n_val:]

        train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)

        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

        return train_mask, val_mask, test_mask

    @staticmethod
    @torch.no_grad()
    def accuracy(logits, y, mask):
        pred = logits.argmax(dim=-1)
        return (pred[mask] == y[mask]).float().mean().item()
