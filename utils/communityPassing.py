import torch
import torch.nn as nn
import torch.nn.functional as F


class CommunityPassing(nn.Module):
    def __init__(self, aggr: str = "mean", eps: float = 1e-12):
        super().__init__()
        if aggr not in {"mean", "sum"}:
            raise ValueError("aggr must be 'mean' or 'sum'")
        self.aggr = aggr
        self.eps = eps

    def forward(self, x: torch.Tensor, community: torch.Tensor) -> torch.Tensor:
        community = community.long()
        num_comms = int(community.max().item()) + 1 if community.numel() > 0 else 0
        if num_comms == 0:
            return x

        comm_sum = torch.zeros((num_comms, x.size(1)), device=x.device, dtype=x.dtype)
        comm_sum.index_add_(0, community, x)

        if self.aggr == "sum":
            comm_out = comm_sum
        else:
            comm_cnt = torch.zeros((num_comms,), device=x.device, dtype=x.dtype)
            comm_cnt.index_add_(0, community, torch.ones((x.size(0),), device=x.device, dtype=x.dtype))
            comm_out = comm_sum / comm_cnt.clamp_min(self.eps).unsqueeze(-1)

        return comm_out[community]
