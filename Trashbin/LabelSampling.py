import torch
import torch.nn as nn


class LabelSampling(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        seed: int = 0,
        noise_std: float = 0.0,
        standardize: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.noise_std = noise_std
        self.standardize = standardize
        self.eps = eps

        gen = torch.Generator(device="cpu").manual_seed(seed)
        weight = torch.randn(in_channels, num_classes, generator=gen)
        bias = torch.randn(num_classes, generator=gen)
        self.register_buffer("weight", weight)
        self.register_buffer("bias", bias)

    def _standardize(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True).clamp_min(self.eps)
        return (x - mean) / std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.standardize:
            x = self._standardize(x)
        logits = x @ self.weight + self.bias
        if self.noise_std > 0:
            logits = logits + self.noise_std * torch.randn_like(logits)
        return logits
