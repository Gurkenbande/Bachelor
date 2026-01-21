import torch
import torch.nn.functional as F
from Trashbin.TeacherNet import TeacherNet
from Trashbin.LabelSampling import LabelSampling


class LabelGeneration:
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden: int = 64,
        temperature: float = 1.0,
        stochastic: bool = False,
        seed: int = 0,
        device=None,
        *,
        method: str = "teacher",
        sampler_noise_std: float = 0.0,
        sampler_standardize: bool = True,
    ):
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hidden = hidden
        self.temperature = temperature
        self.stochastic = stochastic
        self.seed = seed
        self.device = device
        self.method = self._normalize_method(method)
        self.sampler_noise_std = sampler_noise_std
        self.sampler_standardize = sampler_standardize

        self._build_teacher()
        self._build_sampler()

    def _normalize_method(self, method: str) -> str:
        if method in {"labelsampling", "sampling"}:
            return "labelsampling"
        if method == "teacher":
            return "teacher"
        raise ValueError("method must be 'teacher' or 'labelsampling'")

    def _build_teacher(self):
        torch.manual_seed(self.seed)

        self.teacher = TeacherNet(
            self.in_channels,
            hidden=self.hidden,
            num_classes=self.num_classes,
        ).to(self.device)

        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    def _build_sampler(self):
        self.sampler = LabelSampling(
            self.in_channels,
            self.num_classes,
            seed=self.seed,
            noise_std=self.sampler_noise_std,
            standardize=self.sampler_standardize,
        )
        if self.device is not None:
            self.sampler = self.sampler.to(self.device)
        self.sampler.eval()
        for p in self.sampler.parameters():
            p.requires_grad_(False)


    @torch.no_grad()
    def teacher_labels(self, data):
        logits = self.teacher(
            data.x.to(self.device),
            data.edge_index.to(self.device),
        ) / self.temperature

        probs = torch.softmax(logits, dim=-1)

        if self.stochastic:
            return torch.multinomial(probs, 1).squeeze(-1)

        return probs.argmax(dim=-1)

    @torch.no_grad()
    def teacher_logits(self, data):
        return self.teacher(
            data.x.to(self.device),
            data.edge_index.to(self.device),
        )

    @torch.no_grad()
    def teacher_probs(self, data):
        logits = self.teacher_logits(data) / self.temperature
        return torch.softmax(logits, dim=-1)

    @torch.no_grad()
    def sampler_logits(self, data):
        x = data.x.to(self.device) if self.device is not None else data.x
        return self.sampler(x)

    @torch.no_grad()
    def sampler_probs(self, data):
        logits = self.sampler_logits(data) / self.temperature
        return torch.softmax(logits, dim=-1)

    @torch.no_grad()
    def sampler_labels(self, data):
        probs = self.sampler_probs(data)
        if self.stochastic:
            return torch.multinomial(probs, 1).squeeze(-1)
        return probs.argmax(dim=-1)

    @torch.no_grad()
    def labels(self, data, method: str | None = None):
        method = self._normalize_method(method or self.method)
        if method == "teacher":
            return self.teacher_labels(data)
        return self.sampler_labels(data)

    @torch.no_grad()
    def logits(self, data, method: str | None = None):
        method = self._normalize_method(method or self.method)
        if method == "teacher":
            return self.teacher_logits(data)
        return self.sampler_logits(data)

    @torch.no_grad()
    def probs(self, data, method: str | None = None):
        method = self._normalize_method(method or self.method)
        if method == "teacher":
            return self.teacher_probs(data)
        return self.sampler_probs(data)
