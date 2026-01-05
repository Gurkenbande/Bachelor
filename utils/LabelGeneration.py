import torch
import torch.nn.functional as F
from utils.TeacherNet import TeacherNet


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
    ):
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hidden = hidden
        self.temperature = temperature
        self.stochastic = stochastic
        self.seed = seed
        self.device = device

        self._build_teacher()

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

    # -------------------------------------------------
    # Label generation methods (add more freely)
    # -------------------------------------------------

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
