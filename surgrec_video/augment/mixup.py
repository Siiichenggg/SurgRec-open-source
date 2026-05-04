import torch
import torch.nn.functional as F
import random


class Mixup:
    def __init__(
        self,
        mixup_alpha: float = 0.8,
        cutmix_alpha: float = 0.0,
        cutmix_minmax=None,
        prob: float = 1.0,
        switch_prob: float = 0.5,
        mode: str = 'batch',
        label_smoothing: float = 0.0,
        num_classes: int = 1000,
    ) -> None:
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.switch_prob = switch_prob
        self.mode = mode
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing

    def _sample_lambda(self, alpha: float) -> float:
        if alpha > 0.0:
            lam = torch.distributions.Beta(alpha, alpha).sample().item()
        else:
            lam = 1.0
        return lam

    def _one_hot(self, targets: torch.Tensor) -> torch.Tensor:
        return F.one_hot(targets.to(torch.int64), num_classes=self.num_classes).float()

    def __call__(self, x: torch.Tensor, target: torch.Tensor):
        # Only perform Mixup (ignore CutMix for minimal implementation)
        if random.random() > self.prob:
            return x, self._one_hot(target)

        lam = self._sample_lambda(self.mixup_alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)

        mixed_x = lam * x + (1.0 - lam) * x[index, :]
        y1 = self._one_hot(target)
        y2 = self._one_hot(target[index])
        mixed_y = lam * y1 + (1.0 - lam) * y2
        return mixed_x, mixed_y
