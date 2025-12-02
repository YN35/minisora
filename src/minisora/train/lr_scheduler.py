"""Learning-rate schedules for training."""

import math
from typing import List, Optional

from torch.optim.lr_scheduler import _LRScheduler


class LinearWarmupLR(_LRScheduler):
    """Linear warmup scheduler that holds the base LR afterwards."""

    def __init__(self, optimizer, warmup_steps: int = 0, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self) -> List[float]:
        if self.warmup_steps == 0:
            return self.base_lrs
        if self.last_epoch < self.warmup_steps:
            scale = (self.last_epoch + 1) / float(self.warmup_steps)
            return [base_lr * scale for base_lr in self.base_lrs]
        return self.base_lrs


class LinearWarmupCosineLR(_LRScheduler):
    """Linear warmup followed by cosine decay."""

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: Optional[float] = None,
        last_epoch: int = -1,
    ):
        if total_steps <= warmup_steps:
            raise ValueError("`total_steps` must be larger than `warmup_steps`.")
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super().__init__(optimizer, last_epoch=last_epoch)
        self.min_lr = min_lr if min_lr is not None else self.base_lrs[0] * 0.1

    def get_lr(self) -> List[float]:
        if self.last_epoch < self.warmup_steps:
            scale = (self.last_epoch + 1) / float(self.warmup_steps)
            return [base_lr * scale for base_lr in self.base_lrs]
        if self.last_epoch < self.total_steps:
            progress = (self.last_epoch - self.warmup_steps) / float(self.total_steps - self.warmup_steps)
            cosine = 0.5 * (1 + math.cos(math.pi * progress))
            return [self.min_lr + (base_lr - self.min_lr) * cosine for base_lr in self.base_lrs]
        return [self.min_lr for _ in self.base_lrs]


__all__ = ["LinearWarmupLR", "LinearWarmupCosineLR"]
