"""Data samplers used during training."""

from typing import Optional

import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler


class ResumableDistributedSampler(DistributedSampler):
    """Distributed sampler that can resume from a previous step offset."""

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed, drop_last=drop_last)
        self.offset = 0
        self.batch_size = 1

    def set_batch_size(self, batch_size: int) -> None:
        if batch_size < 1:
            raise ValueError("`batch_size` must be >= 1.")
        self.batch_size = batch_size

    def set_epoch(self, epoch: int, step: int = 0) -> None:  # type: ignore[override]
        self.offset = step * self.batch_size
        super().set_epoch(epoch)

    def __iter__(self):  # type: ignore[override]
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=generator).tolist()
        else:
            indices = list(range(len(self.dataset)))

        if not self.drop_last:
            padding = self.total_size - len(indices)
            if padding > 0:
                repeat_factor = (padding + len(indices) - 1) // len(indices)
                indices += (indices * repeat_factor)[:padding]
        else:
            indices = indices[: self.total_size]

        indices = indices[self.rank : self.total_size : self.num_replicas]
        offset = min(self.offset, len(indices))
        indices = indices[offset:]
        return iter(indices)

    def __len__(self) -> int:  # type: ignore[override]
        base_len = super().__len__()
        return max(0, base_len - self.offset)


__all__ = ["ResumableDistributedSampler"]
