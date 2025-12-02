import time
from typing import Optional

import torch
from colossalai.cluster import DistCoordinator


class Timer:
    def __init__(self, name, log=False, coordinator: Optional[DistCoordinator] = None):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.log = log
        self.coordinator = coordinator

    @property
    def elapsed_time(self):
        return self.end_time - self.start_time

    def __enter__(self):
        torch.cuda.synchronize()
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.coordinator is not None:
            self.coordinator.block_all()
        torch.cuda.synchronize()
        self.end_time = time.time()
        if self.log:
            print(f"Elapsed time for {self.name}: {self.elapsed_time:.2f} s")
