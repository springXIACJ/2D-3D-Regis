#  Copyright (c) 2023. by Yi GU <gu.yi.gu4@naist.ac.jp>, Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed without the express permission of Yi GU.

from collections import defaultdict
import torch


class EpochLossLogger:
    def __init__(self, device: torch.device | None = None):
        self._device = torch.device('cpu') if device is None else device
        self._losses: dict[str, torch.Tensor] = defaultdict(lambda: torch.tensor([0.], device=self._device))
        self._iter_counts = defaultdict(lambda: 0)

    def log(self, tag: str, value: torch.Tensor):
        self._losses[tag] += value.mean().detach()
        self._iter_counts[tag] += 1

    def summary(self) -> dict[str, torch.Tensor]:
        ret = {}
        for tag in self._losses:
            loss = self._losses[tag]
            epoch_loss: torch.Tensor = loss / self._iter_counts[tag]
            ret[tag] = epoch_loss
        return ret

