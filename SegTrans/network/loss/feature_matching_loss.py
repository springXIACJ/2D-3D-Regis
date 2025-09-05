#  Copyright (c) 2023. by Yi GU <gu.yi.gu4@is.naist.jp>, Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed without the express permission of Yi GU.


import torch.nn as nn
import torch


class FeatureMatchingLoss(nn.Module):

    def __init__(self, n_disc=2, n_layer=3):
        super().__init__()
        self.n_disc = n_disc
        self.n_layer = n_layer

        self.disc_weight = 1 / self.n_disc
        self.layer_weight = 4. / (self.n_layer + 1)
        self.criterion = nn.L1Loss()

    def forward(
            self,
            pred_fake: list[list[torch.Tensor]],
            pred_real: list[list[torch.Tensor]]):
        assert isinstance(pred_fake, list) and isinstance(pred_fake[0], list)
        loss_all = 0
        for i_pred_fake, i_pred_real in zip(pred_fake, pred_real):
            for j in range(self.n_layer + 1):
                loss = self.criterion(i_pred_fake[j], i_pred_real[j].detach())
                loss_all = loss_all + self.disc_weight * self.layer_weight * loss
        return loss_all


