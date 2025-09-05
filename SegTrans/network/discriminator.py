# Original from https://github.com/NVIDIA/pix2pixHD

#  Modification
#  Copyright (c) 2023. by Yi GU <gu.yi.gu4@is.naist.jp>, Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed without the express permission of Yi GU.

import torch.nn as nn
import numpy as np
from utils import TorchHelper


class NLayerDiscriminator(nn.Module):

    def __init__(self,
                 input_nc: int,
                 ndf: int = 64,
                 n_layers: int = 3):
        super(NLayerDiscriminator, self).__init__()
        self.__n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                     nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                # nn.GroupNorm(1, nf, eps=1e-6),
                nn.InstanceNorm2d(nf, eps=1e-6),
                nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            # nn.GroupNorm(1, nf, eps=1e-6),
            nn.InstanceNorm2d(nf, eps=1e-6),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        for n in range(len(sequence)):
            setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        self.init_weights()

    def forward(self, input):
        res = [input]
        for n in range(self.__n_layers+2):
            model = getattr(self, 'model'+str(n))
            res.append(model(res[-1]))
        return res[1:]

    def init_weights(self):
        for m in self.modules():
            is_normalayer = False
            is_normalayer |= isinstance(m, nn.GroupNorm)
            is_normalayer |= isinstance(m, nn.InstanceNorm2d)
            is_normalayer |= isinstance(m, nn.BatchNorm2d)
            if is_normalayer:
                TorchHelper.constant_init(m, 1.)
            elif isinstance(m, nn.Conv2d):
                if m.kernel_size == (1, 1):
                    TorchHelper.trunc_normal_init(m, std=.02, bias=0.)
            elif isinstance(m, nn.Linear):
                TorchHelper.trunc_normal_init(m, std=.02, bias=0.)


class MultiscaleDiscriminator(nn.Module):

    def __init__(self,
                 input_nc: int,
                 ndf: int = 64,
                 n_layer: int = 3,
                 num_D: int = 2):
        super(MultiscaleDiscriminator, self).__init__()
        self.__num_D = num_D
        self.__n_layers = n_layer

        for i in range(1, num_D + 1):
            netD = NLayerDiscriminator(input_nc=input_nc,
                                       ndf=ndf,
                                       n_layers=n_layer)
            setattr(self, "model{}".format(i), netD)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
        self.init_weights()

    def forward(self, x):
        result = []
        downsampled = x
        for i in range(1, self.__num_D):
            model = getattr(self, "model{}".format(i))
            model_out = model(downsampled)
            result.append(model_out)
            downsampled = self.downsample(downsampled)

        model = getattr(self, "model{}".format(self.__num_D))
        result.append(model(downsampled))
        return result

    def init_weights(self):
        for m in self.modules():
            is_normalayer = False
            is_normalayer |= isinstance(m, nn.GroupNorm)
            is_normalayer |= isinstance(m, nn.InstanceNorm2d)
            is_normalayer |= isinstance(m, nn.BatchNorm2d)
            if is_normalayer:
                TorchHelper.constant_init(m, 1.)
            elif isinstance(m, nn.Conv2d):
                if m.kernel_size == (1, 1):
                    TorchHelper.trunc_normal_init(m, std=.02, bias=0.)
            elif isinstance(m, nn.Linear):
                TorchHelper.trunc_normal_init(m, std=.02, bias=0.)