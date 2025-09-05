#  Copyright (c) 2021-2023. by Yi GU <gu.yi.gu4@naist.ac.jp>, Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed without the express permission of Yi GU.

import numpy as np
import torch
import skimage.transform as skt
import numpy.typing as npt
from typing import TypeVar
from utils.typing import TypeNPDTypeFloat

T = TypeVar("T")


class ImageHelper:

    @staticmethod
    def resize(image: npt.NDArray[T], output_shape: list[int] | tuple[int, int],
               order: int, mode='reflect', cval=0, clip=True, preserve_range=True,
               anti_aliasing=True,
               anti_aliasing_sigma: None | float | tuple[float, float] = None) -> npt.NDArray[T]:
        if order == 0:
            anti_aliasing = False
        return skt.resize(image, output_shape=output_shape,
                          order=order, mode=mode, cval=cval, clip=clip, preserve_range=preserve_range,
                          anti_aliasing=anti_aliasing, anti_aliasing_sigma=anti_aliasing_sigma)


    @staticmethod
    def min_max_scale(x: npt.NDArray[TypeNPDTypeFloat] | torch.Tensor,
                      min_val: float | npt.NDArray[TypeNPDTypeFloat] | torch.Tensor | None = None,
                      max_val: float | npt.NDArray[TypeNPDTypeFloat] | torch.Tensor | None = None) \
            -> tuple[
                npt.NDArray[TypeNPDTypeFloat] | torch.Tensor,
                float | npt.NDArray[TypeNPDTypeFloat] | torch.Tensor,
                float | npt.NDArray[TypeNPDTypeFloat] | torch.Tensor]:
        """

        :param x:
        :param min_val:
        :param max_val:
        :return:
            [0., 1.]
        """
        if min_val is None and max_val is None:
            min_val = x.min()
            max_val = x.max()
        if max_val == min_val:
            if isinstance(x, np.ndarray):
                ret = np.zeros_like(x)
            elif isinstance(x, torch.Tensor):
                ret = torch.zeros_like(x, device=x.device)
            else:
                raise NotImplementedError(f"Unknown type {type(x)}")
        else:
            ret = (x - min_val) / (max_val - min_val)
        return ret, min_val, max_val

    @staticmethod
    def standardize(image: npt.NDArray[TypeNPDTypeFloat] | torch.Tensor,
                    mean: float | npt.NDArray[TypeNPDTypeFloat] | torch.Tensor | None = None,
                    std: float | npt.NDArray[TypeNPDTypeFloat] | torch.Tensor | None = None) \
            -> tuple[
                npt.NDArray[TypeNPDTypeFloat] | torch.Tensor,
                float | npt.NDArray[TypeNPDTypeFloat] | torch.Tensor,
                float | npt.NDArray[TypeNPDTypeFloat] | torch.Tensor]:
        if mean is None:
            mean = image.mean()
        if std is None:
            std = image.std(ddof=1)
        ret = image - mean
        if std == 0.:
            return ret, mean, std
        return ret / std, mean, std

    @staticmethod
    def denormal(image: npt.NDArray[TypeNPDTypeFloat] | torch.Tensor,
                 ret_min_val: float | npt.NDArray[TypeNPDTypeFloat] | torch.Tensor = 0.,
                 ret_max_val: float | npt.NDArray[TypeNPDTypeFloat] | torch.Tensor = 255.) \
            -> npt.NDArray[TypeNPDTypeFloat] | torch.Tensor:
        """
        [-1, 1.] -> [0, 255.]
        :param ret_max_val:
        :param ret_min_val:
        :param image: Normalized image with range [-1, 1]
        :return: Denormalized image with range [min, max]
        """
        return (image + 1.) * (ret_max_val - ret_min_val) / 2. + ret_min_val
