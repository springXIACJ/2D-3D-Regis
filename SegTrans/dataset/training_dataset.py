#  Copyright (c) 2023. by Yi GU <gu.yi.gu4@naist.ac.jp>, Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed without the express permission of Yi GU.


from .base_dataset import BaseDataset
from utils.typing import TypePathLike, TypeNPDTypeFloat, TypeNPDTypeUnsigned
from utils import ConfigureHelper, ContainerHelper
import numpy.typing as npt
import numpy as np


class TrainingDataset(BaseDataset):

    def __init__(
            self, data_root: TypePathLike, preload: bool,
            image_size: int | tuple[int, int], n_worker=ConfigureHelper.max_n_worker, debug=False):
        super().__init__(
            data_root=data_root, split="trainval", ret_size=image_size,
            n_preload_worker=n_worker, preload_dataset=preload, debug=debug)

    def _augment(self, image: npt.NDArray[TypeNPDTypeFloat], seg: npt.NDArray[TypeNPDTypeUnsigned]) \
            -> tuple[npt.NDArray[TypeNPDTypeFloat], npt.NDArray[TypeNPDTypeUnsigned]]:
        # # vertical flip
        # if np.random.random() < 0.5:
        #     image = np.flip(image, axis=0)
        #     seg = np.flip(seg, axis=0)

        # horizontal flip
        if np.random.random() < 0.5:
            image = np.flip(image, axis=1)
            seg = np.flip(seg, axis=1)
        return image, seg
