#  Copyright (c) 2023. by Yi GU <gu.yi.gu4@naist.ac.jp>, Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed without the express permission of Yi GU.


from .base_dataset import BaseDataset
from utils.typing import TypePathLike
from utils import ConfigureHelper


class TestDataset(BaseDataset):

    def __init__(
            self,
            data_root: TypePathLike, preload: bool,
            image_size: int | tuple[int, int], n_worker=ConfigureHelper.max_n_worker, debug=False):
        super().__init__(
            data_root=data_root, split="test", ret_size=image_size,
            n_preload_worker=n_worker, preload_dataset=preload, debug=debug)