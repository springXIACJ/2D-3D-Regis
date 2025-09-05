#  Copyright (c) 2023. by Yi GU <gu.yi.gu4@naist.ac.jp>, Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed without the express permission of Yi GU.

from .test_dataset import TestDataset
from . import DatasetProtocol
from torch.utils.data import Dataset
import random


class VisualizationDataset(Dataset, DatasetProtocol):

    def __init__(self, test_dataset: TestDataset):
        self._test_dataset = test_dataset

    def __len__(self):
        return 1

    def __getitem__(self, item):
        idx = random.randint(0, len(self._test_dataset) - 1)
        return self._test_dataset[idx]
