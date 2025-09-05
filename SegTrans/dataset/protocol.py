#  Copyright (c) 2023. by Yi GU <gu.yi.gu4@naist.ac.jp>, Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed without the express permission of Yi GU.


from typing import Protocol, Sized
from abc import abstractmethod


class DatasetProtocol(Sized, Protocol):

    @abstractmethod
    def __getitem__(self, item):
        raise NotImplementedError

