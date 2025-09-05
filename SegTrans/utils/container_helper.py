#  Copyright (c) 2023. by Yi GU <gu.yi.gu4@naist.ac.jp>, Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed without the express permission of Yi GU.


from collections.abc import Sequence
from abc import ABC
from typing import TypeVar

T = TypeVar('T')


class ContainerHelper(ABC):

    @staticmethod
    def to_tuple(x: T | Sequence[T]) -> tuple[T, T]:
        if not isinstance(x, Sequence):
            return x, x
        return tuple(x)