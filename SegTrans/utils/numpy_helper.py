#  Copyright (c) 2023. by Yi GU <gu.yi.gu4@naist.ac.jp>, Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed without the express permission of Yi GU.
import numpy.typing as npt
from typing import TypeVar
from utils.typing import TypeNPDTypeUnsigned

T = TypeVar("T")


class NumpyHelper:

    @staticmethod
    def unpad(array: npt.NDArray[T], pads: npt.NDArray[TypeNPDTypeUnsigned]) -> npt.NDArray[T]:
        slices: list[slice] = []
        for c in pads.tolist():
            e = None if c[1] == 0 else -c[1]
            slices.append(slice(c[0], e))
        return array[tuple(slices)]
