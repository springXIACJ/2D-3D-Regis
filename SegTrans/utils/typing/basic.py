#  Copyright (c) 2023. by Yi GU <gu.yi.gu4@naist.ac.jp>, Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed without the express permission of Yi GU.

import numpy as np
import os
from typing import AnyStr

TypeConfig = dict[str, any]
TypeNPDTypeUnsigned = np.ubyte | np.ushort | np.uintc | np.uint | np.ulonglong
TypeNPDTypeFloat = np.float32 | np.float64

TypePathLike = AnyStr | os.PathLike[AnyStr]
