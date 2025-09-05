#  Copyright (c) 2022-2023. by Yi GU <gu.yi.gu4@naist.ac.jp>, Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed without the express permission of Yi GU.

import re
from typing import AnyStr
from .typing import TypePathLike


class OSHelper:

    @staticmethod
    def format_path_string(path: TypePathLike, src_sep='\\', ret_ep='/') -> AnyStr:
        ret = path.replace(src_sep, ret_ep)
        for server in ["conger", "scallop", "salmon"]:
            ret = re.sub(f"^//{server}/user", f"/win/{server}/user", ret, flags=re.IGNORECASE)
        return ret
