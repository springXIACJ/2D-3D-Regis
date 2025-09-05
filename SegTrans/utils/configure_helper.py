#  Copyright (c) 2022-2023. by Yi GU <gu.yi.gu4@naist.ac.jp>, Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed without the express permission of Yi GU.

import torch
import numpy as np
import random
import os

_max_num_worker_suggest = 0
if hasattr(os, 'sched_getaffinity'):
    try:
        _max_num_worker_suggest = len(os.sched_getaffinity(0))
    except Exception:
        pass
if _max_num_worker_suggest == 0:
    # os.cpu_count() could return Optional[int]
    # get cpu count first and check None in order to satify mypy check
    cpu_count = os.cpu_count()
    if cpu_count is not None:
        _max_num_worker_suggest = cpu_count

if "SLURM_CPUS_PER_TASK" in os.environ:
    _max_num_worker_suggest = int(os.environ["SLURM_CPUS_PER_TASK"])

class ConfigureHelper:
    max_n_worker = _max_num_worker_suggest
