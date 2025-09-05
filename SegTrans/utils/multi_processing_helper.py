#  Copyright (c) 2023. by Yi GU <gu.yi.gu4@naist.ac.jp>, Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed without the express permission of Yi GU.


import multiprocessing.pool as mpp
from typing import Any, Callable, Generator, Optional
from collections.abc import Iterable, Sequence
from tqdm import tqdm


class MultiProcessingHelper:

    def __init__(self, pool_class: str = "torch"):

        if pool_class in ["torch"]:
            # try:
            from torch.multiprocessing import Pool
        else:
            from multiprocessing import Pool
        self.__pool = Pool

    @staticmethod
    def _function_proxy(fun: Callable, kwargs: dict[str, Any]) -> Any:
        return fun(**kwargs)

    def run(self,
            args: Sequence[Any],
            n_worker: int,
            func: Optional[Callable] = None,
            desc: None | str = None,
            mininterval: None | float = None,
            maxinterval: None | float = None):
        assert len(args) > 0
        tqdm_args = {"total": len(args), "desc": desc}
        if mininterval is not None:
            tqdm_args["mininterval"] = mininterval
        if maxinterval is not None:
            tqdm_args["maxinterval"] = maxinterval
        if n_worker > 0:
            with self.__pool(n_worker) as pool:
                provider = pool.istarmap(MultiProcessingHelper._function_proxy
                                         if func is None else func,
                                         iterable=args)
                provider = tqdm(provider, **tqdm_args)
                return [ret for ret in provider]
        else:
            """
                Version 1
            """
            # re = []
            # if func is None:
            #     for fun, kwargs in tqdm(args, **tqdm_args):
            #         re.append(self._function_proxy(fun=fun, kwargs=kwargs))
            # else:
            #     for arg in tqdm(args, **tqdm_args):
            #         re.append(func(*arg))
            # return re

            exec_fun = MultiProcessingHelper._function_proxy if func is None else func
            """
                Version 2
            """
            # re = []
            # for item in tqdm(args, **tqdm_args):
            #     re.append(exec_fun(*item))
            # return re

            """
                Version 3
            """
            return [exec_fun(*item) for item in tqdm(args, **tqdm_args)]
        pass


def istarmap(self: object, func: Callable, iterable: Iterable, chunksize=1) -> Generator[Any, Any, None]:
    """starmap-version of imap
    """
    self._check_running()
    if chunksize < 1:
        raise ValueError(
            "Chunksize must be 1+, not {0:n}".format(
                chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    result = mpp.IMapIterator(self)
    self._taskqueue.put((self._guarded_task_generation(result._job, mpp.starmapstar, task_batches),
                         result._set_length))
    return (item for chunk in result for item in chunk)

mpp.Pool.istarmap = istarmap
