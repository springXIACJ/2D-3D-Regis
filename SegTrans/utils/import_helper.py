#  Copyright (c) 2022-2023. by Yi GU <gu.yi.gu4@naist.ac.jp>, Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed without the express permission of Yi GU.

from importlib import import_module
from typing import Callable, AnyStr, Any


class ImportHelper:

    @staticmethod
    def get_class(path: AnyStr) -> Callable:
        splited = path.split('.')
        module_path = '.'.join(splited[:-1])
        return import_module(module_path).__getattribute__(splited[-1])

    @classmethod
    def init_from_config(cls, config: dict[AnyStr, Any], **kwargs) -> Any:
        assert "class" in config, config
        class_path: str = config.pop("class")
        class_path = class_path.replace('\\', '.')
        class_ = cls.get_class(class_path)
        obj = class_(**config, **kwargs)
        config["class"] = class_path
        return obj
