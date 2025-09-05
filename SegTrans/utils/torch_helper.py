#  Copyright (c) 2022-2023. by Yi GU <gu.yi.gu4@naist.ac.jp>, Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed without the express permission of Yi GU.


import torch
from typing import AnyStr, Any
from torch.optim import lr_scheduler, Optimizer
from torch.optim.lr_scheduler import LRScheduler
import torch.nn as nn
from timm.models.layers import trunc_normal_
from collections import OrderedDict
from .typing import TypePathLike
import logging

_logger = logging.getLogger(__name__)


class TorchHelper:

    @classmethod
    def load_network_by_path(cls, net: torch.nn.Module, path: TypePathLike, strict=True) -> tuple[list[str], list[str]]:
        missing, unexpected = cls.load_network_by_dict(net, torch.load(path, map_location="cpu"), strict=strict)
        if len(missing) > 0:
            _logger.warning(
                f"Missed {len(missing)}/{len(net.state_dict())} keys "
                f"when loading {net.__class__.__name__} from {path}.")
            _logger.warning(f"Missing keys: {missing}")
        if len(unexpected) > 0:
            _logger.warning(
                f"Encountered {len(unexpected)} unexpected keys "
                f"when loading {net.__class__.__name__} from {path}.")
            _logger.warning(f"Unexpected keys: {unexpected}")
        _logger.info(
            f"Model {net.__class__.__name__} ({len(net.state_dict()) - len(missing)}/{len(net.state_dict())}) "
            f"loaded from {path}.")
        return missing, unexpected

    @staticmethod
    def load_network_by_dict(
            net: torch.nn.Module, params_dict: dict[AnyStr, Any], strict=True) -> tuple[list, list]:
        if strict:
            return net.load_state_dict(params_dict, strict=strict)
        else:
            try:
                missing, unexpected = net.load_state_dict(params_dict, strict=strict)
            except RuntimeError:
                loaded = []
                model_dict = net.state_dict()
                for key, value in params_dict.items():
                    if key in model_dict:
                        if model_dict[key].shape == value.shape:
                            model_dict[key] = value
                            loaded.append(key)
                loaded_keys = set(loaded)
                missing = list(set(model_dict.keys()) - loaded_keys)
                unexpected = list(set(params_dict.keys()) - loaded_keys)
                net.load_state_dict(OrderedDict(model_dict))
        return missing, unexpected

    @staticmethod
    def set_requires_grad(nets: torch.nn.Module | list[torch.nn.Module], requires_grad=False) -> None:
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            for param in net.parameters():
                param.requires_grad = requires_grad

    @staticmethod
    def get_scheduler(optimizer: Optimizer, config: dict[AnyStr, Any], epochs: int) -> LRScheduler:

        policy = config["policy"]
        if policy == "constant":
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1.0)
        elif policy == "linear":
            assert not config["infinite"]
            decay_epoch = config["decay_epoch"]
            def lambda_rule(epoch):
                lr_l = 1.0 - max(0, epoch - decay_epoch) / float(epochs - decay_epoch + 1)
                return lr_l

            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        elif policy == "cosine_warm":
            min_lr = config.get("min_lr")
            T_0 = config.get("T_0")
            if T_0 is None:
                T_0 = 10
            T_mult = config.get("T_mul")
            if T_mult is None:
                T_mult = 1
            scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                 T_0=T_0,
                                                                 T_mult=T_mult,
                                                                 eta_min=min_lr)
        else:
            raise NotImplementedError('learning rate policy [%s] is not implemented', policy)
        return scheduler

    @staticmethod
    def constant_init(module, val, bias=0):
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.constant_(module.weight, val)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    @staticmethod
    def trunc_normal_init(module: nn.Module,
                          mean: float = 0,
                          std: float = 1,
                          a: float = -2,
                          b: float = 2,
                          bias: float = 0) -> None:
        if hasattr(module, 'weight') and module.weight is not None:
            trunc_normal_(module.weight, mean, std, a, b)  # type: ignore
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)  # type: ignore




