from functools import partial

import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched

from .fastai_optim import OptimWrapper
from .learning_schedules_fastai import CosineWarmupLR, OneCycle


def build_optimizer(model, optim_cfg, for_predictor=False):
    # INFO: if optim_cfg has the predictor
    if optim_cfg.get('PREDICTOR',None) is not None:
        param_group_without_predictor = []
        param_group_with_predictor = []
        for n,m in model.named_parameters():
            if 'predictor' in n:
                param_group_with_predictor.append(m)
            else:
                param_group_without_predictor.append(m)
        if for_predictor:
            param_group_ = param_group_with_predictor
        else:
            param_group_ = param_group_without_predictor
    else:
        param_group_ = model.parameters()

    if optim_cfg.OPTIMIZER == 'adam':
        optimizer = optim.Adam(param_group_, lr=optim_cfg.LR, weight_decay=optim_cfg.WEIGHT_DECAY)
    elif optim_cfg.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            param_group_, lr=optim_cfg.LR, weight_decay=optim_cfg.WEIGHT_DECAY,
            momentum=optim_cfg.MOMENTUM
        )
    elif optim_cfg.OPTIMIZER == 'adam_onecycle':
        def children(m: nn.Module):
            return list(m.children())
        def num_children(m: nn.Module) -> int:
            return len(children(m))
        def flatten_model(m):
            # TODO: get group for predictor & non_predictor part respectively
            if num_children(m):
                return sum(map(flatten_model, m.children()), [])
            else:
                return [m]
        def get_layer_groups(model, for_predictor=False):
            non_predictor_groups = []
            predictor_groups = []
            if hasattr(model,'predictor'):
                for n,c in model.named_children():
                    if 'predictor' in n:
                        predictor_group = flatten_model(c)
                        predictor_groups += predictor_group
                    else:
                        non_predictor_group = flatten_model(c)
                        non_predictor_groups += non_predictor_group
                if for_predictor:
                    flattened_model = predictor_groups
                else:
                    flattened_model = non_predictor_groups

            else:
                non_predictor_groups = model
                flattened_model = flatten_model(model)
            print('Initilize Opt with predictor: {}, Number of Params:{}'.format(for_predictor, len(flattened_model)))
            return [nn.Sequential(*flattened_model)]

        # flatten_model = lambda m: sum(map(flatten_model, m.children()), []) if num_children(m) else [m]
        # get_layer_groups = lambda m: [nn.Sequential(*flatten_model(m))]

        optimizer_func = partial(optim.Adam, betas=(0.9, 0.99))
        optimizer = OptimWrapper.create(
            optimizer_func, optim_cfg.LR, get_layer_groups(model, for_predictor=for_predictor), wd=optim_cfg.WEIGHT_DECAY, true_wd=True, bn_wd=True,
        )
    else:
        raise NotImplementedError

    return optimizer


def build_scheduler(optimizer, total_iters_each_epoch, total_epochs, last_epoch, optim_cfg):
    decay_steps = [x * total_iters_each_epoch for x in optim_cfg.DECAY_STEP_LIST]
    def lr_lbmd(cur_epoch):
        cur_decay = 1
        for decay_step in decay_steps:
            if cur_epoch >= decay_step:
                cur_decay = cur_decay * optim_cfg.LR_DECAY
        return max(cur_decay, optim_cfg.LR_CLIP / optim_cfg.LR)

    lr_warmup_scheduler = None
    total_steps = total_iters_each_epoch * total_epochs
    if optim_cfg.OPTIMIZER == 'adam_onecycle':
        lr_scheduler = OneCycle(
            optimizer, total_steps, optim_cfg.LR, list(optim_cfg.MOMS), optim_cfg.DIV_FACTOR, optim_cfg.PCT_START
        )
    else:
        lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lbmd, last_epoch=last_epoch)

        if optim_cfg.LR_WARMUP:
            lr_warmup_scheduler = CosineWarmupLR(
                optimizer, T_max=optim_cfg.WARMUP_EPOCH * len(total_iters_each_epoch),
                eta_min=optim_cfg.LR / optim_cfg.DIV_FACTOR
            )

    return lr_scheduler, lr_warmup_scheduler
