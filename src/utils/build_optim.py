import torch


def build_optimizer(cfg, model):
    params = []
    for key, val in model.named_parameters():
        if not val.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"params": [val], "lr": lr, "weight_decay": weight_decay}]
    optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(
        params, momentum=cfg.SOLVER.MOMENTUM)
    return optimizer
