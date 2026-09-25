import torch


def build_optimizer(cfg, model):
    non_bias_params = []
    bias_params = []

    for key, val in model.named_parameters():
        if not val.requires_grad:
            continue
        if "bias" in key:
            bias_params.append(val)
        else:
            non_bias_params.append(val)

    params = []
    if non_bias_params:
        params.append({
            "params": non_bias_params,
            "lr": cfg.SOLVER.BASE_LR,
            "weight_decay": cfg.SOLVER.WEIGHT_DECAY,
        })
    if bias_params:
        params.append({
            "params": bias_params,
            "lr": cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR,
            "weight_decay": cfg.SOLVER.WEIGHT_DECAY_BIAS,
        })

    opt_name = cfg.SOLVER.OPTIMIZER_NAME
    # Case-insensitive resolution for common torch.optim classes
    opt_map = {name.lower(): name for name in dir(torch.optim) if not name.startswith("_")}
    resolved_name = opt_map.get(opt_name.lower(), opt_name)

    if not hasattr(torch.optim, resolved_name):
        raise ValueError(f"Unknown optimizer: {opt_name}")

    opt_cls = getattr(torch.optim, resolved_name)
    kwargs = {}
    if resolved_name.lower() in ("sgd", "rmsprop"):
        kwargs["momentum"] = cfg.SOLVER.MOMENTUM

    optimizer = opt_cls(params, **kwargs)
    return optimizer
