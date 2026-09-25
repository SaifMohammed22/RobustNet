import torch


def build_scheduler(cfg, optimizer):
    """Builds a learning rate scheduler with warmup and multi-step decay.

    Supports both epoch-based (WARMUP_EPOCHS) and iteration-based (WARMUP_ITERS)
    warmup configurations, and safely parses milestone steps from various formats
    (lists, tuples, or serialized YAML strings).
    """
    warmup_iters = getattr(cfg.SOLVER, "WARMUP_EPOCHS", None)
    if warmup_iters is None:
        warmup_iters = getattr(cfg.SOLVER, "WARMUP_ITERS", 0)

    raw_steps = cfg.SOLVER.STEPS
    if isinstance(raw_steps, str):
        cleaned = raw_steps.strip("()[] ")
        milestones = [int(s.strip()) for s in cleaned.split(",") if s.strip()]
    elif isinstance(raw_steps, (int, float)):
        milestones = [int(raw_steps)]
    else:
        milestones = [int(s) for s in raw_steps]

    if warmup_iters <= 0:
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=cfg.SOLVER.GAMMA,
        )

    decay_milestones = [max(1, m - warmup_iters) for m in milestones]

    scheduler1 = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=cfg.SOLVER.WARMUP_FACTOR,
        total_iters=warmup_iters,
    )
    scheduler2 = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=decay_milestones,
        gamma=cfg.SOLVER.GAMMA,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[scheduler1, scheduler2],
        milestones=[warmup_iters],
    )
    return scheduler
