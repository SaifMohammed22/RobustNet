import torch


def build_scheduler(cfg, optimizer):
    # Warmup + MultiStepLR

    # Warmup
    scheduler1 = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=cfg.SOLVER.WARMUP_FACTOR,
        total_iters=cfg.SOLVER.WARMUP_ITERS
    )
    # LR Decay after warmup
    scheduler2 = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=list(cfg.SOLVER.STEPS),
        gamma=cfg.SOLVER.GAMMA,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [scheduler1, scheduler2], milestones=[
            cfg.SOLVER.WARMUP_ITERS]
    )
    return scheduler
