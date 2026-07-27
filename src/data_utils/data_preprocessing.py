import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2


def build_transform(cfg, is_train=True):
    if is_train:
        transform = v2.Compose([
            v2.RandomResizedCrop(
                size=cfg.INPUT.SIZE_TRAIN,
                scale=(cfg.INPUT.MIN_SCALE_TRAIN, cfg.INPUT.MAX_SCALE_TRAIN)),
            v2.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            v2.RandomErasing(),
        ])
    else:
        transform = v2.Compose([
            v2.Resize(cfg.INPUT.SIZE_TEST),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        ])
    return transform


def prep_data_loader(cfg, dataset, is_train=True):
    batch_size = cfg.SOLVER.IMS_PER_BATCH if is_train else cfg.TEST.IMS_PER_BATCH
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
