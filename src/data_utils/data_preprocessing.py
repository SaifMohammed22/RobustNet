import torch
from torch.utils.data import DataLoader, random_split
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


def train_val_split(train_set, val_ratio=0.1, seed=42):
    N = train_set.data.shape[0]
    val_size = int(val_ratio * N)
    train_size = N - val_size
    train_subset, val_subset = random_split(
        train_set, [train_size, val_size], generator=torch.Generator().manual_seed(seed))
    return train_subset, val_subset
