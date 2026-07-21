from torchvision.transforms import v2


def build_transform(cfg, is_train=True):
    if is_train:
        transform = v2.Compose(
            v2.RandomResizedCrop(
                size=cfg.INPUT.SIZE_TRAIN,
                scale=(cfg.INPUT.MIN_SCALE_TRAIN, cfg.INPUT.MAX_SCALE_TRAIN)),
            v2.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            v2.ToTensor(),
            v2.Normalize(mean=cfg.PIXEL_MEAN, std=cfg.PIXEL_STD),
            v2.RandomErasing(),
        )
    else:
        transform = v2.Compose(
            v2.Resize(cfg.INPUT.SIZE_TEST),
            v2.ToTensor(),
            v2.Normalize(mean=cfg.PIXEL_MEAN, std=cfg.PIXEL_STD),
        )
    return transform
