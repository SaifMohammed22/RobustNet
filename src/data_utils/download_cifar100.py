from torchvision.datasets import CIFAR100


# Download CIFAR100 dataset into data/ directory
def download_dataset(cfg, transform, is_train=True):
    return CIFAR100(root=cfg.DATASET.DIR, train=is_train,
                    transform=transform, download=True)
