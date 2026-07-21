import os
from torchvision.datasets import CIFAR100

from data_preprocessing import build_transform
from config import cfg


# Download CIFAR100 dataset into data/ directory
def download_dataset(transform, is_train=True):
    if is_train:
        dataset = CIFAR100(root=cfg.DATASET.DIR, train=is_train,
                           transform=transform, download=True)
    else:
        dataset = CIFAR100(root=cfg.DATASET.DIR, train=is_train,
                           transform=transform, download=True)
    return dataset


if __name__ == "__main__":
    train_transform = build_transform(cfg, is_train=True)
    test_transform = build_transform(cfg, is_train=False)
    train_set = download_dataset(train_transform, is_train=True)
    test_set = download_dataset(test_transform, is_train=False)
    print("Train and Test sets downloaded successfully.")
