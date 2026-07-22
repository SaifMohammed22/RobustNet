import os
import sys
sys.path.append(os.path.dirname("config"))

from data_preprocessing import build_transform
from config import cfg
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader


# Download CIFAR100 dataset into data/ directory
def download_dataset(transform, is_train=True):
    if is_train:
        dataset = CIFAR100(root=cfg.DATASET.DIR, train=is_train,
                           transform=transform, download=True)
    else:
        dataset = CIFAR100(root=cfg.DATASET.DIR, train=is_train,
                           transform=transform, download=True)
    return dataset

def prep_data_loader(cfg, dataset, is_train=True):
    if is_train:
        loader = DataLoader(dataset, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True)
    else:
        loader = DataLoader(dataset, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=True)
    return loader

if __name__ == "__main__":
    train_transform = build_transform(cfg, is_train=True)
    test_transform = build_transform(cfg, is_train=False)
    train_set = download_dataset(train_transform, is_train=True)
    test_set = download_dataset(test_transform, is_train=False)
    train_loader = prep_data_loader(cfg, train_set, is_train=True)
    test_loader = prep_data_loader(cfg, test_set, is_train=False)
    train_img, train_label = next(iter(train_loader))
    test_img, test_label = next(iter(test_loader))
    print(train_img.shape)
    print(test_img.shape)
