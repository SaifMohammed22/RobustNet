import hydra
from .data_preprocessing import build_transform, prep_data_loader
from torchvision.datasets import CIFAR100


# Download CIFAR100 dataset into data/ directory
def download_dataset(cfg, transform, is_train=True):
    return CIFAR100(root=cfg.DATASET.DIR, train=is_train,
                    transform=transform, download=True)



@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg):
    train_transform = build_transform(cfg, is_train=True)
    test_transform = build_transform(cfg, is_train=False)
    train_set = download_dataset(cfg, train_transform, is_train=True)
    test_set = download_dataset(cfg, test_transform, is_train=False)
    train_loader = prep_data_loader(cfg, train_set, is_train=True)
    test_loader = prep_data_loader(cfg, test_set, is_train=False)
    train_img, train_label = next(iter(train_loader))
    test_img, test_label = next(iter(test_loader))
    print(train_img.shape)
    print(test_img.shape)


if __name__ == "__main__":
    main()
