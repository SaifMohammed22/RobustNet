import argparse
import hydra
import torch.nn.functional as F
from models import build_model
from engine import train_one_epoch
from utils import build_optimizer
from data_utils import download_dataset, prep_data_loader, build_transform


@hydra.main(version_base=None, config_path="config", config_name="config")
def train(cfg):
    EPOCHS = 2  # cfg.SOLVER.MAX_EPOCHS
    model = build_model(cfg, is_train=True)
    train_transform = build_transform(cfg, is_train=True)
    train_set = download_dataset(cfg, train_transform, is_train=True)
    train_loader = prep_data_loader(cfg, train_set, is_train=True)
    loss = F.cross_entropy
    optimizer = build_optimizer(cfg, model)

    for _ in range(EPOCHS):
        avg_loss = train_one_epoch(cfg, model, optimizer, train_loader, loss)


if __name__ == "__main__":
    train()
