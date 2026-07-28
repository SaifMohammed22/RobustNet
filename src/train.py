import os
import argparse
import hydra
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from utils import get_logger
from models import build_model
from engine import train_one_epoch
from utils import build_optimizer
from data_utils import download_dataset, prep_data_loader, build_transform

logger = get_logger(__name__)
writer = SummaryWriter("logs")

@hydra.main(version_base=None, config_path="../config", config_name="config.yaml")
def train(cfg):
    EPOCHS = cfg.SOLVER.MAX_EPOCHS
    model = build_model(cfg, cfg.MODEL.NAME, is_train=True)
    train_transform = build_transform(cfg, is_train=True)
    train_set = download_dataset(cfg, train_transform, is_train=True)
    train_loader = prep_data_loader(cfg, train_set, is_train=True)
    loss = F.cross_entropy
    optimizer = build_optimizer(cfg, model)

    for epoch in range(EPOCHS):
        logger.info(f"Epoch {epoch + 1}:")
        avg_loss = train_one_epoch(cfg, model, optimizer, train_loader, loss)
        logger.info(f"train loss: {avg_loss}")



if __name__ == "__main__":
    train()
