"""Test/Evaluate the model"""
import torch
from utils import get_logger

logger = get_logger(__name__)


def test_one_epoch(cfg, model, test_loader, loss_fn):
    running_loss = 0.0
    last_loss = 0.0

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            images, labels = data
            images, labels = images.to(
                cfg.MODEL.DEVICE), labels.to(cfg.MODEL.DEVICE)

            outputs = model(images)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item()
            if (i + 1) % cfg.SOLVER.LOG_PERIOD == 0:
                last_loss = running_loss / cfg.SOLVER.LOG_PERIOD
                # print(f"    batch {i + 1} test loss: {last_loss}")
                running_loss = 0.0
    return last_loss
