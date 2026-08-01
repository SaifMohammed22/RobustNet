"""The training epoch function"""
from utils import get_logger

logger = get_logger(__name__)


def train_one_epoch(cfg, model, optimizer, train_loader, loss_fn):
    running_loss = 0.0
    last_loss = 0.0

    for i, data in enumerate(train_loader):
        images, labels = data
        images, labels = images.to(
            cfg.MODEL.DEVICE), labels.to(cfg.MODEL.DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (i + 1) % cfg.SOLVER.LOG_PERIOD == 0:
            last_loss = running_loss / cfg.SOLVER.LOG_PERIOD
            # logger.info(f"  batch {i + 1} train loss: {last_loss}")
            running_loss = 0.0
    return last_loss
