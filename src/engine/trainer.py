"""The training epoch function"""
from utils import get_logger

logger = get_logger(__name__)


def train_one_epoch(cfg, model, optimizer, train_loader, loss_fn):
    running_loss = 0.0

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

    return running_loss / (i + 1)
