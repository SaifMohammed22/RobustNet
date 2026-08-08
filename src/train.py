import hydra
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from utils import get_logger
from models import build_model
from engine import train_one_epoch, test_one_epoch
from utils import build_optimizer
from data_utils import download_dataset, prep_data_loader, build_transform, train_val_split

logger = get_logger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="config.yaml")
def train(cfg):
    writer = SummaryWriter("runs")
    EPOCHS = cfg.SOLVER.MAX_EPOCHS
    model = build_model(cfg, cfg.MODEL.NAME)
    train_transform = build_transform(cfg, is_train=True)
    train_set = download_dataset(cfg, train_transform, is_train=True)
    train_subset, val_subset = train_val_split(train_set)
    train_loader = prep_data_loader(cfg, train_subset, is_train=True)
    val_loader = prep_data_loader(cfg, val_subset, is_train=False)
    loss = F.cross_entropy
    optimizer = build_optimizer(cfg, model)

    best_val_loss = float("inf")
    patience, trigger = 5, 0
    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_acc = train_one_epoch(cfg, model, optimizer, train_loader, loss)
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)

        model.eval()
        val_loss, val_acc = test_one_epoch(cfg, model, val_loader, loss)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        logger.info(
            f"Epoch {epoch + 1}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if (epoch + 1) % cfg.SOLVER.CHECKPOINT_PERIOD == 0:
            torch.save(model.state_dict(), f"checkpoint_epoch_{epoch + 1}.pth")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            trigger += 1
            if trigger >= patience:
                logger.info("Early stopping triggered")
                break

    test_transform = build_transform(cfg, is_train=False)
    test_set = download_dataset(cfg, transform=test_transform, is_train=False)
    test_loader = prep_data_loader(cfg, test_set, is_train=False)

    model.load_state_dict(torch.load(
        "best_model.pth", map_location=cfg.MODEL.DEVICE))
    model.eval()
    test_loss, test_acc = test_one_epoch(cfg, model, test_loader, loss)
    logger.info(f"Test loss: {test_loss:.4f}  Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    train()
