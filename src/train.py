import os
import hydra
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from utils import build_optimizer, build_scheduler, get_logger
from models import build_model
from engine import train_one_epoch, test_one_epoch
from data_utils import download_dataset, prep_data_loader, build_transform, train_val_split

logger = get_logger(__name__)
os.makedirs("checkpoints", exist_ok=True)


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
    scheduler = build_scheduler(cfg, optimizer)

    start_epoch = 0
    best_val_loss = float("inf")
    # Early stopping
    patience, trigger = 10, 0

    if cfg.RESUME and cfg.CHECKPOINT:
        ckpt = torch.load(cfg.CHECKPOINT, map_location=cfg.MODEL.DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"]
        best_val_loss = ckpt["best_val_loss"]
        trigger = ckpt["trigger"]
        logger.info(f"Resume from epoch {start_epoch}")

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        train_loss, train_acc = train_one_epoch(
            cfg, model, optimizer, train_loader, loss)
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)

        model.eval()
        val_loss, val_acc = test_one_epoch(cfg, model, val_loader, loss)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        logger.info(
            f"Epoch {epoch + 1}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        scheduler.step()

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "trigger": trigger,
        }

        if (epoch + 1) % cfg.SOLVER.CHECKPOINT_PERIOD == 0:
            torch.save(
                checkpoint, f"checkpoints/checkpoint_epoch_{epoch + 1}.pth")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger = 0
            checkpoint["best_val_loss"] = best_val_loss
            checkpoint["trigger"] = trigger
            torch.save(checkpoint, "best_model.pth")
        else:
            trigger += 1
            if trigger >= patience:
                logger.info("Early stopping triggered")
                break

    test_transform = build_transform(cfg, is_train=False)
    test_set = download_dataset(cfg, transform=test_transform, is_train=False)
    test_loader = prep_data_loader(cfg, test_set, is_train=False)

    ckpt = torch.load("best_model.pth", map_location=cfg.MODEL.DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    test_loss, test_acc = test_one_epoch(cfg, model, test_loader, loss)
    test_error = 1 - test_acc
    logger.info(f"Test loss: {test_loss:.4f}  Test accuracy: {test_acc:.4f}  Test Error Rate: {test_error:.4f}")


if __name__ == "__main__":
    train()
