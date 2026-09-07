"""Inference with conformal prediction sets."""
import os
import hydra
import torch
from utils import get_logger
from models import build_model
from data_utils import download_dataset, prep_data_loader, build_transform
from cp import prediction_sets

logger = get_logger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="config.yaml")
def inference(cfg):
    if not cfg.CHECKPOINT:
        raise ValueError("CHECKPOINT required for inference")

    model = build_model(cfg, cfg.MODEL.NAME)
    ckpt = torch.load(cfg.CHECKPOINT, map_location=cfg.MODEL.DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    quantile = ckpt["quantile"] if "quantile" in ckpt else 0.0
    alpha = ckpt.get("alpha", cfg.CP.ALPHA)
    n_classes = ckpt.get("num_classes", cfg.MODEL.NUM_CLASSES)
    logger.info(f"Using quantile q={quantile:.4f} (alpha={alpha})")

    test_transform = build_transform(cfg, is_train=False)
    test_set = download_dataset(cfg, transform=test_transform, is_train=False)
    test_loader = prep_data_loader(cfg, test_set, is_train=False)

    correct = 0
    total = 0
    set_sizes = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(cfg.MODEL.DEVICE), labels.to(cfg.MODEL.DEVICE)
            logits = model(images)
            probas = torch.softmax(logits, dim=1)
            sets = prediction_sets(probas, quantile, n_classes)

            labels_np = labels.cpu().numpy()
            for _, (pred_set, label) in enumerate(zip(sets, labels_np)):
                covered = int(label in pred_set)
                correct += covered
                total += 1
                set_sizes.append(len(pred_set))
                logger.info(
                    f"Sample {total}: true={label} set={sorted(pred_set)} "
                    f"size={len(pred_set)} covered={bool(covered)}")

    coverage = correct / total
    avg_set_size = sum(set_sizes) / len(set_sizes)
    logger.info(
        f"Test coverage: {coverage:.4f} (target {1 - alpha:.4f})  "
        f"Avg prediction set size: {avg_set_size:.4f}")


if __name__ == "__main__":
    inference()
