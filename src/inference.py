"""Conformal prediction inference for a single input sample."""
import hydra
import torch
from PIL import Image
from utils import get_logger
from models import build_model
from data_utils import build_transform
from cp import prediction_sets

logger = get_logger(__name__)


def inference(cfg, image):
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

    transform = build_transform(cfg, is_train=False)

    with torch.no_grad():
        if isinstance(image, torch.Tensor):
            image = image.unsqueeze(0) if image.ndim == 3 else image
            image = image.to(cfg.MODEL.DEVICE)
        else:
            image = transform(image).unsqueeze(0).to(cfg.MODEL.DEVICE)
        logits = model(image)
        probas = torch.softmax(logits, dim=1)
        pred_set = prediction_sets(probas, quantile, n_classes)[0]

    logger.info(f"set={sorted(pred_set)} size={len(pred_set)}")
    return pred_set


@hydra.main(version_base=None, config_path="../config", config_name="config.yaml")
def main(cfg):
    image_path = cfg.get("IMAGE_PATH")
    if not image_path:
        raise ValueError("IMAGE_PATH config is required (e.g. IMAGE_PATH=/path/to/img.png)")

    image = Image.open(image_path).convert("RGB")
    logger.info(f"Running inference on {image_path}")
    pred_set = inference(cfg, image)
    logger.info(f"Result: set_size={len(pred_set)}")


if __name__ == "__main__":
    main()