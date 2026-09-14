import hydra
import torch
import torch.nn.functional as F
from models import build_model
from engine import test_one_epoch
from data_utils import download_dataset, prep_data_loader, build_transform
from cp import prediction_sets


@hydra.main(version_base=None, config_path="../config", config_name="config.yaml")
def eval(cfg):
    if not cfg.CHECKPOINT:
        raise ValueError("CHECKPOINT required for eval mode")
    model = build_model(cfg, cfg.MODEL.NAME)
    ckpt = torch.load(cfg.CHECKPOINT, map_location=cfg.MODEL.DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_transform = build_transform(cfg, is_train=False)
    test_set = download_dataset(
        cfg, transform=test_transform, is_train=False)
    test_loader = prep_data_loader(cfg, test_set, is_train=False)
    loss = F.cross_entropy
    test_loss, test_acc = test_one_epoch(cfg, model, test_loader, loss)
    test_error = 1 - test_acc
    print(
        f"Test loss: {test_loss:.4f}  Test accuracy: {test_acc:.4f}  Test Error Rate: {test_error:.4f}")

    quantile = ckpt.get("quantile")
    if quantile is None:
        print("No conformal quantile in checkpoint; run train.py to compute one.")
        return

    alpha = ckpt.get("alpha", cfg.CP.ALPHA)
    n_classes = ckpt.get("num_classes", cfg.MODEL.NUM_CLASSES)

    correct = 0
    total = 0
    set_sizes = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(cfg.MODEL.DEVICE)
            outputs = model(images)
            probas = torch.softmax(outputs, dim=1)
            sets = prediction_sets(probas, quantile, n_classes)
            labels_np = labels.numpy()
            for pred_set, label in zip(sets, labels_np):
                correct += int(label in pred_set)
                total += 1
                set_sizes.append(len(pred_set))

    coverage = correct / total
    avg_set_size = sum(set_sizes) / total
    print(
        f"Conformal coverage: {coverage:.4f} (target {1 - alpha:.4f})  "
        f"Avg prediction set size: {avg_set_size:.4f}")


if __name__ == "__main__":
    eval()