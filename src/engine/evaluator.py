"""Test/Evaluate the model"""
import torch


def test_one_epoch(cfg, model, test_loader, loss_fn):
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            images, labels = data
            images, labels = images.to(
                cfg.MODEL.DEVICE), labels.to(cfg.MODEL.DEVICE)

            outputs = model(images)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / len(test_loader), correct / total
