"""Test/Evaluate the model"""
import torch


def test_one_epoch(cfg, model, test_loader, loss_fn):
    running_loss = 0.0
    last_loss = 0.0

    model.eval()
    with torch.no_grad:
        for i, data in enumerate(test_loader):
            images, labels = data

            outputs = model(images)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item()
            if i % cfg.SOLVER.LOG_PERIOD == 0:
                        last_loss = running_loss / cfg.SOLVER.LOG_PERIOD
                        print(f"    batch {i + 1} test loss: {last_loss}")
                        running_loss = 0.0
    return last_loss
    
