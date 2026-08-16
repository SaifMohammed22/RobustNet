"""The training epoch function"""


def train_one_epoch(cfg, model, optimizer, train_loader, loss_fn):
    running_loss = 0.0
    correct = 0
    total = 0

    for data in train_loader:
        images, labels = data
        images, labels = images.to(
            cfg.MODEL.DEVICE), labels.to(cfg.MODEL.DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / len(train_loader), correct / total
