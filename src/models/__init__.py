from .model import CNN


def build_model(cfg, is_train=True):
    model = CNN(cfg.MODEL.NUM_CLASSES)
    return model.train() if is_train else model.eval()
