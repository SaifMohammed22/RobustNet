from .model import ResNet50


def build_model(cfg, is_train=True):
    model = ResNet50(cfg.MODEL.NUM_CLASSES, channels=3)
    return model.train() if is_train else model.eval()
