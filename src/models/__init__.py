from .model import ResNet20, ResNet32, ResNet44, ResNet56

_MODELS = {
    "ResNet20": ResNet20,
    "ResNet32": ResNet32,
    "ResNet44": ResNet44,
    "ResNet56": ResNet56,   
}


def build_model(cfg, model_name):
    if model_name not in _MODELS:
        raise NotImplementedError(f"Unknown model: {model_name}")
    model = _MODELS[model_name](cfg.MODEL.NUM_CLASSES, channels=3)
    return model
