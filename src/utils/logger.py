import hydra
import logging


def get_logger(name=__name__):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(name)s: %(message)s",
            datefmt = "%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        file_handler = logging.FileHandler(f"{hydra.utils.get_original_cwd()}/outputs/{hydra.utils.get_run_dir().split('/')[-2]}/train.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger