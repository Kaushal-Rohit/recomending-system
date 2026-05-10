import logging


def setup_logging() -> None:
    if getattr(setup_logging, "_configured", False):
        return

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('recommendation_model.log'),
            logging.StreamHandler()
        ]
    )
    setup_logging._configured = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
