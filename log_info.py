import logging
import sys

loggers = {}
formatter = logging.Formatter("[ %(levelname)s ] %(message)s")


def setup_logger(name: str, level: int = logging.INFO, log_file: str = None):
    """ "
    Method to create loggers across the project
    Usage:
        logger_name = setup(name)
    """
    if loggers.get(name):
        return loggers.get(name)
    else:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        if log_file is not None:
            handler = logging.FileHandler(log_file)

        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.addHandler(handler)

        loggers[name] = logger

        return logger
