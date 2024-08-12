""""
This is a utility script for generating logs
"""
import logging


def get_logger(name):
    """
    Returns a logging object
    Args:
        name(str): Name of the module for which logging
    Return:
        logger(obj): Logging object
    """
    file_formatter = logging.Formatter(
        "%(asctime)s~%(levelname)s~%(message)s~module:%(module)s~function:%(module)s"
    )
    console_formatter = logging.Formatter("%(levelname)s -- %(message)s")

    file_handler = logging.FileHandler("logfile.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    logger = logging.getLogger(name)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)

    return logger
