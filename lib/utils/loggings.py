import logging
import os
import time
from logging.handlers import RotatingFileHandler


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    now = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    log_path = os.path.join("log", now)
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")

    file_name = os.path.join("log", os.path.join(now, "{}.log".format(name)))
    log_files_handler = RotatingFileHandler(file_name, encoding="utf-8")
    log_files_handler.setLevel(logging.INFO)
    log_files_handler.setFormatter(formatter)
    logger.addHandler(log_files_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)

    return logger
