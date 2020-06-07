import logging
from logging import Logger
from logging.handlers import TimedRotatingFileHandler
import sys

def init_logger(logger_name, logging_path):
    if logger_name not in Logger.manager.loggerDict:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        consolehandler = logging.StreamHandler(sys.stdout)
        consolehandler.setLevel(logging.FATAL)

        handler = TimedRotatingFileHandler(filename=logging_path, when='D', backupCount = 7)
        datefmt = '%Y-%m-%d %H:%M:%S'
        format_str = '[%(asctime)s]: %(name)s %(filename)s[line:%(lineno)s] %(levelname)s  %(message)s'
        formatter = logging.Formatter(format_str, datefmt)
        handler.setFormatter(formatter)
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)
        logger.addHandler(consolehandler)
    logger = logging.getLogger(logger_name)
    return logger
