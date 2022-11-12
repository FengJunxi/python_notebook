import logging
import time

import logging
from logging import handlers
from logger import create_logger
from termcolor import colored


class Logger(object):
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }  # 日志级别关系映射

    def __init__(self, filename, level='info', when='S', backCount=3, ):
        fmt = '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
        # color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
        #             colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)  # 设置日志格式
        self.logger.setLevel(self.level_relations.get(level))  # 设置日志级别
        stream_handler = logging.StreamHandler()  # 往屏幕上输出
        stream_handler.setFormatter(format_str)  # 设置屏幕上显示的格式
        # 实例化TimedRotatingFileHandler
        time_handler = handlers.TimedRotatingFileHandler(filename=filename, when=when, backupCount=backCount, encoding='utf-8')  # 往文件里写入指定间隔时间自动生成文件的处理器
        # interval是时间间隔，backupCount是备份文件的个数，如果超过这个个数，就会自动删除，when是间隔的时间单位，单位有以下几种：
        # S 秒
        # M 分
        # H 小时、
        # D 天、
        # W 每星期（interval==0时代表星期一）
        # midnight 每天凌晨
        time_handler.setFormatter(format_str)  # 设置文件里写入的格式
        self.logger.addHandler(stream_handler)  # 把对象加到logger里
        self.logger.addHandler(time_handler)


def mylogger_test():
    log = Logger('all.log', level='debug')
    log.logger.debug('debug')
    log.logger.info('info')
    log.logger.warning('警告')
    log.logger.error('报错')
    log.logger.critical('严重')
    # error_logger = Logger('error.log', level='error')
    # error_logger.logger.error('error')


def log_test():
    LOG_FORMAT = "%(asctime)s -%(pathname)s - %(lineno)s-%(levelname)s - %(message)s"
    # logging.basicConfig(filename='my.log', level=logging.DEBUG, format=LOG_FORMAT)
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    logging.debug("This is a debug log.")
    time.sleep(1)
    logging.info("This is a info log.")
    logging.warning("This is a warning log.")
    logging.error("This is a error log.")
    logging.critical("This is a critical log.")


class CustomFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def custom_logger_test():
    # create logger with 'spam_application'
    logger = logging.getLogger("My_app")
    logger.setLevel(logging.DEBUG)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    ch.setFormatter(CustomFormatter())

    logger.addHandler(ch)

    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")

if __name__ == '__main__':
    # import os
    # os.system("color")  # enables ansi escape characters in terminal
    # log_test()
    mylogger_test()
    # 通过字典设置参数
    # site = {"url": "www.runoob.com", "name": "菜鸟教程"}
    # print("{url}, {name}".format(**site))

    # OUTPUT = "./"
    # logger = create_logger(output_dir=OUTPUT, dist_rank=0, name="name")
    # #
    # logger.info("test")


