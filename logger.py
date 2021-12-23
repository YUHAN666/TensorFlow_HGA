import logging
import os
import time


class Logger(object):

    def __init__(self, log_path="./log"):

        self.log_path = log_path
        if not os.path.exists(self.log_path):
            os.mkdir(self.log_path)
        timer = time.strftime("%Y-%m-%d-%H-%M-%S_", time.localtime())
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(levelname)s]   %(asctime)s    %(message)s')
        txthandle = logging.FileHandler((self.log_path + '/' + timer + 'log.txt'))
        txthandle.setFormatter(formatter)
        self.logger.addHandler(txthandle)

    def debug(self, string):
        self.logger.debug(string)

    def info(self, string):
        self.logger.info(string)

    def warning(self, string):
        self.logger.warning(string)

    def error(self, string):
        self.logger.error(string)

    def critical(self, string):
        self.logger.critical(string)

