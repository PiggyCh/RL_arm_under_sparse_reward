import logging
import os.path
import time
from colorama import Fore, Style
import sys

class Logger(object):
    def __init__(self, logger):
        self.start_time = time.time()
        self.logger = logging.getLogger(name=logger)
        self.logger.setLevel(logging.DEBUG) 
        rq = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))
        log_path = os.getcwd() + "/logs/"
        log_name = log_path + rq + ".log"
        if not self.logger.handlers:
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.DEBUG)
            formatter = logging.Formatter(
                "%(asctime)s - %(filename)s[line:%(lineno)d] - %(name)s - %(message)s")
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

    def _get_past_time(self):
        s_time = int(time.time() - self.start_time)
        day = s_time // (24 * 3600)
        s_time = s_time % (24 * 3600)
        hour = s_time // 3600
        s_time = s_time % 3600
        minutes = s_time // 60
        s_time = s_time % 60
        return f'day {day} - {hour}h:{minutes}m:{s_time}s'

    def debug(self, msg):
        self.logger.debug(Fore.WHITE + "DEBUG - " + str(msg +'      past time : '+self._get_past_time()) + Style.RESET_ALL)
        
    def info(self, msg):
        self.logger.info(Fore.GREEN + "INFO - " + str(msg +'      past time : '+self._get_past_time()) + Style.RESET_ALL)

    def warning(self, msg):
        self.logger.warning(Fore.RED + "WARNING - " + str(msg +'      past time : '+self._get_past_time()) + Style.RESET_ALL)

    def error(self, msg):
        self.logger.error(Fore.RED + "ERROR - " + str(msg +'      past time : '+self._get_past_time()) + Style.RESET_ALL)

    def critical(self, msg):
        self.logger.critical(Fore.RED + "CRITICAL - " + str(msg +'      past time : '+self._get_past_time()) + Style.RESET_ALL)

if __name__ == '__main__':
    log = Logger(logger="test")
    log.debug("debug")
    log.info("info")
    log.error("error")
    log.warning("warning")
    log.critical("asdasdasdqwfqf")