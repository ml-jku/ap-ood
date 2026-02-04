import json

from ap_ood_experiments.logger.base import Logger


class ConsoleLogger(Logger):
    def log(self, log_dict: dict, epoch=None):
        print(json.dumps(log_dict))
