from ap_ood_experiments.logger.base import Logger


class ComposedLogger(Logger):
    def __init__(self, loggers):
        self.loggers = loggers
    
    def log(self, log_dict: dict, epoch=None):
        for logger in self.loggers:
            logger.log(log_dict, epoch=epoch)
