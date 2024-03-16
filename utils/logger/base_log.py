class BaseLogger():
       def __init__(self, config):
            self.config = config
       
       def log_metrics(self, metrics: dict, step: int):
              """
              log the evaluation results
              """
              raise NotImplementedError

       def finish(self):
              """
              close the logger
              """
              raise NotImplementedError