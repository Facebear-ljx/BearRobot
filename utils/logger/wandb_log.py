import wandb
    
from utils.logger.base_log import BaseLogger

class WandbLogger(BaseLogger):
    def __init__(self, project_name, config):
              super().__init__(config)
              """
              project_name (str): wandb project name
              config (dict): dict
              """              
              wandb.init(project=project_name, 
                         config=config,
                         monitor_gym=False,  # do not record system info (e.g. GPU/CPU utils)
              )

              self.config = wandb.config

    def log_metrics(self, metrics: dict, step: int):
        """
            metrics (dict):
            step (int, optional): epoch or step
        """
        wandb.log(metrics, step=step)

    def finish(self):
        wandb.finish()
