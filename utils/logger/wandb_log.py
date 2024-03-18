import wandb
    
from utils.logger.base_log import BaseLogger

class WandbLogger(BaseLogger):
    def __init__(self, project_name, run_name, config, record_freq: int=100):
              super().__init__(config)
              """
              project_name (str): wandb project name
              config (dict): dict
              """              
              wandb.init(project=project_name, 
                         config=config,
                         monitor_gym=False,  # do not record system info (e.g. GPU/CPU utils)
              )

              wandb.run.name = run_name
              self.config = wandb.config
              self.record_freq = record_freq

    def log_metrics(self, metrics: dict, step: int):
        """
            metrics (dict):
            step (int, optional): epoch or step
        """
        wandb.log(metrics, step=step)

    def finish(self):
        wandb.finish()
