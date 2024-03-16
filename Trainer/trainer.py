import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from utils.evaluation.base_eval import BaseEval
from utils.logger.base_log import BaseLogger
from Agent.base_agent import BaseAgent
from torch.utils.data import DataLoader


OPTIMIZER = {"adam": torch.optim.Adam}

class DiffusionBCTrainer:
       def __init__(
              self,
              diffusion_agent: BaseAgent,
              train_dataloader: DataLoader,
              val_dataloader: DataLoader,
              logger: BaseLogger,
              evaluator: BaseEval,
              lr: float=1e-4,
              optimizer: str='adam',
              device: str='cpu',
       ):
              # model
              self.diffusion_agent = diffusion_agent
              
              # dataloader
              self.train_dataloader = train_dataloader
              self.val_dataloader = val_dataloader
              
              # optimizer
              self.optimizer = OPTIMIZER[optimizer](self.diffusion_agent.parameters(), lr=lr)
              
              # logger
              self.logger = logger
              self.device = device
              
              # evaluator
              self.evaluator = evaluator
              
       def train_epoch(self, epochs: int):
              """
              train some epochs
              """
              self.diffusion_agent.train()
              self.evaluator.eval_episodes(self.diffusion_agent, 0)
              for epoch in range(0, epochs):
                     epoch_loss = 0.
                     with tqdm(self.train_dataloader, unit="batch") as pbar:
                            for batch in pbar:
                                   cond = batch['s'].to(self.device)
                                   x = batch['a'].to(self.device)
                                   
                                   self.optimizer.zero_grad()
                                   loss = self.diffusion_agent.loss(x, cond)
                                   loss.backward()
                                   self.optimizer.step()
                                   
                                   pbar.set_description(f"Epoch {epoch} Loss: {loss.item():.4f}")
                                   epoch_loss += loss.item()
                     
                     avg_loss = epoch_loss / len(self.train_dataloader)
                     self.logger.log_metrics({"train/loss": avg_loss}, step=epoch)
                     rewards = self.evaluator.eval_episodes(self.diffusion_agent, epoch+1)
                     print(f"Epoch {epoch} Average Loss: {avg_loss:.4f}, Average return: {rewards:.4f}")
              
              self.logger.finish()
                     
                            
       
       def train_steps(self, steps):
              """
              train some gradient steps
              """
              pass
              
       def save_model(self, path: str):
              """
              save the model to path
              """
              torch.save(self.diffusion_agent.state_dict(), path)
              
       def load_model(self, path: str):
              """
              load ckpt from path
              """
              self.diffusion_agent.load_state_dict(torch.load(path, map_location=self.device))