import copy

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
              ema: float=1e-3,
              optimizer: str='adam',
              device: str='cpu',
       ):
              # model
              self.diffusion_agent = diffusion_agent
              self.target_diffusion_agent = copy.deepcopy(self.diffusion_agent)
              
              # dataloader
              self.train_dataloader = train_dataloader
              self.val_dataloader = val_dataloader
              
              # optimizer
              self.optimizer = OPTIMIZER[optimizer](self.diffusion_agent.parameters(), lr=lr)
              self.ema = ema
              
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
                                   self.ema_update()
                                   
                     avg_loss = epoch_loss / len(self.train_dataloader)
                     self.logger.log_metrics({"train/loss": avg_loss}, step=epoch)
                     print(f"Epoch {epoch} Average Loss: {avg_loss:.4f}")
                     
                     if (epoch + 1) % self.evaluator.eval_freq == 0:
                            rewards = self.evaluator.eval_episodes(self.target_diffusion_agent, epoch+1)
                            print(f"Epoch {epoch} Average return: {rewards:.4f}")
              
              self.logger.finish()
                     
       def train_steps(self, steps: int):
              """
              train some steps
              """
              self.diffusion_agent.train()
              self.evaluator.eval_episodes(self.target_diffusion_agent, 0)
              
              iterator = iter(self.train_dataloader)
              for step in tqdm(range(steps)):
                     # with tqdm(self.train_dataloader, unit="batch") as pbar:
                     try:
                            batch = next(iterator)
                            
                     except:
                            iterator = iter(self.train_dataloader)
                            batch = next(iterator)
                            
                     cond = batch['s'].to(self.device)
                     x = batch['a'].to(self.device)
                     
                     self.optimizer.zero_grad()
                     loss = self.diffusion_agent.loss(x, cond)
                     loss.backward()
                     self.optimizer.step()
                     self.ema_update()
                     
                     self.logger.log_metrics({"train/loss": loss.item()}, step=step)
                     
                     if (step + 1) % self.evaluator.eval_freq == 0:
                            rewards = self.evaluator.eval_episodes(self.target_diffusion_agent, step)
                            print(f"Epoch {step} Average return: {rewards:.4f}")
              
              self.logger.finish()
       
       def ema_update(self):
              for param, target_param in zip(self.diffusion_agent.parameters(), self.target_diffusion_agent.parameters()):
                     target_param.data.copy_(self.ema * param.data + (1 - self.ema) * target_param.data)
              
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