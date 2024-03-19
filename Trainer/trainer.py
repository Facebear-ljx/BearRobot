import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from tqdm import tqdm

from utils.evaluation.base_eval import BaseEval
from utils.logger.base_log import BaseLogger
from Agent.base_agent import BaseAgent
from torch.utils.data import DataLoader


OPTIMIZER = {"adam": torch.optim.Adam,
             "adamw": torch.optim.AdamW}

LR_SCHEDULE = {"cosine": torch.optim.lr_scheduler.CosineAnnealingLR}


class DiffusionBCTrainer:
       def __init__(
              self,
              diffusion_agent: BaseAgent,
              train_dataloader: DataLoader,
              val_dataloader: DataLoader,
              logger: BaseLogger,
              evaluator: BaseEval,
              num_steps: int,
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
              
              # learning rate schedule
              self.scheduler = LR_SCHEDULE['cosine'](self.optimizer, num_steps)
              
              # logger
              self.logger = logger
              self.device = device
              
              # evaluator
              self.evaluator = evaluator
              
              self.num_steps = num_steps
              
       def train_epoch(self):
              """
              train some epochs
              """
              epochs = self.num_steps
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
                            
                            self.scheduler.step()
                            
                     avg_loss = epoch_loss / len(self.train_dataloader)
                     self.logger.log_metrics({"train/loss": avg_loss}, step=epoch)
                     print(f"Epoch {epoch} Average Loss: {avg_loss:.4f}")
                     
                     if (epoch + 1) % self.evaluator.eval_freq == 0:
                            rewards = self.evaluator.eval_episodes(self.target_diffusion_agent, epoch+1)
                            print(f"Epoch {epoch} Average return: {rewards:.4f}")
              
              self.logger.finish()
                     
       def train_steps(self):
              """
              train some steps
              """
              steps = self.num_steps
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
                     loss = self.diffusion_agent.policy_loss(x, cond)
                     loss.backward()
                     self.optimizer.step()
                     self.ema_update()
                     self.scheduler.step()
                     
                     self.logger.log_metrics({"train/policy_loss": loss.item(),
                                              "train/lr": self.scheduler.get_last_lr()[0]}, step=step)
                     
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
              
              
class RLTrainer:
       def __init__(
              self,
              agent: BaseAgent,
              train_dataloader: DataLoader,
              val_dataloader: DataLoader,
              logger: BaseLogger,
              evaluator: BaseEval,
              num_steps: int,
              lr: float=1e-4,
              policy_ema: float=1e-3,
              critic_ema: float=5e-3,
              optimizer: str='adam',
              device: str='cpu',
       ):
              # model
              self.agent = agent
              
              # dataloader
              self.train_dataloader = train_dataloader
              self.val_dataloader = val_dataloader
              
              # optimizer
              self.policy_optimizer = OPTIMIZER['adamw'](self.agent.policy.parameters(), lr=lr)
              self.v_optimizer = OPTIMIZER[optimizer](self.agent.v_model.parameters(), lr=lr)
              self.q_optimizer = OPTIMIZER[optimizer](self.agent.q_models.parameters(), lr=lr)
              self.policy_ema = policy_ema
              self.critic_ema = critic_ema
              
              # learning rate schedule
              self.scheduler = LR_SCHEDULE['cosine'](self.policy_optimizer, num_steps * 2)
              
              # logger
              self.logger = logger
              self.device = device
              
              # evaluator
              self.evaluator = evaluator
              
              self.num_steps = num_steps
       
       def train_steps(self):
              """
              train some steps
              """
              steps = self.num_steps
              # self.agent.eval()
              # self.evaluator.eval_episodes(self.agent, 0)
              self.agent.train()
              
              iterator = iter(self.train_dataloader)
              for step in tqdm(range(steps)):
                     try:
                            batch = next(iterator)
                     except:
                            iterator = iter(self.train_dataloader)
                            batch = next(iterator)
                     for key in batch.keys(): 
                            batch[key] = batch[key].to(self.device)
                     
                     s, a, r, next_s, d = batch['s'], batch['a'], batch['r'], batch['next_s'], batch['d']
                     
                     # update policy
                     self.policy_optimizer.zero_grad()
                     p_loss = self.agent.policy_loss(a, s)
                     p_loss.backward()
                     self.policy_optimizer.step()
                     self.scheduler.step()
                     self.ema_update_policy()
                     
                     # update v
                     self.v_optimizer.zero_grad()
                     v_loss, v_mean = self.agent.v_loss(s[:256], a[:256])
                     v_loss.backward()
                     self.v_optimizer.step()
                     
                     # update q
                     self.q_optimizer.zero_grad()
                     q_loss = self.agent.q_loss(s[:256], a[:256], r[:256], next_s[:256], d[:256])
                     q_loss.backward()
                     self.q_optimizer.step()
                     self.ema_update_q()
                     
                     # log the training process
                     if (step + 1) % self.logger.record_freq == 0:
                            self.logger.log_metrics({"train/policy_loss": p_loss.item(),
                                                 "train/v_loss": v_loss.item(),
                                                 "train/q_loss": q_loss.item(),
                                                 "train/v_mean": v_mean.item(),
                                                 "train/lr": self.scheduler.get_last_lr()[0]}, step=step)
                     
                     # evaluate
                     if (step + 1) % self.evaluator.eval_freq == 0:
                            self.agent.eval()
                            rewards = self.evaluator.eval_episodes(self.agent, step)
                            print(f"Epoch {step} Average return: {rewards:.4f}")
                            self.agent.train()
              
              self.logger.finish()
       
       
       def ema_update_policy(self):
              for param, target_param in zip(self.agent.policy.parameters(), self.agent.policy_target.parameters()):
                     target_param.data.copy_(self.policy_ema * param.data + (1 - self.policy_ema) * target_param.data)
       
       
       def ema_update_q(self):
              for param, target_param in zip(self.agent.q_models.parameters(), self.agent.q_models_target.parameters()):
                     target_param.data.copy_(self.critic_ema * param.data + (1 - self.critic_ema) * target_param.data)