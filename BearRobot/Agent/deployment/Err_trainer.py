import copy
import io
from mmengine import fileio
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from torch.nn.parallel import DistributedDataParallel as DDP

from tqdm import tqdm

from BearRobot.utils.evaluation.base_eval import BaseEval
from BearRobot.utils.logger.base_log import BaseLogger
from BearRobot.Agent.base_agent import BaseAgent
from torch.utils.data import DataLoader
from BearRobot.Trainer.lr_schedule import CosineAnnealingWarmUpRestarts


OPTIMIZER = {"adam": torch.optim.Adam,
             "adamw": torch.optim.AdamW}

LR_SCHEDULE = {"cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
               "cosinewarm": CosineAnnealingWarmUpRestarts}

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
              self.ddpm_optimizer = OPTIMIZER[optimizer](self.agent.policy.parameters(), lr=lr)
              
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
                     
                     # load from airkitchendataloader_err
                     imgs = batch['imgs'].to(self.device)
                     label = batch['label'].to(self.device)
                     proprio = batch['proprio'].to(self.device)
                     lang = batch['lang']
                     t = batch['t']
                     T = batch['T']
     
                     try:
                            img_begin = batch["img_begin"]
                            img_end =  batch["img_end"]
                     except:
                            img_begin = None
                            img_end = None
                     
                     cond = {"lang": lang,
                            "img_begin": img_begin,
                            "img_end": img_end
                            }
                     
                     # update ddpm policy
                     self.policy_optimizer.zero_grad()
                     p_loss = self.agent.policy(imgs, cond, label, proprio, img_goal=False)
                     p_loss['policy_loss'].backward()
                     self.policy_optimizer.step()
                     
                     # not in use
                     # self.ema_update_policy()
                     
                     # update v
                     self.v_optimizer.zero_grad()
                     v_loss, v_mean = self.agent.v_loss(imgs, proprio, t, T)
                     v_loss.backward()
                     self.v_optimizer.step()
                     self.scheduler.step()
                     
                     # update q (not in use)
                     # self.q_optimizer.zero_grad()
                     # q_loss = self.agent.q_loss(s, a, r, next_s, d)
                     # q_loss.backward()
                     # self.q_optimizer.step()
                     # self.ema_update_q()
                     
                     # log the training process
                     if (step + 1) % self.logger.record_freq == 0:
                            self.logger.log_metrics({"train/policy_loss": p_loss.item(),
                                                 "train/v_loss": v_loss.item(),
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