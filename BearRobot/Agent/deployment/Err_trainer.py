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
              save:bool=True,
              save_freq:int=10000, 
              save_path:str=None,
              resume_path: str=None,
              args=None,
              **kwargs,
       ):
              # model
              self.agent = agent
              
              # dataloader
              self.train_dataloader = train_dataloader
              self.val_dataloader = val_dataloader
              
              # optimizer
              self.policy_optimizer = OPTIMIZER['adamw'](self.agent.policy.parameters(), lr=lr)
              self.v_optimizer = OPTIMIZER[optimizer](self.agent.v_model.parameters(), lr=lr)
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

              # save
              self.save = save
              self.save_freq = save_freq
              self.save_path = save_path
              
       def save_model(self, step, loss, save_optimizer=False, save_schedule=False):
              """
              save the model to path
              """
              save_model = {# 'model': self.agent.state_dict(), 
                            'v_model': self.agent.v_model.state_dict(), 
                            'step': step}

              with io.BytesIO() as f:
                     torch.save(save_model, f)
                     fileio.put(f.getvalue(), f"{self.save_path}/{step}_{loss}.pth")
                     fileio.put(f.getvalue(), f"{self.save_path}/latest.pth")
                     
       def train_steps(self):
              """
              train some steps
              """
              steps = self.num_steps
              # self.agent.eval()
              # self.evaluator.eval_episodes(self.agent, 0)
              self.agent.train()
              self.agent.policy.train()
              
              epoch = 0
              self.train_dataloader.sampler.set_epoch(epoch)
              iterator = iter(self.train_dataloader)
              with tqdm(range(steps)) as pbar:
                     
                     for step in pbar:
                            try:
                                   batch = next(iterator)
                            except:
                                   iterator = iter(self.train_dataloader)
                                   batch = next(iterator)

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
                            # self.policy_optimizer.zero_grad()
                            # p_loss = self.agent.policy(imgs, cond, label, proprio, img_goal=False)
                            # p_loss['policy_loss'].backward()
                            # self.policy_optimizer.step()
                            
                            # not in use
                            # self.ema_update_policy()
                            
                            # update v
                            self.v_optimizer.zero_grad()
                            v_loss = self.agent.v_loss(imgs, proprio, t, T)
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
                            pbar.set_description(f"Step {step}: v_loss:{v_loss}")
                            # pbar.set_description(f"Step {step}: p_loss:{p_loss['policy_loss'].item():.4f}")
                            # loss_log = {f"train/{key}": value.item() for key, value in p_loss.items()}
                            # self.logger.log_metrics(loss_log, step=step)
                            self.logger.log_metrics({
                                                 "train/v_loss": v_loss.item(),
                                                 "train/lr": self.scheduler.get_last_lr()[0]}, step=step)

                            if self.evaluator:
                            # evaluate
                                   if (step + 1) % self.evaluator.eval_freq == 0:
                                          self.agent.eval()
                                          rewards = self.evaluator.eval_episodes(self.agent, step)
                                          print(f"Epoch {step} Average return: {rewards:.4f}")
                                          self.agent.train()
                            if self.save:
                                   if (step + 1) % self.save_freq == 0:
                                          self.save_model(step,v_loss)
              
              self.logger.finish()
       
       
       def ema_update_policy(self):
              for param, target_param in zip(self.agent.policy.parameters(), self.agent.policy_target.parameters()):
                     target_param.data.copy_(self.policy_ema * param.data + (1 - self.policy_ema) * target_param.data)
       
       
       def ema_update_q(self):
              for param, target_param in zip(self.agent.q_models.parameters(), self.agent.q_models_target.parameters()):
                     target_param.data.copy_(self.critic_ema * param.data + (1 - self.critic_ema) * target_param.data)