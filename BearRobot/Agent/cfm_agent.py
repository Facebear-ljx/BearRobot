import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import copy

from PIL import Image

from BearRobot.Agent.base_agent import BaseAgent
from BearRobot.Net.my_model.diffusion_model import VisualDiffusion
from BearRobot.Net.my_model.t5 import T5Encoder
from BearRobot.Net.encoder.DecisionNCE import DecisionNCE_encoder, DecisionNCE_lang, DecisionNCE_visual_diff

from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import Solver, ODESolver
from flow_matching.utils import ModelWrapper

def extract(a, x_shape):
       '''
       align the dimention of alphas_cumprod_t to x_shape
       
       a: alphas_cumprod_t, B
       x_shape: B x F x F x F
       output: alphas_cumprod_t B x 1 x 1 x 1]
       '''
       b, *_ = a.shape
       return a.reshape(b, *((1,) * (len(x_shape) - 1)))

                     
class CFM_BC(BaseAgent):
       def __init__(
              self, 
              policy: torch.nn.Module, 
              # beta: str='cosine',
              T: int=5,  # only used for sampling
              ac_num: int=1,
       ):
              super().__init__(
                     policy, None, None, ac_num=ac_num
              )
              
              self.device = policy.device
              self.num_timesteps = T

       def forward(self, xt: torch.Tensor, t: torch.Tensor, cond: torch.Tensor=None):
              """
              predict the velocity
              Input:
              xt: the samples from p_t(x_t|x1), x1 is the ground truth action and x0 is typically sampled from the source distribution
              t: timestep
              cond: condition information, here cond is typically the state rather than other condition like language or goal image
              
              Return: predicted velocity
              """
              v_pred = self.policy(xt, t, cond)
              return v_pred
       
       def policy_loss(self, x1: torch.Tensor, cond: torch.Tensor=None):
              '''
              calculate FM loss
              
              Input:
              x1: ground truth value, here x1 is typically the ground truth action
              cond: condition information, here cond is typically the state rather than other condition like language or goal image
              
              Return: flow matching loss
              '''
              batch_size = x1.shape[0]
              
              x0 = torch.randn_like(x1, device=self.device)
              t = torch.rand((batch_size, 1)).to(x1.device)
              
              xt, v_gt = self.q_sample(x0, x1, t)
              
              v_pred = self.forward(xt, t, cond)
              loss = (((v_pred - v_gt) ** 2).sum(axis = -1)).mean()
              
              return loss
              
       def q_sample(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor):
              """
              construct the conditional propability path p_t(x_t|x0, x1) and returns x_t and the ground truth velocity
              
              Input:
              x0: source distribution samples
              x1: target distribution samples (ground truth)
              t: timestep
              
              Return: 
              xt: xt ~ pt(xt|x0, x1)
              v_gt: ground truth velocity
              """

              # currently, we only support CondOT path
              xt = (1 - t) * x0 + t * x1
              v_gt = x1 - x0
              return xt, v_gt
       
       def wrapped_model(self):
              class WrappedModel(ModelWrapper):
                     def __init__(
                            self, 
                            cond,
                     ):
                            super().__init__()
                            self.cond = cond
                     
                     def forward(self, x: torch.Tensor, t: torch.Tensor):
                            t = t.reshape(1, 1).repeat(x.shape[0], 1)
                            return self.model(x, t, self.cond)
              
              self.wrapped_policy = WrappedModel(self.policy)    
       
       
       @torch.no_grad()
       def sampler(self, shape, cond=None, guidance_strength=0, clip_sample=False):
              """
              sample x1 (target) from x0 (source)
              
              Input:
              shape: the desired shape of the sample
              cond: condition information, here cond is typically the state rather than other condition like language or goal image
              guidance_strength: the strength of the guidance
              clip_sample: whether to clip the sample to [-1, 1]
              
              Return: sampled data
              """
              # if not hasattr(self, 'wrapped_policy'):
              self.wrapped_model(cond)
              x_init = torch.randn(shape).to(self.device)
              T = torch.linspace(0, 1, self.num_timesteps)  # sample times
              step_size = 1 / T
              
              solver = ODESolver(velocity_model=self.wrapped_policy)  # create an ODESolver class
              sol = solver.sample(time_grid=T, x_init=x_init, method='midpoint', step_size=step_size, return_intermediates=False)  # sample from the model
              return sol

       @torch.no_grad()
       def get_action(self, state, num=1, clip_sample=False):
              return self.sampler((num, self.policy.output_dim), cond=state, clip_sample=clip_sample)
              
              

class VLCFM_BC(CFM_BC):
       """
       this is a vision-language flow matching agent, based on ConOT-FM
       """
       def __init__(
              self, 
              policy: VisualDiffusion, 
              # beta: str='cosine',  
              T: int=5,
              ac_num: int=1,
              text_encoder: str="T5",
              device = 'cuda',
              *args, **kwargs
       ):
              super().__init__(policy, T, ac_num)
              
              # self.img_size = self.policy.img_size
              assert text_encoder in ['T5', 'DecisionNCE-T', 'DecisionNCE-P', "DecisionNCE-V"]
              if text_encoder == 'T5':
                     self.lang_encoder = T5Encoder(device=device)
              elif text_encoder in ['DecisionNCE-T', 'DecisionNCE-P']:
                     mm_encoder = DecisionNCE_encoder(text_encoder, device=device)
                     self.lang_encoder = DecisionNCE_lang(mm_encoder)
              elif text_encoder == "DecisionNCE-V":
                     text_encoder = 'DecisionNCE-T'
                     mm_encoder = DecisionNCE_encoder(text_encoder, device=device)
                     self.frame_diff_encoder = DecisionNCE_visual_diff(mm_encoder, *args, **kwargs)
              else:
                     raise ValueError(f"Invalid text_encoder '{text_encoder}'. Expected one of: ['t5', 'DecisionNCE-T', 'DecisionNCE-P']")
              print("lang encoder load success")
              self.device = device
              
       def forward(self, images: torch.Tensor, cond: dict, action_gt: torch.Tensor, state=None, img_goal=False):
              '''
              calculate flow matchin loss
              # images: batch of frames of different views, [B, Frame, View, C, H, W]
              # texts: list of instructions
              # state shape [B, D_s], batch of robot arm x,y,z, gripper state, et al
              # action_gt shape [B, D_a], batch of robot control value, e.g., delta_x, delta_y, delta_z,..., et al.
              # img_begin: one begin frame of a video
              # img_end: one end frame of a video
              # img_goal: bool, whether to use img_begin and img_end to calculate visual difference as goal
              '''
              img_begin = cond['img_begin']
              img_end = cond['img_end']
              texts = cond['lang']
       
              if img_goal:
                     if img_begin != None and img_end != None:      
                            text_emb_visiual_diff = self.frame_diff_encoder.embed_frame(img_begin, img_end).to(images.device).detach()
                            loss = self.policy_loss(action_gt, images, text_emb_visiual_diff, state)
                            loss_dict = {"policy_loss": loss}
                            return loss_dict
                     else:
                            raise ValueError("img_begin or img_end is None")
              else:
                     text_emb = self.lang_encoder.embed_text(texts).to(images.device).detach()
                     loss = self.policy_loss(action_gt, images, text_emb, state)
                     loss_dict = {"policy_loss": loss}
                     return loss_dict
       
       
       def policy_loss(self, x1: torch.Tensor, imgs: torch.Tensor, condition: torch.Tensor, state: torch.Tensor=None):
              '''
              calculate flow matchin loss
              Input:
              x1: [B, D_a] ground truth action
              imgs: [B, F, V, C, H, W] batch of frames of different views
              condition: [B, D] batch of instruction embbeding
              state: [B, D_s] batch of robot arm x,y,z, gripper state
              
              return: loss
              '''
              batch_size = x1.shape[0]
              
              x0 = torch.randn_like(x1, device=self.device)
              t = torch.randint(0, self.num_timesteps, (batch_size, ), device=self.device)
              xt, v_gt = self.q_sample(x0, x1, t)
              
              v_pred = self.policy(xt, t, imgs, condition, state)
              loss = (((v_pred - v_gt) ** 2).sum(axis = -1)).mean()
              return loss

       def wrapped_model(self):
              class WrappedModel(ModelWrapper):
                     def __init__(
                            self, 
                            imgs,
                            cond,
                            state,
                     ):
                            super().__init__()
                            self.imgs = imgs
                            self.cond = cond
                            self.state = state
                     
                     def forward(self, x: torch.Tensor, t: torch.Tensor):
                            t = t.reshape(1, 1).repeat(x.shape[0], 1)
                            return self.model(x, t, self.imgs, self.cond, self.state)
              
              self.wrapped_policy = WrappedModel(self.policy)    
       
       
       @torch.no_grad()
       def ddpm_sampler(self, shape, imgs: torch.Tensor, lang: list, state: torch.Tensor=None, guidance_strength=0, clip_sample=False, img_begin=None, img_end=None, img_goal=False):
              """
              sample x1 from x0, reverse process.
              
              Input:
              shape: [num_samples, D_a], desired shapes of samples
              imgs: [B, F, V, C, H, W] batch of frames of different views
              lang: list of instructions
              state: [B, D_s] batch of robot arm x,y,z, gripper state. Do not use this when is None
              guidance_strength: strength of guidance
              clip_sample: whether to clip samples to [-1, 1]
              
              Return: [num_samples, D_a] samples
              """
              x = torch.randn(shape, device=self.device)
              if img_goal:
                     if img_begin != None and img_end != None:
                            cond = self.frame_diff_encoder.embed_frame(img_begin, img_end)
                     else:
                            raise ValueError("img_begin or img_end is None")
              else:
                     cond = self.lang_encoder.embed_text(lang) if self.lang_encoder is not None else lang     
                 
              cond = cond.repeat(x.shape[0], 1)
              imgs = imgs.repeat(x.shape[0], *((1,) * (len(imgs.shape) - 1)))
              state = state.repeat(x,shape[0], 1)
              
              # for t in reversed(range(self.num_timesteps)):
              #        t_tensor = torch.tensor([t]).unsqueeze(0).repeat(x.shape[0], 1).to(self.device)
                     # noise_pred = self.policy(x, t_tensor, imgs, cond, state)
              #        x = self.p_sample(x, torch.full((shape[0], 1), t, device=self.device), noise_pred, clip_sample=clip_sample)
              
              # if not hasattr(self, 'wrapped_policy'):
              

              self.wrapped_model(imgs, cond, state)
              x_init = torch.randn(shape).to(self.device)
              T = torch.linspace(0, 1, self.num_timesteps)  # sample times
              step_size = 1 / T
              
              solver = ODESolver(velocity_model=self.wrapped_policy)  # create an ODESolver class
              sol = solver.sample(time_grid=T, x_init=x_init, method='midpoint', step_size=step_size, return_intermediates=False)  # sample from the model
              return sol
       

       @torch.no_grad()
       def get_action(self, imgs, lang, state=None, num=1, t=1, k=0.25, clip_sample=True, img_begin=None, img_end=None, img_goal=False):
              if not isinstance(imgs, torch.Tensor):
                # transform lists to torch.Tensor
                imgs = torch.stack([self.transform(Image.fromarray(frame).convert("RGB")) for frame in imgs]).unsqueeze(0).unsqueeze(0).to('cuda')
              else:
                imgs = imgs.to('cuda')

              B, F, V, C, H, W = imgs.shape
              try:
                   s_dim = self.policy.s_dim
              except:
                   s_dim = self.policy.module.s_dim
              state = torch.from_numpy(state.astype(np.float32)).view(-1, s_dim) if state is not None else None
              state = ((state - self.s_mean) / self.s_std).to('cuda') if state is not None else None
            
              try:
                   output_dim = self.policy.output_dim
              except:
                   output_dim = self.policy.module.output_dim
              
              # use img goal or language goal
              if img_goal:
                     if img_begin != None and img_end != None:
                            if len(img_begin.shape) == 3:
                                   img_begin_pools = img_begin.unsqueeze(0).repeat(B, 1, 1, 1)
                            elif len(img_begin.shape) == 4 and img_begin.shape[0] == B:
                                   img_begin_pools = img_begin
                            else:
                                   raise ValueError(f"Please check the shape of img_begin: {img_begin.shape}")
                            img_end_pools = img_end.unsqueeze(0).repeat(B, 1, 1, 1) 
                            action = self.ddpm_sampler((B, output_dim), imgs,lang,state, clip_sample=clip_sample,img_begin=img_begin_pools,img_end=img_end_pools, img_goal=True).detach().cpu()
                            action = action.view(B, -1, 7)
                     else:
                            raise ValueError("img_begin or img_end is None")
              else:
                     action = self.ddpm_sampler((B, output_dim), imgs, [lang] * B, state, clip_sample=clip_sample).detach().cpu()
                     action = action.view(B, -1, 7)

              
              B, N, D_a = action.shape
              a_max = self.a_max.repeat(B, N, 1)
              a_min = self.a_min.repeat(B, N, 1)
              a_mean = self.a_mean.repeat(B, N, 1)
              a_std = self.a_std.repeat(B, N, 1)
              
              action = (action + 1) * (a_max - a_min) / 2 + a_min
              action = self.get_ac_action(action.numpy(), t, k)
              # action = action * a_std + a_mean
              return action
       
       @torch.no_grad()
       def forward_loss_test(self, imgs, lang, action_gt, img_begin=None, img_end=None, img_goal=False):
              imgs = torch.stack([self.transform(Image.fromarray(frame).convert("RGB")) for frame in imgs]).unsqueeze(0).unsqueeze(0).to('cuda')

              action_gt = torch.from_numpy(action_gt.astype(np.float32)).unsqueeze(0).to('cuda')

              text_emb = self.lang_encoder.embed_text([lang]).to(imgs.device).detach()

              loss = self.policy_loss(action_gt, imgs, text_emb, None)

              return loss

if __name__ == '__main__':
       timesteps = 100
       schedule = SCHEDULE['linear']
       print(schedule(timesteps)[0])

       